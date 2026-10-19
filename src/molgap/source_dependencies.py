"""Deterministic source closure from reviewed roots, without importing code.

The resulting list is input to the existing immutable packager. Registry roots
remain reviewed; AST discovery removes per-addon transitive file bookkeeping.
Dynamic imports are reported and still require registry/runtime qualification.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path

from .experiment_package import _allowlist, _source


@dataclass(frozen=True)
class DependencyClosure:
    roots: tuple[str, ...]
    source_files: tuple[str, ...]
    external_modules: tuple[str, ...]
    dynamic_imports: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    def to_dict(self):
        return asdict(self)


def _module(path: str) -> str | None:
    if not path.startswith("src/molgap/") or not path.endswith(".py"):
        return None
    name = path[4:-3].replace("/", ".")
    return name[:-9] if name.endswith(".__init__") else name


def _module_path(root: Path, name: str) -> str | None:
    if name != "molgap" and not name.startswith("molgap."):
        return None
    base = "src/" + name.replace(".", "/")
    # Match Python's FileFinder: a package wins over a same-named module.
    candidates = (base + "/__init__.py", base + ".py")
    return next((path for path in candidates if (root / path).is_file()), None)


def _package_paths(root: Path, name: str):
    pieces = name.split(".")
    for length in range(1, len(pieces)):
        path = "src/" + "/".join(pieces[:length]) + "/__init__.py"
        if (root / path).is_file():
            yield path


def _relative_module(path: str, level: int, name: str | None) -> str:
    module = _module(path)
    if module is None:
        raise ValueError(f"Relative import outside a reviewed package: {path}")
    package = module if path.endswith("/__init__.py") else module.rpartition(".")[0]
    pieces = package.split(".")
    if level > len(pieces):
        raise ValueError(f"Relative import escapes molgap: {path}")
    prefix = pieces[:len(pieces) - level + 1]
    return ".".join((*prefix, *((name or "").split(".") if name else ())))


def _import_nodes(tree, *, package_startup):
    if not package_startup:
        yield from ast.walk(tree)
        return
    # Importing a package does not call its lazy public API functions. Treat
    # only their startup imports as roots; explicitly selected submodules are
    # followed normally, including all function-local dependencies.
    pending = [tree]
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__getattr__":
            continue
        yield node
        pending.extend(ast.iter_child_nodes(node))


def discover_source_dependencies(repo_root, roots, *, require_static=False) -> DependencyClosure:
    """Follow all static imports, including imports inside functions/branches.

    Lazy package export functions are not selected by importing the package;
    select their actual implementation module as a root when using that API.
    Only ``src/molgap`` imports become additional source files. Third-party
    modules are named in the report, never copied from the host environment.
    Dynamic names cannot be inferred safely: optional strict mode rejects them.
    Existing source confinement/link/secret boundaries are applied to every file.
    """
    root = Path(repo_root).resolve()
    seeds = tuple(sorted(_allowlist(roots)))
    pending, seen, external, dynamic, edges = list(seeds), set(), set(), set(), set()
    for seed in seeds:
        module = _module(seed)
        if module is not None:
            pending.extend(_package_paths(root, module))
    parsed, import_aliases, module_aliases = {}, {}, {}

    def tree_for(relative):
        if relative not in parsed:
            _allowlist([relative])
            parsed[relative] = ast.parse(_source(root, relative).read_bytes(), filename=relative)
        return parsed[relative]

    def import_functions_for(relative):
        if relative not in import_aliases:
            import_aliases[relative] = {alias.asname or alias.name for statement in ast.walk(tree_for(relative))
                if isinstance(statement, ast.ImportFrom) and statement.module == "importlib"
                for alias in statement.names if alias.name == "import_module"}
        return import_aliases[relative]

    def referenced_module(relative, node):
        if relative not in module_aliases:
            aliases = {}
            for statement in ast.walk(tree_for(relative)):
                if isinstance(statement, ast.Import):
                    for alias in statement.names:
                        aliases[alias.asname or alias.name.split(".", 1)[0]] = (
                            alias.name if alias.asname else alias.name.split(".", 1)[0])
                elif isinstance(statement, ast.ImportFrom):
                    module = (_relative_module(relative, statement.level, statement.module)
                        if statement.level else statement.module)
                    if module:
                        for alias in statement.names:
                            aliases[alias.asname or alias.name] = module + "." + alias.name
            module_aliases[relative] = aliases
        if isinstance(node, ast.Name):
            return module_aliases[relative].get(node.id)
        if isinstance(node, ast.Attribute):
            parent = referenced_module(relative, node.value)
            return parent + "." + node.attr if parent else None
        return None

    def unresolved(importer, node, callee=None):
        # Report the source location/callee, never literal arguments which may
        # contain private configuration in a trusted caller's source.
        callee = callee or ast.unparse(node.func)
        dynamic.add(f"{importer}:{node.lineno}: {callee}(...) requires registry binding")

    def export_nodes(module, symbol):
        path = _module_path(root, module)
        if path is None or not path.endswith("/__init__.py"):
            return ()
        tree = tree_for(path)
        declared = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                declared.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                declared.update(alias.asname or alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.Assign):
                declared.update(target.id for target in node.targets if isinstance(target, ast.Name))
        if symbol in declared:
            return ()
        getter = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == "__getattr__"), None)
        return tuple((path, node) for node in ast.walk(getter)) if getter is not None else ()

    def include(importer, name, *, required=True):
        if name != "molgap" and not name.startswith("molgap."):
            external.add(name.split(".", 1)[0])
            return
        relative = _module_path(root, name)
        if relative is None:
            if required:
                raise ValueError(f"Missing internal dependency {name!r} imported by {importer}")
            return
        _allowlist([relative])
        _source(root, relative)
        edges.add((importer, relative))
        pending.extend((*_package_paths(root, name), relative))

    while pending:
        relative = pending.pop()
        if relative in seen:
            continue
        path = _source(root, relative)
        seen.add(relative)
        if path.suffix != ".py":
            continue
        tree = tree_for(relative)
        nodes = [(relative, node) for node in _import_nodes(tree,
                    package_startup=relative.endswith("/__init__.py"))]
        visited_nodes = set()
        while nodes:
            importer, node = nodes.pop()
            node_identity = (importer, id(node))
            if node_identity in visited_nodes:
                continue
            visited_nodes.add(node_identity)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    include(importer, alias.name)
            elif isinstance(node, ast.Attribute):
                module = referenced_module(importer, node.value)
                if module is not None and (module == "molgap" or module.startswith("molgap.")):
                    nodes.extend(export_nodes(module, node.attr))
            elif isinstance(node, ast.ImportFrom):
                name = (_relative_module(importer, node.level, node.module)
                        if node.level else node.module)
                if not name:
                    raise ValueError(f"Unresolved import at {importer}:{node.lineno}")
                include(importer, name)
                for alias in node.names:
                    if alias.name == "*":
                        # __all__ may be computed and may select lazy modules.
                        unresolved(importer, node, "from " + name + " import *")
                        nodes.extend(export_nodes(name, "*"))
                    else:
                        # A from-import may refer to a module or to a symbol;
                        # only actual module files add an edge.
                        include(importer, name + "." + alias.name, required=False)
                        if _module_path(root, name + "." + alias.name) is None:
                            nodes.extend(export_nodes(name, alias.name))
            elif isinstance(node, ast.Call):
                function = node.func
                import_functions = import_functions_for(importer)
                builtin = isinstance(function, ast.Name) and function.id == "__import__"
                is_dynamic = ((isinstance(function, ast.Name) and
                               (builtin or function.id in import_functions)) or
                              (isinstance(function, ast.Attribute) and function.attr == "import_module"))
                if not is_dynamic:
                    continue
                name = node.args[0] if node.args else next(
                    (item.value for item in node.keywords if item.arg == "name"), None)
                if isinstance(name, ast.Constant) and isinstance(name.value, str):
                    resolved = name.value
                    if name.value.startswith("."):
                        level = len(name.value) - len(name.value.lstrip("."))
                        package = node.args[1] if len(node.args) > 1 and not builtin else next(
                            (item.value for item in node.keywords if item.arg == "package"), None)
                        if (not isinstance(package, ast.Constant) or not isinstance(package.value, str) or
                                level > len(package.value.split("."))):
                            unresolved(importer, node)
                            continue
                        parts = package.value.split(".")
                        resolved = ".".join((*parts[:len(parts) - level + 1], name.value[level:]))
                    if builtin:
                        level = node.args[4] if len(node.args) > 4 else next(
                            (item.value for item in node.keywords if item.arg == "level"), None)
                        if level is not None:
                            if not isinstance(level, ast.Constant) or type(level.value) is not int or level.value < 0:
                                unresolved(importer, node)
                                continue
                            if level.value:
                                resolved = _relative_module(importer, level.value, name.value)
                    include(importer, resolved)
                    # __import__('molgap', fromlist=['some_module']) also
                    # loads literal submodules; symbol names remain optional.
                    fromlist = node.args[3] if builtin and len(node.args) > 3 else next(
                        (item.value for item in node.keywords if item.arg == "fromlist"), None)
                    if isinstance(fromlist, (ast.List, ast.Tuple)):
                        for item in fromlist.elts:
                            if isinstance(item, ast.Constant) and isinstance(item.value, str) and item.value != "*":
                                include(importer, resolved + "." + item.value, required=False)
                                if _module_path(root, resolved + "." + item.value) is None:
                                    nodes.extend(export_nodes(resolved, item.value))
                            else:
                                unresolved(importer, node)
                    elif fromlist is not None and not (isinstance(fromlist, ast.Constant) and fromlist.value is None):
                        unresolved(importer, node)
                else:
                    unresolved(importer, node)
    if require_static and dynamic:
        raise ValueError("Dynamic import requires reviewed registry roots and runtime checks: " +
                         "; ".join(sorted(dynamic)))
    return DependencyClosure(seeds, tuple(_allowlist(sorted(seen))), tuple(sorted(external)),
                             tuple(sorted(dynamic)), tuple(sorted(edges)))


def registered_dependencies(spec, repo_root, extras=(), *, require_static=False):
    """Reuse Spec/execution roots and resolve addon import closure automatically."""
    from .experiment_source_inventory import registered_source_files
    return discover_source_dependencies(repo_root, registered_source_files(spec, extras),
                                        require_static=require_static)


__all__ = ["DependencyClosure", "discover_source_dependencies", "registered_dependencies"]
