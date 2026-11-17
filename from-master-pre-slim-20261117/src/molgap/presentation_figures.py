"""Build reproducible presentation figures from the frozen evidence pack."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class FigureTheme:
    """Colors shared by the Matplotlib and TikZ presentation figures."""

    name: str
    ink: str
    muted: str
    grid: str
    blue: str
    teal: str
    orange: str
    red: str
    purple: str
    neutral: str
    figure_bg: str
    axes_bg: str


LIGHT_THEME = FigureTheme(
    name="light",
    ink="#102A43",
    muted="#52606D",
    grid="#D9E2EC",
    blue="#1976A8",
    teal="#16817A",
    orange="#D9822B",
    red="#C05640",
    purple="#6B5B95",
    neutral="#8A9BA8",
    figure_bg="#FFFFFF",
    axes_bg="#FFFFFF",
)

# The deck uses a near-black navy background. These colors are intentionally
# brighter than the evidence-report palette so small labels stay legible when
# a figure is placed on a full-slide raster background.
DARK_THEME = FigureTheme(
    name="dark",
    ink="#F3F7FA",
    muted="#B9C7D4",
    grid="#2B3B4B",
    blue="#62B8FF",
    teal="#55D6C2",
    orange="#FFBD5A",
    red="#FF7568",
    purple="#C3A6FF",
    neutral="#8495A5",
    figure_bg="#080D13",
    axes_bg="#0E1721",
)


def _resolve_theme(theme: str | FigureTheme) -> FigureTheme:
    if isinstance(theme, FigureTheme):
        return theme
    themes = {LIGHT_THEME.name: LIGHT_THEME, DARK_THEME.name: DARK_THEME}
    try:
        return themes[theme.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown figure theme {theme!r}; choose 'light' or 'dark'.") from exc


def _configure_matplotlib(theme: FigureTheme) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 11,
            "axes.edgecolor": theme.grid,
            "axes.labelcolor": theme.ink,
            "axes.titlecolor": theme.ink,
            "xtick.color": theme.muted,
            "ytick.color": theme.muted,
            "text.color": theme.ink,
            "figure.facecolor": theme.figure_bg,
            "axes.facecolor": theme.axes_bg,
            "savefig.facecolor": theme.figure_bg,
        }
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _clean_axes(ax: plt.Axes, theme: FigureTheme) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(theme.grid)
    ax.spines["bottom"].set_color(theme.grid)
    ax.grid(axis="y", color=theme.grid, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def _annotate_bars(
    ax: plt.Axes,
    bars: Iterable,
    theme: FigureTheme,
    fmt: str = "{:.3f}",
) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.0015,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=9,
            color=theme.ink,
        )


def build_accuracy(evidence: dict, output_dir: Path, theme: FigureTheme) -> None:
    scopes = [("all", "Common"), ("ood1000", "OOD 1K"), ("p8_targeted_hard", "P8-hard")]
    methods = [
        ("routed_v4_500k", "Routed v4", theme.neutral),
        ("repaired_2m_equal_2d", "2M equal", theme.blue),
        ("repaired_2m_dense_2d", "2M dense", theme.orange),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(scopes))
    width = 0.24
    bar_groups = []
    for index, (key, label, color) in enumerate(methods):
        values = [evidence["accuracy"]["scopes"][scope]["methods"][key]["average_mae_eV"] for scope, _ in scopes]
        bars = ax.bar(x + (index - 1) * width, values, width, label=label, color=color)
        _annotate_bars(ax, bars, theme)
        bar_groups.append((bars, values))
    ax.legend(frameon=False, ncols=3, loc="upper left")
    # A light outline makes the best value in each scope immediately visible
    # without changing the model colors used by the legend.
    for scope_index in range(len(scopes)):
        best = min(values[scope_index] for _, values in bar_groups)
        for bars, values in bar_groups:
            if np.isclose(values[scope_index], best, atol=1e-12):
                bars[scope_index].set_edgecolor(theme.ink)
                bars[scope_index].set_linewidth(2.4)
                bars[scope_index].set_zorder(3)
    ax.set_xticks(x, [label for _, label in scopes])
    ax.set_ylabel("Average MAE (eV), lower is better")
    ax.set_title("Track A: the repaired-2M pure-2D presets improve every frozen scope")
    ax.set_ylim(0, 0.14)
    _clean_axes(ax, theme)
    fig.text(0.01, 0.01, "Same-molecule external comparison; 1,973 paired rows. Source: presentation_evidence.json.", fontsize=9, color=theme.muted)
    _save(fig, output_dir, "07_track_a_accuracy")


def build_r2(evidence: dict, output_dir: Path, theme: FigureTheme) -> None:
    scopes = [("all", "Common"), ("ood1000", "OOD 1K"), ("p8_targeted_hard", "P8-hard")]
    methods = [
        ("routed_v4_500k", "Routed v4", theme.neutral),
        ("repaired_2m_equal_2d", "2M equal", theme.blue),
        ("repaired_2m_dense_2d", "2M dense", theme.orange),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(scopes))
    width = 0.24
    for index, (key, label, color) in enumerate(methods):
        values = [evidence["accuracy"]["scopes"][scope]["methods"][key]["average_r2"] for scope, _ in scopes]
        bars = ax.bar(x + (index - 1) * width, values, width, label=label, color=color)
        _annotate_bars(ax, bars, theme, "{:.3f}")
    ax.set_xticks(x, [label for _, label in scopes])
    ax.set_ylabel("Average R2, higher is better")
    ax.set_title("Track A: correlation stays strong under OOD and targeted-hard evaluation")
    ax.set_ylim(0.90, 0.97)
    ax.legend(
        frameon=False,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        borderaxespad=0.0,
    )
    _clean_axes(ax, theme)
    fig.text(0.01, 0.01, "Average across HOMO, LUMO and Gap; same frozen paired evaluation.", fontsize=9, color=theme.muted)
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    _save(fig, output_dir, "08_track_a_r2")


def _load_latency(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_latency(latency_dir: Path, output_dir: Path, theme: FigureTheme) -> None:
    dense = _load_latency(latency_dir / "repaired_2m_dense_2d_local.json")
    equal = _load_latency(latency_dir / "repaired_2m_equal_2d_local.json")
    routed = _load_latency(latency_dir / "routed_gps7_gps9_schnet_500k_v4_local.json")
    forced = _load_latency(latency_dir / "routed_gps7_gps9_schnet_500k_v4_routed_inputs_local.json")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
    names = ["Routed v4\nbase", "2M equal\n2 GPS", "2M dense\n3 GPS"]
    throughput = [routed["results"][2]["median_molecules_per_s"], equal["results"][2]["median_molecules_per_s"], dense["results"][2]["median_molecules_per_s"]]
    bars = axes[0].bar(names, throughput, color=[theme.neutral, theme.blue, theme.orange], width=0.62)
    for bar, value in zip(bars, throughput):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 35, f"{value:,.0f}", ha="center", fontsize=10, color=theme.ink)
    axes[0].set_ylabel("Molecules / second")
    axes[0].set_title("Batch 64 throughput")
    axes[0].set_ylim(0, max(throughput) * 1.2)
    _clean_axes(axes[0], theme)

    names = ["Routed v4\nbase", "Routed v4\nforced route", "2M equal", "2M dense"]
    ms = [routed["results"][2]["median_ms_per_molecule"], forced["results"][1]["median_ms_per_molecule"], equal["results"][2]["median_ms_per_molecule"], dense["results"][2]["median_ms_per_molecule"]]
    bars = axes[1].bar(names, ms, color=[theme.neutral, theme.red, theme.blue, theme.orange], width=0.62)
    for bar, value in zip(bars, ms):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.9, f"{value:.2f}", ha="center", fontsize=10, color=theme.ink)
    axes[1].set_ylabel("Warm end-to-end ms / molecule")
    axes[1].set_title("New-SMILES latency")
    axes[1].set_ylim(0, max(ms) * 1.25)
    _clean_axes(axes[1], theme)
    fig.suptitle("Inference cost: pure 2D avoids conformer construction", fontsize=17, x=0.04, ha="left", color=theme.ink)
    fig.text(0.04, 0.01, "RTX 5060; warm timings include parsing and graph construction, exclude checkpoint load. Forced route is a worst-case routed-v4 input suite.", fontsize=9, color=theme.muted)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    _save(fig, output_dir, "09_inference_cost")


def build_corpus(evidence: dict, output_dir: Path, theme: FigureTheme) -> None:
    corpus = evidence["corpus"]
    fig, ax = plt.subplots(figsize=(10, 5.7))
    labels = ["Reconciled source\nledger", "Materialized\nrepaired-2M", "Immutable targeted\nsubset"]
    values = [corpus["ledger_rows_reconciled"], corpus["rows"], corpus["immutable_targeted_rows"]]
    colors = [theme.purple, theme.blue, theme.orange]
    bars = ax.barh(labels, values, color=colors, height=0.52)
    for bar, value in zip(bars, values):
        ax.text(value + 50000, bar.get_y() + bar.get_height() / 2, f"{value:,.0f}", va="center", fontsize=11, color=theme.ink)
    ax.set_xlim(0, 3_900_000)
    ax.set_xlabel("Rows")
    ax.set_title("Data foundation: reconcile broadly, train on a fixed materialized corpus")
    _clean_axes(ax, theme)
    ax.grid(axis="x", color=theme.grid, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.text(0.01, 0.01, "B3LYP/6-31G*, gas phase; targets are Kohn-Sham HOMO, LUMO and Gap in eV. Elements: CHONSFCl.", fontsize=9, color=theme.muted)
    _save(fig, output_dir, "01_corpus_profile")


def build_geometry(evidence: dict, output_dir: Path, theme: FigureTheme) -> None:
    geom = evidence["geometry_leverage"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1.1, 1]})
    labels = ["ETKDG -> DFT\ngeometry", "Conformer average\nK=1 -> K=6"]
    values = [abs(geom["etkdg_to_dft_geometry_average_mae_gain_eV"]), abs(geom["conformer_averaging_k1_to_k6_gain_eV"])]
    bars = ax.bar(labels, values, color=[theme.teal, theme.blue], width=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0006, f"{value:.5f} eV", ha="center", fontsize=10, color=theme.ink)
    ax.set_ylabel("Average MAE gain magnitude (eV)")
    ax.set_ylim(0, 0.028)
    ax.set_title("Geometry quality is a real lever")
    _clean_axes(ax, theme)
    ax2.axis("off")
    ax2.text(0.02, 0.86, "What the result means", fontsize=15, weight="bold", color=theme.ink)
    ax2.text(0.02, 0.70, "Sampling more ETKDG conformers\nremoves only about one fifth\nof the geometry gap.", fontsize=14, color=theme.ink, va="top")
    ax2.text(0.02, 0.36, "Decision boundary", fontsize=13, weight="bold", color=theme.orange)
    ax2.text(0.02, 0.22, "Use ETKDG consistently for the\ncurrent production contract;\nrevisit relaxed geometries only\nwhen a new compute budget exists.", fontsize=12, color=theme.muted, va="top")
    fig.suptitle("3D was paused for evidence reasons, not because geometry is irrelevant", fontsize=17, x=0.04, ha="left", color=theme.ink)
    fig.text(0.04, 0.01, "Scope caveat: this lever was measured on the QM9 architecture screen, not on PubChemQC.", fontsize=9, color=theme.muted)
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    _save(fig, output_dir, "10_geometry_leverage")


def build_manifest(output_dir: Path, theme: FigureTheme) -> None:
    files = sorted(
        p.name
        for p in output_dir.iterdir()
        if p.is_file() and p.suffix in {".pdf", ".png", ".svg"} and not p.stem.endswith("_inspect")
    )
    (output_dir / "figure_manifest.json").write_text(
        json.dumps({"status": "complete", "theme": theme.name, "files": files}, indent=2),
        encoding="utf-8",
    )


def compile_tikz(source_dir: Path, output_dir: Path, theme: FigureTheme) -> list[str]:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        candidate = Path.home() / "AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdflatex.exe"
        pdflatex = str(candidate) if candidate.exists() else None
    if pdflatex is None:
        raise RuntimeError("pdflatex was not found; install MiKTeX or pass a LaTeX executable on PATH")
    output_dir.mkdir(parents=True, exist_ok=True)
    stems = []
    for tex in sorted(source_dir.glob("*.tex")):
        if tex.name == "figure_style.tex":
            continue
        if theme.name == "dark":
            # Compile through a tiny wrapper so the same source files can use
            # the dark palette without duplicating every TikZ diagram.
            command = [
                pdflatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={output_dir}",
                f"-jobname={tex.stem}",
                f"\\def\\DARKTHEME{{}}\\input{{{tex.name}}}",
            ]
        else:
            command = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", f"-output-directory={output_dir}", tex.name]
        result = subprocess.run(command, cwd=source_dir, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            raise RuntimeError(f"LaTeX failed for {tex.name}:\n{result.stdout[-4000:]}\n{result.stderr[-1000:]}")
        stems.append(tex.stem)
        for suffix in (".aux", ".log"):
            generated = output_dir / f"{tex.stem}{suffix}"
            if generated.exists():
                generated.unlink()
    return stems


def build_all(
    evidence_path: Path,
    latency_dir: Path,
    output_dir: Path,
    source_dir: Path,
    theme: str | FigureTheme = "light",
) -> None:
    selected_theme = _resolve_theme(theme)
    output_dir = output_dir.resolve()
    _configure_matplotlib(selected_theme)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    build_corpus(evidence, output_dir, selected_theme)
    build_accuracy(evidence, output_dir, selected_theme)
    build_r2(evidence, output_dir, selected_theme)
    build_latency(latency_dir, output_dir, selected_theme)
    build_geometry(evidence, output_dir, selected_theme)
    compile_tikz(source_dir, output_dir, selected_theme)
    build_manifest(output_dir, selected_theme)
