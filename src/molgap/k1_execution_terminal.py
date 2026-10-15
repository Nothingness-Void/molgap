"""K1 compatibility entry point for the shared excluded diagnostic closure."""


def close_execution(*, repo_root, output_root, analysis_path, summary_path, decision_path, finalized_at):
    from . import k1_infrastructure_comparison as comparison
    from .execution_terminal import close_execution as close_shared
    return close_shared(repo_root=repo_root, output_root=output_root, analysis_path=analysis_path,
        summary_path=summary_path, decision_path=decision_path, finalized_at=finalized_at,
        comparison=comparison, model_identity="neural-atom-k1-reference-192",
        selection_semantics="best-development-live")
