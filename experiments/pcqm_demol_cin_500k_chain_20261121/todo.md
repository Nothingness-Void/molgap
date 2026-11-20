# To do (infrastructure), 2026-11-21

- Missing RML command: append an action (for example A002 for a new run id or attempt) to an already published prospective trajectory.
  `plan` refuses an existing trajectory id and no other command adds an action. Until it exists, a changed run needs a new trajectory,
  as done here: `prospective_j1_chain` and `prospective_j2_chain` supersede the stand-alone jobs 2077090-2077094 (cancelled; not in the
  queue by `jobinfo -c` on 2026-11-20). Their old trajectories stay unchanged.
- This note is not in `protocol.md` on purpose: `protocol.md` is hash-bound in the five trajectories' frozen decision state.
