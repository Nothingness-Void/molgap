# Version 2 cancellation and unchanged retry

On 2026-09-28 the controller reconciled kernel `136108187`, version 2, as
`CANCEL_ACKNOWLEDGED`. The retained [platform observation](platform_observation.json)
contains no failure message, downloadable output or log. The last retained
Luna observation was QUEUED. Cancellation cause, actual worker execution, role
access and native cost cannot be established from absent artifacts; none is
converted to a zero measurement or a scientific result.

The user explicitly authorized one resubmission. Version 3 is a separate
physical attempt of the unchanged dual-arm [protocol](../protocol.md), not a
scientific successor or a checkpoint continuation. Version 2 evidence and all
original prospective files are preserved. No recoverable checkpoint was exposed
by the API. The cancelled attempt has no scientific or replay-ready claim.

The monitor now maps the observed `CANCEL_ACKNOWLEDGED` terminal to `CANCELLED`
while retaining its raw API spelling; an unacknowledged cancellation request
remains UNKNOWN. This does not modify the remote training recipe.
