# General overnight acquisition round 02

Kernel `nothingnessvoid/active-molgap-general-overnight-fetch-r02` targets
500,000 additional in-domain PubChemQC rows with a new seed/window schedule.
It mounts the complete 938,824-row accepted acquisition inventory as exclusions.

The run is divided into five 100,000-row chunks. Each chunk has an atomic
progress file, CSV hash, report, and independently downloadable ZIP. Any rows
that overlap concurrently running complementary rounds 08-09 must be removed
during post-run acceptance.
