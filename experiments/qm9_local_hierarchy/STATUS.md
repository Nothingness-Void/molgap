# Local hierarchy screen status

The Track C seed-42 screen was released to SCNet Kunshan from source commit
`e9e7e6a` on 2026-09-08. Mechanical state and terminal metrics belong in this
file and the adjacent launch record; scientific interpretation belongs in a
dated decision after acceptance.

- DCU preflight job: `121316216`
- CPU cache job: `121316249`
- dependent training job: `121316265`
- remote root:
  `/public/home/scnaqkfcy3/molgap-trackc-qm9-local-hierarchy-e9e7e6a`
- initial state: preflight/cache pending for priority; training pending on
  `afterok` dependencies
- official PCQM roles and QM9 held-out role: not read

An earlier submission attempt from `d4c0548` produced no job IDs because all
three resource requests exceeded Kunshan's `DefMemPerCPU=3569M` ratio. Commit
`e9e7e6a` corrected resource declarations without changing the scientific
contract.
