#!/bin/bash
# Submit all 10 Gaussian B3LYP/6-31G(d) opt+freq jobs
# Small molecules: 8 cores (default) is sufficient
# Large molecules (>40 atoms): use 16 cores for more memory

# Small molecules (<=30 heavy atoms) - default 8 cores
g16sub BCP.gjf
g16sub CBP.gjf
g16sub Coumarin-6.gjf
g16sub mCP.gjf

# Medium molecules (30-40 heavy atoms) - 16 cores
g16sub -np 16 TPBi.gjf
g16sub -np 16 NPB.gjf

# Large molecules (>40 heavy atoms) - 16 cores, extended time
g16sub -np 16 --walltime 120:00:00 DPEPO.gjf
g16sub -np 16 --walltime 120:00:00 CzSi.gjf
g16sub -np 16 --walltime 120:00:00 TCTA.gjf
g16sub -np 16 --walltime 120:00:00 Spiro-OMeTAD.gjf

echo 'All 10 jobs submitted. Use jobinfo to check status.'
