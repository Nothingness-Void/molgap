#!/bin/sh
#PBS -l select=1:ncpus=16:mpiprocs=1:ompthreads=16
#PBS -l walltime=120:00:00

if [ ! -z "${PBS_O_WORKDIR}" ]; then
  cd "${PBS_O_WORKDIR}"
  WORK=/lwork/users/${USER}/${PBS_JOBID}/gaussian
else
  WORK=/gwork/users/${USER}/tmp.$$
fi

if [ ! -d ${WORK} ]; then
  mkdir ${WORK}
fi

. /apl/gaussian/16c02/g16/bsd/g16.profile
export LANG=C
export GAUSS_SCRDIR=${WORK}

NCPUS=16
export GAUSS_CDEF=`/apl/gaussian/16c02/rccs/cpu.pl -n $NCPUS`

INP='Spiro-OMeTAD.gjf.ap'
ORG=${INP%.ap}
MOL=${ORG%.*}
OUT=${MOL}.out

g16 < ${INP} >& ${OUT}

if [ -d ${WORK} ]; then
  /bin/rm -rf ${WORK}
fi
exit 0
