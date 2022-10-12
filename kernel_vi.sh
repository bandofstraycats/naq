#!/usr/bin/env bash

DRY_RUN=$1
NB_SEED=$2

MAX_ITER=20
LOG_FREQ=1
M=1
S=1
kernel_type=ntk
beta=0.1

PLAN=plans/plan0.txt
OPT_V=plans/opt_v0_rew_5_rs_0.15.txt
KERNEL_OPT_V=plans/kernel_${kernel_type}_beta_${beta}_opt_v0_rew_5_rs_0.15.txt
EXPORT_DIR=export

if [ $DRY_RUN -eq 0 ]
then
    echo_and_run() { echo "$@" ; "$@" ; }
else
    echo_and_run() { echo "$@" ; }
fi

if [ ! -d "$EXPORT_DIR" ]; then
  mkdir $EXPORT_DIR
fi

seeds=($(shuf -i 0-1000 -n $NB_SEED))

for seed in "${seeds[@]}"
do
  export RND_SEED=$seed

  export_file=${EXPORT_DIR}/seed_${seed}_approx_vi.csv
  echo_and_run python kernel_vi.py --plan $PLAN --opt-v $OPT_V --random-slide 0.15 \
                                   --max-iter $MAX_ITER --m $M --s $S \
                                   --soft --beta $beta \
                                   --export $export_file \
                                   --log-freq $LOG_FREQ

  export_file=${EXPORT_DIR}/seed_${seed}_approx_kernel_${kernel_type}_vi.csv
  echo_and_run python kernel_vi.py --plan $PLAN --opt-v $KERNEL_OPT_V --random-slide 0.15 \
                                   --max-iter $MAX_ITER --m $M --s $S \
                                   --kernel --kernel-type=$kernel_type \
                                   --soft --beta $beta \
                                   --export $export_file \
                                   --log-freq $LOG_FREQ

done
