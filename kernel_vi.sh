#!/usr/bin/env bash

DRY_RUN=$1
NB_SEED=$2

MAX_ITER=20
M=1
PLAN=plans/plan7.txt
OPT_V=plans/opt_v7_rew_5_rs_0.15.txt
KERNEL_OPT_V=plans/kernel_opt_v7_rew_5_rs_0.15.txt
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

  export_file=${EXPORT_DIR}/seed_${seed}_vi.csv
  echo_and_run python kernel_vi.py --plan $PLAN --opt-v $OPT_V --random-slide 0.15 \
                                   --max-iter $MAX_ITER --m $M \
                                   --export-errors $export_file

  kernel_type=identity
  beta=0.1
  export_file=${EXPORT_DIR}/seed_${seed}_kernel_pi.csv
  echo_and_run python kernel_vi.py --plan $PLAN --opt-v $OPT_V --random-slide 0.15 \
                                  --max-iter $MAX_ITER --m inf \
                                  --kernel --kernel-type=$kernel_type \
                                  --soft --beta $beta \
                                  --save-v $KERNEL_OPT_V --export-errors $export_file

  export_file=${EXPORT_DIR}/seed_${seed}_kernel_vi.csv
  echo_and_run python kernel_vi.py --plan $PLAN --opt-v $KERNEL_OPT_V --random-slide 0.15 \
                                   --max-iter $MAX_ITER --m $M \
                                   --kernel --kernel-type=$kernel_type \
                                   --soft --beta $beta \
                                   --export-errors $export_file

done
