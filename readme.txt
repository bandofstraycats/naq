Repository for Approximate Smooth Kernel Modified Policy Iteration:
- kernel_vi.py - Approximate Smooth Kernel MPI
- gridworld_mpd.py - GridWorld domain
- kernel.py - Kernel definition
- plot.R - Plot performance metrics across runs
- plot.py - Plot performance metrics of one run
- kernel_vi.sh - Generates runs across random seeds
Requirement: Python numpy package

Generates 30 runs of Approximate Smooth Kernel Value Iteration across random seeds,
Saves performance metrics across iterations into 'export' directory for each seed:
`./kernel_vi.sh 0 30`

Example of a run of Value Iteration:
`python kernel_vi.py --plan plans/plan7.txt --random-slide 0.15
                    --max-iter 20 --m 1
                    --opt-v plans/opt_v7_rew_5_rs_0.15.txt
                    --plot metrics.png`

Example of a run of Kernel Value Iteration:
`python kernel_vi.py --plan plans/plan7.txt --random-slide 0.15
                    --max-iter 20 --m 1 --kernel --kernel-type ntk
                    --opt-v plans/opt_v7_rew_5_rs_0.15.txt
                    --plot metrics.png`

Description of parameters is provided in the help message:
`python kernel_vi.py --help`

Plot the performance across runs:
- requirements: install.packages(c('ggplot2', 'reshape2', 'dplyr'))
Generates plot.pdf in the repository directory:
`Rscript plot.R`
