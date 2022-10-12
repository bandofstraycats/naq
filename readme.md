# Implementation of Approximate Smooth Kernel Value Iteration

## Repository
- kernel_vi.py - Approximate Smooth Kernel Value Iteration
- kernel.py - Kernel definition
- gridworld_mdp.py - GridWorld domain
- plot.py - Plot performance metrics of one run
- plot.R - Plot performance metrics across runs
- kernel_vi.sh - Generates multiple runs across random seeds

## Installation 

Requirements: Python, numpy, matplotlib

## Usage
Description of parameters is provided in the help message
`python kernel_vi.py --help`

## Example on a stochastic cliff walking problem

### Value Iteration
- Value Iteration 
`python kernel_vi.py --plan plans/plan0.txt --random-slide 0.15 --opt-v plans/opt_v0_rew_5_rs_0.15.txt --max-iter 20 --s 1 --plot metrics.png`
- Approximate Value Iteration with sampled Bellman operator at 10 states
`python kernel_vi.py --plan plans/plan0.txt --random-slide 0.15 --opt-v plans/opt_v0_rew_5_rs_0.15.txt --max-iter 100 --s 10 --log-freq 10 --plot metrics.png`

### Kernel Value Iteration
- Kernel Value Iteration with Neural Tangent Kernel
`python kernel_vi.py --plan plans/plan0.txt --random-slide 0.15 --opt-v plans/opt_v0_rew_5_rs_0.15.txt --max-iter 20 --s 1 --kernel --kernel-type ntk --plot metrics.png`
- Approximate Kernel Value Iteration with Neural Tangent Kernel and sampled Bellman operator at 10 states
`python kernel_vi.py --plan plans/plan0.txt --random-slide 0.15 --opt-v plans/opt_v0_rew_5_rs_0.15.txt --max-iter 100 --s 10 --log-freq 10 --kernel --kernel-type ntk --plot metrics.png`

### Aggregate performance metrics
Generates NUM_RUNS runs of Approximate Smooth Kernel Value Iteration across random seeds.
Saves performance metrics across iterations into 'export' directory for each seed.

`bash kernel_vi.sh 0 NUM_RUNS`

#### Plot performance metrics across runs
Requirements: R-project, install.packages(c('ggplot2', 'reshape2', 'dplyr'))

Reads 'export' directory from previous step and generates plot.pdf in the current directory
`Rscript plot.R`

## References
[1] [Smirnova, Elena. On Convergence of Neural asynchronous Q-iteration. EWRL, 2022.](https://ewrl.files.wordpress.com/2022/09/on_convergence_of_neural_asynchronous_q_iteration_final.pdf)
