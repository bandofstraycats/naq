from copy import copy, deepcopy

import os
RND_SEED = int(os.environ.get('RND_SEED', 123))
print('seed', RND_SEED)
import numpy as np
np.random.seed(RND_SEED)

from gridworld_mdp import GridworldMDP
from kernel import get_kernel

def modified_policy_iteration(mdp, epsilon=None, max_iter=None,
                              nb_samples=None, m=1,
                              is_kernel=False, kernel_type=None,
                              is_soft=False, beta=None,
                              V_opt=None,
                              log_freq=1):

    def Q_V(V):
        q_v = mdp.R + mdp.gamma * np.matmul(mdp.P, V)
        return q_v

    def bellman_op(V, R_pi, P_pi, m, is_soft=False, beta=None):
        if m == 0:
            return V

        TS = R_pi + mdp.gamma * np.matmul(P_pi, bellman_op(V, R_pi, P_pi, m-1))
        if is_soft:
            TS = (1 - beta) * V + beta * TS
        return TS

    def empirical_bellman_op(R, V, is_soft=False, beta=None):
        TS = R + mdp.gamma*V
        if is_soft:
            TS = (1 - beta) * V + beta * TS
        return TS

    def empirical_value_iteration(V, pi, is_soft=False, beta=None):
        V_update = deepcopy(V)
        for s in range(mdp.nS):
            V_update[s] = 0
            playing_policy = pi[s, :]

            sampled_actions = np.random.choice(mdp.nA, size=nb_samples, p=playing_policy)
            for action in sampled_actions:
                next_state = np.random.choice(mdp.nS, size=1, p=mdp.P[s, action, :])
                V_update[s] += empirical_bellman_op(mdp.R[s, action], V[next_state],
                                                    is_soft=is_soft, beta=beta)
            V_update[s] /= nb_samples

        return V_update

    def value_iteration(V, pi, m, is_soft=False, beta=None):
        P_pi = np.zeros((mdp.nS, mdp.nS))
        R_pi = np.zeros(mdp.nS)
        for s in range(mdp.nS):
            P_pi[s] = np.matmul(pi[s,:], mdp.P[s, :, :])
            R_pi[s] = np.matmul(pi[s,:], mdp.R[s, :])

        if np.isinf(m):
            # v^\pi_k
            PE = np.matmul(np.linalg.inv(np.identity(mdp.nS) - mdp.gamma * P_pi), R_pi)
        else:
            PE = bellman_op(V, R_pi, P_pi, m, is_soft=is_soft, beta=beta)
        return PE

    def greedy_policy_Q(Q):
        policy = np.zeros([mdp.nS, mdp.nA])
        for s in range(mdp.nS):
            policy[s, np.argmax(Q[s,:])] = 1.0
        return policy

    def greedy_step(V):
        return greedy_policy_Q(Q_V(V))

    V = np.zeros(mdp.nS)
    pi = (1./mdp.nA) * np.ones_like(mdp.R)
    V_k, pi_k = list(), list()
    metrics = {'beta': [],
              'max_opt_v': [], 'avg_opt_v': []}

    if is_kernel:
        kernel = get_kernel(kernel_type, mdp.state_vars, mdp.nS)
        kernel_optimality = (1. / (1 - mdp.gamma)) * \
                            np.max(np.abs(np.matmul(np.eye(mdp.nS) - kernel,
                                                    value_iteration(V_opt, greedy_step(V_opt), 1,
                                                                    is_soft=is_soft, beta=beta)
                                                    )))
        print("Kernel optimality upper bound")
        print(kernel_optimality)

    n = 0
    while True:
        # Stopping condition
        delta = 0.0
        # Policy Evaluation
        if nb_samples:
            V_pi_eval = empirical_value_iteration(V, pi, is_soft=is_soft, beta=beta)
        else:
            V_pi_eval = value_iteration(V, pi, m, is_soft=is_soft, beta=beta)

        # Kernel
        if is_kernel:
            V_pi_eval = np.matmul(kernel, V_pi_eval)

        # Policy Improvement
        pi = greedy_step(V_pi_eval)

        # Parameters
        metrics['beta'].append(beta if beta else 1.0)

        # Error to optimality
        max_opt_error = np.max(np.abs(V_opt - V_pi_eval)) if V_opt is not None else 0.0
        avg_opt_error = np.average(np.abs(V_opt - V_pi_eval)) if V_opt is not None else 0.0
        metrics['max_opt_v'].append(max_opt_error)
        metrics['avg_opt_v'].append(avg_opt_error)

        # Calculate difference between subsequent iterates
        delta = max(delta, np.max(np.abs(V_pi_eval - V)))
        # Update the value function
        V = deepcopy(V_pi_eval)

        # V_k
        v = deepcopy(np.transpose(np.reshape(V, [mdp.nS, 1])))
        V_k.append(v)
        # pi_k
        pi_k.append(pi)

        # Iteration
        n += 1

        if n % log_freq == 0:
            print('N, ', n, ', current delta', delta)
            out_str = ""
            for key in metrics.keys():
                out_str += "{}:{:.2f}\t".format(key, np.average(metrics[key][-log_freq:]))
            print(out_str)
        # Check stopping criteria
        if (max_iter and n >= max_iter) or (epsilon and delta < epsilon):
            print("Finish after iterations,", n)
            break

    return pi, V, Q_V(V), V_k, pi_k, metrics

if __name__ == "__main__":
    # execute only if run as a script
    import argparse

    parser = argparse.ArgumentParser(description='Approximate Smooth Kernel Modified Policy Iteration')
    parser.add_argument('--plan', help='Plan file', default='plans/plan7.txt', required=True)
    # random slide
    parser.add_argument('--random-slide', help='Random slide for GridWorld', default=0.0, type=float)
    # kernel
    parser.add_argument('--kernel', dest='is_kernel', action='store_true', help='Set Kernel VI')
    parser.add_argument('--kernel-type', help='Kernel type for Kernel VI', default='linear', type=str)
    # soft bellman operator
    parser.add_argument('--soft', dest='is_soft', action='store_true', help='Set Soft Bellman operator')
    parser.add_argument('--beta', help='Beta for soft Bellman operator', default=1.0, type=float)
    # stopping
    parser.add_argument('--epsilon', help='Stopping criteria: Epsilon difference between subsequent iterates', default=None, type=float)
    parser.add_argument('--max-iter', help='Stopping criteria: Maximum number of iterations', default=None, type=int)
    # m
    parser.add_argument('--m', help='Number of applications of Bellman operator', default=1, type=float)
    # number of states to sample
    parser.add_argument('--s', help='Number of samples to evaluate Bellman operator', default=None, type=int)
    # opt v
    parser.add_argument('--opt-v', help='Optimal value function file', default=None)
    # discount factor
    parser.add_argument('--gamma', help='Discount factor', default=0.9, type=float)
    # plots
    parser.add_argument('--plot', default=None, help='Save plot of metrics to png file')
    parser.add_argument('--export', default=None, help='Export metrics to csv file')
    # save v and pi
    parser.add_argument('--save-v', help='Save value function to file', default=None)
    parser.add_argument('--save-pi', help='Save policy to file', default=None)
    parser.add_argument('--save-q', help='Save Q-function to file', default=None)
    # log
    parser.add_argument('--log-freq', help='Logging frequency', default=1, type=int)

    args = parser.parse_args()
    print(args)

    mdp = GridworldMDP(plan_file=args.plan, gamma=args.gamma, random_slide=args.random_slide)
    V_opt = np.loadtxt(args.opt_v) if args.opt_v else None
    pi, V, Q, V_k, pi_k, metrics = modified_policy_iteration(mdp, nb_samples=args.s, m=args.m,
                                                             epsilon=args.epsilon, max_iter=args.max_iter,
                                                             is_kernel=args.is_kernel, kernel_type=args.kernel_type,
                                                             is_soft=args.is_soft, beta=args.beta,
                                                             V_opt=V_opt,
                                                             log_freq=args.log_freq)

    print("Policy Probability Distribution:")
    print(pi)
    print("")

    print("Reshaped Grid Policy (0=up, 1=down, 2=right, 3=left):")
    print(np.reshape(np.argmax(pi, axis=1), mdp.shape))
    print("")

    print("Value Function:")
    print(V)
    print("")

    print("Reshaped Grid Value Function:")
    print(V.reshape(mdp.shape))
    print("")

    if args.opt_v:
        print('||V-V*||_inf')
        print(np.max(np.abs(V_opt - V)))

    if args.save_pi:
        np.savetxt(args.save_pi, pi)
    if args.save_v:
        np.savetxt(args.save_v, V)
    if args.save_q:
        np.savetxt(args.save_q, Q)

    if args.plot:
        import plot
        import re

        title = re.sub("(.{128})", "\\1\n", str(args), 0, re.DOTALL)
        plot.plot_metrics(metrics, title=title, save_file=args.plot)

    if args.export:
        keys = sorted([key for key in metrics.keys() if len(metrics[key]) > 0])
        metrics_export = np.concatenate([np.expand_dims(metrics[key], 1) for key in keys], axis=1)
        header = ",".join(keys)
        np.savetxt(args.export, metrics_export, fmt='%1.6f', delimiter=",", header=header, comments='')