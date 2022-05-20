from copy import copy, deepcopy
from scipy.special import logsumexp
from scipy.stats import entropy
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning

from gridworld_mdp import GridworldMDP

import os
RND_SEED = int(os.environ.get('RND_SEED', 123))
print('seed', RND_SEED)
import numpy as np
np.random.seed(RND_SEED)

def log_softmax(vec):
    return vec - logsumexp(vec)

def softmax(vec):
    probs = np.exp(log_softmax(vec))
    probs /= probs.sum()
    return probs

def neg_entropy(pi):
    return -1*entropy(pi)

def modified_policy_iteration(mdp, epsilon=None, max_iter=None,
                              nb_samples=None, m=1,
                              is_kernel=False, kernel_type=None,
                              is_reg=False, temp=None,
                              is_soft=False, beta=None,
                              is_fa=False, V_opt=None,
                              log_freq=1):

    def Q_V(V):
        q_v = mdp.R + mdp.gamma * np.matmul(mdp.P, V)
        return q_v

    def bellman_op(V, R_pi, P_pi, m, is_reg=False, temp=None, Omega_pi=None, is_soft=False, beta=None):
        if m == 0:
            return V

        TS = R_pi + mdp.gamma * np.matmul(P_pi, bellman_op(V, R_pi, P_pi, m-1, is_reg=is_reg, temp=temp, Omega_pi=Omega_pi))
        if is_reg:
            TS -= temp * Omega_pi
        if is_soft:
            TS = (1 - beta) * V + beta * TS
        return TS

    def empirical_bellman_op(R, V, is_reg=False, temp=None, Omega_pi=None, is_soft=False, beta=None):
        TS = R + mdp.gamma*V
        if is_reg:
            TS -= temp * Omega_pi
        if is_soft:
            TS = (1 - beta) * V + beta * TS
        return TS

    def empirical_value_iteration(V, pi, is_reg=False, temp=None, is_soft=False, beta=None):
        V_update = deepcopy(V)
        for s in range(mdp.nS):
            V_update[s] = 0
            Omega_pi = neg_entropy(pi[s, :])
            playing_policy = pi[s, :]

            sampled_actions = np.random.choice(mdp.nA, size=nb_samples, p=playing_policy)
            for action in sampled_actions:
                next_state = np.random.choice(mdp.nS, size=1, p=mdp.P[s, action, :])
                V_update[s] += empirical_bellman_op(mdp.R[s, action], V[next_state],
                                                    is_reg=is_reg, temp=temp, Omega_pi=Omega_pi,
                                                    is_soft=is_soft, beta=beta)
            V_update[s] /= nb_samples

        return V_update

    def value_iteration(V, pi, m, is_reg=False, temp=None, is_soft=False, beta=None):
        P_pi = np.zeros((mdp.nS, mdp.nS))
        R_pi = np.zeros(mdp.nS)
        for s in range(mdp.nS):
            P_pi[s] = np.matmul(pi[s,:], mdp.P[s, :, :])
            R_pi[s] = np.matmul(pi[s,:], mdp.R[s, :])

        Omega_pi = None
        if is_reg:
            Omega_pi = np.apply_along_axis(neg_entropy, 1, pi)

        if np.isinf(m):
            # v^\pi_k
            R = R_pi - np.multiply(temp, Omega_pi) if is_reg else R_pi
            PE = np.matmul(np.linalg.inv(np.identity(mdp.nS) - mdp.gamma * P_pi), R)
        else:
            PE = bellman_op(V, R_pi, P_pi, m, is_reg=is_reg, temp=temp, Omega_pi=Omega_pi,
                                          is_soft=is_soft, beta=beta)
        return PE

    def greedy_policy_Q(Q, is_reg=False, temp=None):
        policy = np.zeros([mdp.nS, mdp.nA])
        for s in range(mdp.nS):
            if is_reg and temp > 1e-200:
                policy[s] = softmax(Q[s,:] / temp)
            else:
                policy[s, np.argmax(Q[s,:])] = 1.0
        return policy

    def greedy_step(V, is_reg=False, temp=None):
        return greedy_policy_Q(Q_V(V), is_reg=is_reg, temp=temp)

    V = np.zeros(mdp.nS)
    pi = (1./mdp.nA) * np.ones_like(mdp.R)
    V_k, pi_k = list(), list()
    errors = {'avg_ent_pi': [], 'temp': [],
              'beta': [],
              'max_opt_v': [], 'avg_opt_v': []}
    n = 0
    while True:
        # Stopping condition
        delta = 0.0
        # Policy Evaluation
        if nb_samples:
            V_pi_eval = empirical_value_iteration(V, pi, is_reg=is_reg, temp=temp,
                                                  is_soft=is_soft, beta=beta)
        else:
            V_pi_eval = value_iteration(V, pi, m, is_reg=is_reg, temp=temp,
                                                  is_soft=is_soft, beta=beta)

        # FA
        if is_fa:
            fa = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(8,), activation='relu', max_iter=5)
            fa.fit(np.eye(mdp.nS), V_pi_eval)
            V_pi_eval = fa.predict(np.eye(mdp.nS))

        # Kernel
        if is_kernel:
            kernel = np.eye(mdp.nS)
            V_pi_eval = np.matmul(kernel, V_pi_eval)

        # Policy Improvement
        pi = greedy_step(V_pi_eval, is_reg=is_reg, temp=temp)
        ent_pi = np.apply_along_axis(entropy, 1, pi)
        avg_ent_pi = np.average(ent_pi)

        # Per iteration error
        errors['avg_ent_pi'].append(avg_ent_pi)

        errors['beta'].append(beta if beta else 1.0)
        errors['temp'].append(temp if temp else 0)

        # Error to optimality
        max_opt_error = np.max(np.abs(V_opt - V_pi_eval)) if V_opt is not None else 0.0
        avg_opt_error = np.average(np.abs(V_opt - V_pi_eval)) if V_opt is not None else 0.0
        errors['max_opt_v'].append(max_opt_error)
        errors['avg_opt_v'].append(avg_opt_error)

        # Calculate delta across all states seen so far
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
            for key in errors.keys():
                out_str += "{}:{:.2f}\t".format(key, np.average(errors[key][-log_freq:]))
            print(out_str)
        # Check if we can stop
        if (max_iter and n >= max_iter) or (epsilon and delta < epsilon):
            print("Finish after iterations,", n)
            break

    return pi, V, Q_V(V), V_k, pi_k, errors

if __name__ == "__main__":
    # execute only if run as a script
    import argparse

    parser = argparse.ArgumentParser(description='Approximate Smooth Kernel Modified Policy Iteration')
    parser.add_argument('--plan', help='Plan file', default='plans/plan7.txt', required=True)
    # random slide
    parser.add_argument('--random-slide', help='Random slide', default=0.0, type=float)
    # kernel
    parser.add_argument('--kernel', dest='is_kernel', action='store_true', help='Kernel VI')
    parser.add_argument('--kernel-type', help='Kernel type', default='identity', type=str)
    # reg
    parser.add_argument('--reg', dest='is_reg', action='store_true', help='Reg-MPI')
    parser.add_argument('--temp', default=0.01, type=float, help='Temperature for Reg-MPI')
    # FA
    parser.add_argument('--fa', dest='is_fa', action='store_true', help='Function approximation')
    # soft bellman operator
    parser.add_argument('--soft', dest='is_soft', action='store_true', help='Soft Bellman operator')
    parser.add_argument('--beta', help='Beta for soft Bellman operator', default=1.0, type=float)
    # stopping
    parser.add_argument('--epsilon', help='Epsilon', default=None, type=float)
    parser.add_argument('--max-iter', help='Max number of iterations', default=None, type=int)
    # m
    parser.add_argument('--m', help='Number of applications of Bellman op', default=1, type=float)
    # number of states to sample
    parser.add_argument('--s', help='Number of samples', default=None, type=int)
    # opt v
    parser.add_argument('--opt-v', help='Optimal value file', default=None)
    # discount factor
    parser.add_argument('--gamma', help='Gamma discount factor for LT reward', default=0.9, type=float)
    # plots
    parser.add_argument('--plot-errors', default='errors.png', help='Plot errors file')
    parser.add_argument('--export-errors', default=None, help='Export errors file')
    # save v and pi
    parser.add_argument('--save-v', help='Save value file', default=None)
    parser.add_argument('--save-pi', help='Save policy file', default=None)
    parser.add_argument('--save-q', help='Save q function file', default=None)
    # log
    parser.add_argument('--log-freq', help='Logging frequency', default=1, type=int)

    args = parser.parse_args()
    print(args)

    mdp = GridworldMDP(plan_file=args.plan, gamma=args.gamma, random_slide=args.random_slide)
    V_opt = np.loadtxt(args.opt_v) if args.opt_v else None
    pi, V, Q, V_k, pi_k, errors = modified_policy_iteration(mdp, nb_samples=args.s, m=args.m,
                                                            epsilon=args.epsilon, max_iter=args.max_iter,
                                                            is_kernel=args.is_kernel, kernel_type=args.kernel_type,
                                                            is_reg=args.is_reg, temp=args.temp,
                                                            is_soft=args.is_soft, beta=args.beta,
                                                            is_fa=args.is_fa, V_opt=V_opt,
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

    if args.plot_errors:
        import plot
        import re

        title = re.sub("(.{128})", "\\1\n", str(args), 0, re.DOTALL)
        plot.plot_errors(errors, title=title, save_file=args.plot_errors)

    if args.export_errors:
        keys = sorted([key for key in errors.keys() if len(errors[key]) > 0])
        #print([len(errors[key]) for key in keys])
        errors_export = np.concatenate([np.expand_dims(errors[key], 1) for key in keys], axis=1)
        header = ",".join(keys)
        np.savetxt(args.export_errors, errors_export, fmt='%1.6f', delimiter=",", header=header, comments='')