import numpy as np
from collections import deque
import torch

from training.collect import collect_batch_trajectories
from training.losses import compute_policy_loss
from training.updates import apply_policy_update, check_lb_sgd_convergence
from training.logging import print_training_progress
from barriers import update_constraint_value_estimates_batch, compute_barrier_weight


def reinforce_multi_rwd2go_alt_barrier(env, policies, optimizers, n_episodes=30000, max_t=20, gamma=0.9, batch_size=256, print_every=1, num_cons = 8):
    episodic_return_log = [[] for _ in range(env.n_agents)]
    barrier_log = [[] for _ in range(env.n_agents)]
    violation_log = [[] for _ in range(env.n_agents)]
    grad_log = [[] for _ in range(env.n_agents)]
    scores_deque = deque(maxlen=20)

    below_counter = [0] * env.n_agents

    lambda_maze = 10.0
    lambda_decay = 0.99
    alpha_vi = 0.02

    for phase in range(env.n_agents):
        V_i_estimate = [None] * num_cons
        episode_count = 0
        
        for episode in range(n_episodes):
            episode_count += 1
            
            batch_data, batch_violation_count  = collect_batch_trajectories(env, policies, phase, batch_size, max_t)
            C_tau_batch = torch.stack([traj["C_tau"] for traj in batch_data])
            V_i_estimate = update_constraint_value_estimates_batch(C_tau_batch, V_i_estimate, alpha_vi)

            batch_loss = 0.0
            avg_return = 0.0
            barrier_vals = []
            batch_returns = []

            for traj in batch_data:
                bw = compute_barrier_weight(traj["C_tau"], V_i_estimate, lambda_maze)
                loss, mean_G = compute_policy_loss(traj["log_probs"], traj["rewards"], gamma, bw)
                batch_loss += loss
                avg_return += mean_G.item()
                batch_returns.append(sum(traj["rewards"]))
                barrier_vals.append(float(bw))  # or bw.item() if tensor

            batch_loss /= batch_size
            apply_policy_update(optimizers[phase], batch_loss)

            if episode_count % 3 == 0:
                lambda_maze = max(0,lambda_maze * lambda_decay)

            grad_norm, converged = check_lb_sgd_convergence(policies, phase, grad_log, below_counter)

            avg_batch_return = float(np.mean(batch_returns))
            episodic_return_log[phase].append(avg_batch_return)
            scores_deque.append(avg_batch_return)
            avg_reward_window = float(np.mean(scores_deque))
            
            barrier_log[phase].append(np.mean(barrier_vals))
            avg_barrier = float(np.mean(barrier_vals))
            violation_log[phase].append(batch_violation_count)


            print_training_progress(phase, episode_count, avg_reward_window, batch_violation_count, avg_barrier, grad_norm, print_every)

            if converged:
                print(f"Agent {phase} converged.")
                break

    return episodic_return_log, barrier_log, violation_log
