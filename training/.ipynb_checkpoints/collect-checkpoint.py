import torch
from barriers import log_barrier_penalty, initialize_constraint_vector


def collect_batch_trajectories(env, policies, phase, batch_size, max_t):
    """
    Collect a batch of trajectories for one agent (phase).

    Returns:
        batch_data: list of dicts
        batch_violation_count: int
    """
    batch_data = []
    batch_violation_count = 0

    for _ in range(batch_size):
        states = env.reset()

        saved_log_probs = []
        rewards = []
        entropies = []

        C_tau = initialize_constraint_vector(n_constraints=8)
        episode_has_violation = False

        for _ in range(max_t):
            a, lp, ent = policies[phase].act(states[phase])

            next_states, step_rewards, dones = env.step([a])

            C_tau, violated = log_barrier_penalty(next_states, C_tau, env)
            episode_has_violation |= violated

            saved_log_probs.append(lp)
            rewards.append(step_rewards[phase])
            entropies.append(ent)

            states = next_states
            if any(dones):
                break

        if episode_has_violation:
            batch_violation_count += 1

        batch_data.append({
            "log_probs": saved_log_probs,
            "rewards": rewards,
            "entropies": entropies,
            "C_tau": C_tau,
        })

    return batch_data, batch_violation_count
