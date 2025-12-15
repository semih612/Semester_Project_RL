def print_training_progress(phase, episode_count, avg_reward, n_violations,avg_barrier, grad_norm, print_every):
    if episode_count % print_every != 0:
        return

    msg  = f" Agent{phase} Episode {episode_count}"
    msg += f" | avgR={avg_reward:.4f}"
    msg += f" | violations={n_violations}"
    msg += f" | avgBarrier={avg_barrier:.4f}"
    msg += f" | gradNorm={grad_norm:.4f}"
    print(msg)