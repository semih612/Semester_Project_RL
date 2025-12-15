def apply_policy_update(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def check_lb_sgd_convergence(policies, phase, grad_log, below_counter,
                            delta=0.1, K=5):
    grad_sq = 0.0
    for p in policies[phase].parameters():
        if p.grad is not None:
            grad_sq += p.grad.norm(2).item() ** 2

    grad_norm = grad_sq ** 0.5
    grad_log[phase].append(grad_norm)

    threshold = delta / 2
    if grad_norm < threshold:
        below_counter[phase] += 1
    else:
        below_counter[phase] = 0

    converged = below_counter[phase] >= K
    return grad_norm, converged
