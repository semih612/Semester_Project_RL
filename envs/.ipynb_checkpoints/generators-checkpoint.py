import random
from .maze_env import MultiAgentMazeEnv

def random_maze_env(size=(5,5), n_agents=1, n_walls=6, seed=None):
    """
    Build a random MultiAgentMazeEnv with given size, agents, and wall count.
    
    Args:
        size (tuple): grid size (rows, cols)
        n_agents (int): number of agents
        n_walls (int): number of wall cells
        seed (int or None): random seed for reproducibility
    
    Returns:
        env: a MultiAgentMazeEnv instance
    """
    if seed is not None:
        random.seed(seed)
    else:
        seed = random.randint(0, int(1e9))
        random.seed(seed)
        print("Generated seed:", seed)

    rows, cols = size

    # --- choose starts and goals ---
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    starts = random.sample(all_cells, n_agents)
    remaining = [c for c in all_cells if c not in starts]
    goals = random.sample(remaining, n_agents)

    # --- choose walls (avoid starts + goals) ---
    forbidden = set(starts + goals)
    candidates = [c for c in all_cells if c not in forbidden]
    walls = random.sample(candidates, min(n_walls, len(candidates)))

    # --- build env ---
    env = MultiAgentMazeEnv(
        size=size,
        starts=starts,
        goals=goals,
        walls=walls
    )
    return env