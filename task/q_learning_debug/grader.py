import numpy as np
from env import MazeWorld
from buggy_agent import QLearningAgent

def evaluate_convergence():
    """
    Validation script to verify if the agent recovers the optimal value function 
    under off-policy constraints.
    """
    env = MazeWorld()
    # Parameters aligned with the task's expected complexity
    agent = QLearningAgent(state_count=16, action_count=4, alpha=0.1, gamma=0.9)
    
    episodes = 2500
    eps = 0.2  # Sufficient exploration for a 4x4 grid
    
    for _ in range(episodes):
        state = env.reset()
        is_terminal = False
        
        while not is_terminal:
            action = agent.get_action(state, eps)
            next_state, reward, is_terminal = env.step(action)
            
            # Execute the update logic being tested
            agent.apply_update(state, action, reward, next_state)
            state = next_state
            
        # Decay epsilon slightly to stabilize training (human touch)
        eps = max(0.01, eps * 0.999)

    # SUCCESS CRITERIA
    # I check the Value of the starting state. 
    # In a deterministic 4x4 grid, the Max Q-value at state 0 
    # must exceed a threshold if the Bellman optimality update is correct.
    v_start = np.max(agent.q_values[0])
    
    # Off-policy Q-learning will reach ~5.0+; SARSA (the bug) will stay lower 
    # due to the negative rewards of exploration.
    success = v_start > 4.5
    
    print(f"Final V(start): {v_start:.4f}")
    print(f"Status: {'PASSED' if success else 'FAILED'}")
    
    return success

if __name__ == "__main__":
    evaluate_convergence()