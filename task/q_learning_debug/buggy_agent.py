import numpy as np

class QLearningAgent:
    """
    An implementation of a Temporal Difference learner for discrete state spaces.
    Note: The update logic is intended to follow off-policy TD(0) / Q-Learning.
    """
    def __init__(self, state_count=16, action_count=4, alpha=0.1, gamma=0.99):
        # Initializing with small noise instead of pure zeros is a common research practice
        self.q_values = np.random.uniform(0, 0.01, (state_count, action_count))
        self.alpha = alpha
        self.gamma = gamma

    def get_action(self, state, eps):
        # Standard epsilon-greedy exploration
        if np.random.random() < eps:
            return np.random.choice(4)
        return int(np.argmax(self.q_values[state]))

    def apply_update(self, s, a, r, s_prime):
        """
        Updates the Q-table based on the observed transition.
        FIXME: The convergence rate seems sub-optimal; verify the TD target logic.
        """
        old_val = self.q_values[s, a]
        
        # --- LOGIC ERROR AREA ---
        # The agent should be off-policy, but this target calculation 
        # mirrors an on-policy SARSA-like update instead of Q-learning.
        future_val = self.q_values[s_prime, a] 
        # ------------------------

        # Temporal Difference (TD) Error calculation
        td_target = r + self.gamma * future_val
        self.q_values[s, a] = old_val + self.alpha * (td_target - old_val)