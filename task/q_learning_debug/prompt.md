# RL Task: Resolution of Value Estimation Error in Off-Policy Learning

## Background
We are testing a discrete Q-learning agent on a 4x4 grid navigation task (`MazeWorld`). While the infrastructure is functional, the agent is currently failing to converge to the optimal value function $V^*(s)$. 

Preliminary logs suggest that the agent's value estimation is being biased by its exploration policy, which is characteristic of an on-policy update error.

## Your Task
1. **Analyze `buggy_agent.py`**: Specifically, examine the `apply_update` method.
2. **Identify the Logic Error**: The current implementation is failing to correctly utilize the Bellman Optimality Equation for off-policy learning. 
3. **Requirement**: Ensure the Temporal Difference (TD) target is calculated using the maximum possible value for the next state, independent of the current policy's action. 

## Constraints
* Do not change the class or method signatures as the `grader.py` depends on them.
* Use `numpy` operations for the update calculation.
* Your fix must allow the agent's start-state value to exceed a threshold of 4.5 within 2500 episodes.

## Submission
Please provide only the corrected `apply_update` method or the full `QLearningAgent` class.