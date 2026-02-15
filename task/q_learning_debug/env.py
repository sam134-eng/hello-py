import numpy as np

class MazeWorld:
    """
    A 4x4 GridWorld environment with terminal traps and a target goal.
    Designed for testing value convergence in temporal difference learning.
    """
    def __init__(self):
        self.size = 4
        self.state = 0  
        self.goal = 15  
        self.traps = {6, 7, 10, 12} # Used a set for faster lookup O(1)
        
    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        # Directions: 0=Up, 1=Right, 2=Down, 3=Left
        r, c = self.state // self.size, self.state % self.size
        
        # Calculate next position
        if action == 0:    r = max(0, r - 1)
        elif action == 1:  c = min(self.size - 1, c + 1)
        elif action == 2:  r = min(self.size - 1, r + 1)
        elif action == 3:  c = max(0, c - 1)
        
        self.state = r * self.size + c
        
        # Determine reward and termination
        if self.state == self.goal:
            return self.state, 10.0, True
        
        if self.state in self.traps:
            return self.state, -1.0, True
            
        return self.state, -0.1, False