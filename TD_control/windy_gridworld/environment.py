import numpy as np
import copy

class Environment():

    def __init__(self,state_0,lb_x=0,ub_x=9,lb_y=0,ub_y=6):
        
        self.state_0 = list(state_0)
        self.state = copy.deepcopy(self.state_0)
        self.wind_distribution = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
        # self.wind_distribution = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.terminal_state = [7,3]
        self.reach_goal = False
        self.steps = 0
        self. lb_x = lb_x; self.ub_x = ub_x; self.lb_y = lb_y; self.ub_y = ub_y

    def reset(self):
        self.steps = 0
        self.state = copy.deepcopy(self.state_0)
        self.reach_goal = False

    def step(self,action):
        
        if action == "L":
            self.state[0] -= 1

        elif action == "R":
            self.state[0] += 1

        elif action == "D":
            self.state[1] -= 1

        elif action == "U":
            self.state[1] += 1

        elif action == "LD":
            self.state[0] -= 1
            self.state[1] -= 1

        elif action == "LU":
            self.state[0] -= 1
            self.state[1] += 1

        elif action == "RD":
            self.state[0] += 1
            self.state[1] -= 1

        elif action == "RU":
            self.state[0] += 1
            self.state[1] += 1

        self.state[1] = min(self.ub_y, self.state[1] + self.wind_distribution[self.state[0]]) # shift by wind force first

        self.steps += 1

        if self.is_terminal():
            reward = 0
        else:
            reward = -1

        return tuple(self.state), reward

    def get_state(self):
        return tuple(self.state)

    def is_terminal(self):
        if self.state == self.terminal_state:
            self.reach_goal = True
            return True
        else:
            self.reach_goal = False
            return False