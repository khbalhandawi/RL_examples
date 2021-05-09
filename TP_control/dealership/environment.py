import numpy as np
import copy

class Environment():

    def __init__(self,state_0,lb_A=0,ub_A=20,lb_B=0,ub_B=20):
        
        self.state_0 = list(state_0)
        self.state = copy.deepcopy(self.state_0)
        self.reach_goal = False
        self.steps = 0
        self. lb_A = lb_A; self.ub_A = ub_A; self.lb_B = lb_B; self.ub_B = ub_B

    def poisson(self,lamda,lb,ub):
        '''
        Compute the probability of value in range [lb,ub]
        '''
        p_values = np.zeros(len(range(lb,ub+1)))
        for i in range(lb,ub+1):
            p_i = ((lamda**i)/np.math.factorial(i))*np.exp(-lamda)
            p_values[i] = p_i

        return p_values

    def reset(self):
        self.steps = 0
        self.state = copy.deepcopy(self.state_0)
        self.reach_goal = False

    def step(self,action):
        

        '''
        input:      s: dict {loc_A:, loc_B}
                    a: dict {loc_A:, loc_B}

        returns:    r: float
                    s_prime: dict {loc_A:, loc_B}
        '''

        # over night
        self.state[0] -= action
        self.state[1] += action

        # next morning
        cars_requested_A = np.random.choice(np.arange(self.lb_A, self.ub_A+1), p=self.poisson(3,lb=self.lb_A,ub=self.ub_A))
        cars_requested_B = np.random.choice(np.arange(self.lb_B, self.ub_B+1), p=self.poisson(4,lb=self.lb_B,ub=self.ub_B))

        cars_returned_A = np.random.choice(np.arange(self.lb_A, self.ub_A+1), p=self.poisson(3,lb=self.lb_A,ub=self.ub_A))
        cars_returned_B = np.random.choice(np.arange(self.lb_B, self.ub_B+1), p=self.poisson(2,lb=self.lb_B,ub=self.ub_B))

        # Reward function
        r_A = (min(cars_requested_A,self.state[0]) * 10.0)
        r_B = (min(cars_requested_B,self.state[1]) * 10.0)
        r_sell = r_A + r_B
        r_move = abs(action) * 2.0
        r = r_sell - r_move

        # Next state function
        s_A = self.state[0] - cars_requested_A + cars_returned_A
        s_B = self.state[1] - cars_requested_B + cars_returned_B
        self.state[0] = np.clip(s_A, self.lb_A, self.ub_A) # clamp values between 0 and 20
        self.state[1] = np.clip(s_B, self.lb_B, self.ub_B) # clamp values between 0 and 20

        self.steps += 1

        return tuple(self.state), r

    def get_state(self):
        return tuple(self.state)

    def is_terminal(self):
        if self.state == self.terminal_state:
            self.reach_goal = True
            return True
        else:
            self.reach_goal = False
            return False