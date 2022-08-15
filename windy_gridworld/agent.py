import numpy as np

class TDAgent():

    def get_possible_acts(self,state):
        # decide on action
        possible_actions = []

        if state[0] - 1 >= self.lb_x:
            possible_actions += [self.actions[0]] # "L"

        if state[0] + 1 <= self.ub_x:
            possible_actions += [self.actions[1]] # "R"

        if state[1] - 1 >= self.lb_y:
            possible_actions += [self.actions[2]] # "D"

        if state[1] + 1 <= self.ub_y:
            possible_actions += [self.actions[3]] # "U"

        if self.atype == "8D":
            if (state[0] - 1 >= self.lb_x) and (state[1] - 1 >= self.lb_y):
                possible_actions += [self.actions[4]] # "LD"

            if (state[0] - 1 >= self.lb_x) and (state[1] + 1 <= self.ub_y):
                possible_actions += [self.actions[5]] # "LU"

            if (state[0] + 1 <= self.ub_x) and (state[1] - 1 >= self.lb_y):
                possible_actions += [self.actions[6]] # "RD"

            if (state[0] + 1 <= self.ub_x) and (state[1] + 1 <= self.ub_y):
                possible_actions += [self.actions[7]] # "RU"

        return possible_actions

    def __init__(self,lb_x=0,ub_x=9,lb_y=0,ub_y=6,epsilon=0.1,atype="4D"):

        '''
        Initializes the set of actions and states
        returns:    pi: list of dicts
        '''

        self.epsilon = epsilon
        self.lb_x=lb_x; self.ub_x=ub_x; self.lb_y=lb_y; self.ub_y=ub_y
        self.atype = atype

        if self.atype == "4D":
            self.actions = ["L","R","D","U"]
        elif self.atype == "8D":
            self.actions = ["L","R","D","U","LD","LU","RD","RU"]

        print(self.actions)

        states_X = np.arange(self.lb_x,self.ub_x+1)
        states_Y = np.arange(self.lb_y,self.ub_y+1)

        states = np.array(np.meshgrid(states_X, states_Y)).T.reshape(-1,2)

        # Generate initial policy pi
        self.pi = {}; self.Q = {}; self.a = {}
        for state_array in states:
            
            state = tuple(state_array)

            print(state)
            self.a[state] = self.get_possible_acts(state)
            print(self.a[state])

            Q_a = []
            for possible_action in self.a[state]:

                # q_init = np.random.random()
                q_init = 0
                Q_a += [q_init]
                self.Q[(state,possible_action)] = q_init

            self.pi[state] = self.a[state][np.argmax(Q_a)]

    def get_action_soft(self,state):

        if np.random.rand() > self.epsilon and self.pi[state] in self.a[state]:
            self.soft_action = self.pi[state]
        else:
            self.soft_action = np.random.choice(self.a[state])

        return self.soft_action

    def get_action_greedy(self,state):

        if self.pi[state] in self.a[state]:
            self.greedy_action = self.pi[state]
        else:
            self.greedy_action = np.random.choice(self.a[state])

        return self.greedy_action

    def get_probability_action(self):

        num_actions = len(self.a[state])

        # Update behavioral policy
        if self.pi[state] in self.a[state]:

            if self.soft_action == self.pi[state]:
                prob = 1 - self.epsilon + self.epsilon/num_actions
            else:
                prob = self.epsilon/num_actions
        else:
            prob = 1/num_actions

        return prob

    def update_policy_ON(self,state,action,reward,s_prime,a_prime,alpha,gamma):

        error = reward + (gamma * self.Q[(s_prime,a_prime)]) - self.Q[(state,action)]
        Q_updated =  self.Q[(state,action)] + (alpha * error)

        self.Q[(state,action)] = Q_updated

        Q_a = []
        for possible_action in self.a[state]: 
            Q_a += [self.Q[(state,possible_action)]]

        self.pi[state] = self.a[state][np.argmax(Q_a)]

    def update_policy_OFF(self,state,action,reward,s_prime,alpha,gamma):

        Q_a = []
        for a in self.a[s_prime]:
            Q_a += [self.Q[(s_prime,a)]]

        Q_max = np.max(Q_a)

        error = reward + (gamma * Q_max) - self.Q[(state,action)]
        Q_updated =  self.Q[(state,action)] + (alpha * error)

        self.Q[(state,action)] = Q_updated

        Q_a = []
        for possible_action in self.a[state]: 
            Q_a += [self.Q[(state,possible_action)]]

        self.pi[state] = self.a[state][np.argmax(Q_a)]


class MCAgent():

    def get_possible_acts(self,state):
        # decide on action
        possible_actions = []

        if state[0] - 1 >= self.lb_x:
            possible_actions += [self.actions[0]] # "L"

        if state[0] + 1 <= self.ub_x:
            possible_actions += [self.actions[1]] # "R"

        if state[1] - 1 >= self.lb_y:
            possible_actions += [self.actions[2]] # "D"

        if state[1] + 1 <= self.ub_y:
            possible_actions += [self.actions[3]] # "U"

        if self.atype == "8D":
            if (state[0] - 1 >= self.lb_x) and (state[1] - 1 >= self.lb_y):
                possible_actions += [self.actions[4]] # "LD"

            if (state[0] - 1 >= self.lb_x) and (state[1] + 1 <= self.ub_y):
                possible_actions += [self.actions[5]] # "LU"

            if (state[0] + 1 <= self.ub_x) and (state[1] - 1 >= self.lb_y):
                possible_actions += [self.actions[6]] # "RD"

            if (state[0] + 1 <= self.ub_x) and (state[1] + 1 <= self.ub_y):
                possible_actions += [self.actions[7]] # "RU"

        return possible_actions

    def __init__(self,lb_x=0,ub_x=9,lb_y=0,ub_y=6,epsilon=0.1,atype="4D"):

        '''
        Initializes the set of actions and states
        returns:    pi: list of dicts
        '''

        self.epsilon = epsilon
        self.lb_x=lb_x; self.ub_x=ub_x; self.lb_y=lb_y; self.ub_y=ub_y
        self.atype = atype

        if self.atype == "4D":
            self.actions = ["L","R","D","U"]
        elif self.atype == "8D":
            self.actions = ["L","R","D","U","LD","LU","RD","RU"]

        print(self.actions)

        states_X = np.arange(self.lb_x,self.ub_x+1)
        states_Y = np.arange(self.lb_y,self.ub_y+1)

        self.states = np.array(np.meshgrid(states_X, states_Y)).T.reshape(-1,2)

        # Generate initial policy pi
        self.pi = {}; self.Q = {}; self.a = {}; self.C = {}
        for state_array in self.states:
            
            state = tuple(state_array)

            print(state)
            self.a[state] = self.get_possible_acts(state)
            print(self.a[state])

            Q_a = []
            for possible_action in self.a[state]:

                # q_init = np.random.random()
                q_init = 0
                Q_a += [q_init]
                self.Q[(state,possible_action)] = q_init
                self.C[(state,possible_action)] = 0

            self.pi[state] = self.a[state][np.argmax(Q_a)]

    def get_action_soft(self,state):

        if np.random.rand() > self.epsilon and self.pi[state] in self.a[state]:
            self.soft_action = self.pi[state]
        else:
            self.soft_action = np.random.choice(self.a[state])

        return self.soft_action

    def get_action_greedy(self,state):

        if self.pi[state] in self.a[state]:
            self.greedy_action = self.pi[state]
        else:
            self.greedy_action = np.random.choice(self.a[state])

        return self.greedy_action

    def get_probability_action(self,state,action):

        num_actions = len(self.a[state])

        # Update behavioral policy
        if self.pi[state] in self.a[state]:

            if action == self.pi[state]:
                prob = 1 - self.epsilon + self.epsilon/num_actions
            else:
                prob = self.epsilon/num_actions
        else:
            prob = 1/num_actions

        return prob

    def update_policy(self,state,action,reward,s_prime,a_prime,alpha,gamma):

        error = reward + (gamma * self.Q[(s_prime,a_prime)]) - self.Q[(state,action)]
        Q_updated =  self.Q[(state,action)] + (alpha * error)

        self.Q[(state,action)] = Q_updated

        Q_a = []
        for possible_action in self.a[state]: 
            Q_a += [self.Q[(state,possible_action)]]

        self.pi[state] = self.a[state][np.argmax(Q_a)]