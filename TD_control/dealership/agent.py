import numpy as np

class Agent():

    def get_possible_acts(self,state):
        # decide on action
        possible_actions = []
        for action in self.actions:
            if state[0] - action >= self.lb_A and state[1] + action >= self.lb_B and action + state[1] <= self.ub_B and - action + state[0] <= self.ub_A:
                possible_actions += [action]

        return possible_actions

    def __init__(self,lb_A=0,ub_A=20,lb_B=0,ub_B=20,lb_a=-5,ub_a=5,epsilon=0.2):

        '''
        Initializes the set of actions and states
        returns:    pi: list of dicts
        '''

        self.epsilon = epsilon
        self.actions = np.arange(lb_a,ub_a+1)
        self.lb_A = lb_A; self.ub_A = ub_A; self.lb_B = lb_B; self.ub_B = ub_B

        states_A = np.arange(lb_A,ub_A+1)
        states_B = np.arange(lb_B,ub_B+1)

        states = np.array(np.meshgrid(states_A, states_B)).T.reshape(-1,2)

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

        if np.random.rand() > self.epsilon:
            soft_action = self.pi[state]
        else:
            soft_action = np.random.choice(self.a[state])

        return soft_action

    def get_action_greedy(self,state):

        greedy_action = self.pi[state]

        return greedy_action

    def get_probability_action(self,action):

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