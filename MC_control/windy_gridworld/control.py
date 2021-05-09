import matplotlib.pyplot as plt
import time
import numpy as np

class MCControl():

    def __init__(self, agent, environment, visualizer, budget=200000,gamma=1,plotInterval=20,episodeLength=5000,decay=0.9):

        self.budget = budget
        self.t = 0
        self.N_episodes = 0
        self.episodes = []
        self.time_steps = []
        self.length_episodes = []
        self.gamma = gamma
        self.plotInterval = plotInterval
        self.episodeLength = episodeLength
        self.decay = decay

        self.agent = agent
        self.environment = environment
        self.visualizer = visualizer

    def replay_episode(self,s,A,R,probs):

        G = 0
        W = 1

        T = len(s)

        for t in range(T-1,-1,-1):

            S_t = s[t]; A_t = A[t]

            G = self.gamma*G + R[t+1]
            self.agent.C[(S_t,A_t)] += W
            self.agent.Q[(S_t,A_t)] += ( W * (G - self.agent.Q[(S_t,A_t)]) ) / self.agent.C[(S_t,A_t)]

            Q_S_a = []
            for action in self.agent.a[S_t]:
                Q_S_a += [self.agent.Q[(S_t,action)]]

            self.agent.pi[S_t] = self.agent.a[S_t][np.argmax(Q_S_a)]
            if A_t != self.agent.pi[S_t]:
                break

            W /= probs[t]
        
        self.agent.epsilon *= self.decay

    def run_test(self,plotSteps=False):

        states = []; actions = []; probs = []; rewards = [0]

        while not self.environment.reach_goal and self.environment.steps <= self.episodeLength:
            state = self.environment.get_state()
            action = self.agent.get_action_greedy(state)
            prob = 1
            
            s_prime, reward = self.environment.step(action)

            states += [state]; actions += [action]; rewards += [reward]; probs += [prob]

            if plotSteps:
                self.visualizer.draw_policy(self.agent.pi,state)
                time.sleep(0.5)

        # self.replay_episode(states,actions,rewards,probs)

    def train(self):

        states = []; actions = []; probs = []; rewards = [0]

        while self.t <= self.budget:

            state = self.environment.get_state()
            action = self.agent.get_action_soft(state)
            prob = self.agent.get_probability_action(state,action)

            s_prime, reward = self.environment.step(action)

            states += [state]; actions += [action]; rewards += [reward]; probs += [prob]

            if self.environment.reach_goal or self.environment.steps > self.episodeLength:
                
                self.replay_episode(states,actions,rewards,probs)

                self.N_episodes += 1
                self.environment.reset()
                self.run_test()

                self.episodes += [self.N_episodes]
                self.length_episodes += [self.environment.steps]
                self.environment.reset()

                states = []; actions = []; probs = []; rewards = [0]
                
            if self.t % self.plotInterval == 0:
                self.visualizer.draw_policy(self.agent.pi)
                self.visualizer.draw_reward(self.episodes,self.length_episodes)

                print("Time step %i" %(self.t))
            
            self.t += 1