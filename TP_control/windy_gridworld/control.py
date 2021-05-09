import matplotlib.pyplot as plt
import time

class TDControlOnPolicy():

    def __init__(self, agent, environment, visualizer, budget=200000,gamma=1,alpha=0.1,plotInterval=20):

        self.budget = budget
        self.t = 0
        self.N_episodes = 0
        self.episodes = []
        self.time_steps = []
        self.length_episodes = []
        self.alpha = alpha
        self.gamma = gamma
        self.plotInterval = plotInterval

        self.agent = agent
        self.environment = environment
        self.visualizer = visualizer

    def run_test(self,plotSteps=False):
        
        state = self.environment.get_state()
        action = self.agent.get_action_greedy(state)

        while not self.environment.reach_goal and self.environment.steps < 5000:
            s_prime, reward = self.environment.step(action)
            a_prime = self.agent.get_action_greedy(s_prime)

            self.agent.update_policy(state,action,reward,s_prime,a_prime,self.alpha,self.gamma)

            state = s_prime
            action = a_prime

            if plotSteps:
                self.visualizer.draw_policy(self.agent.pi,state)
                time.sleep(0.5)

    def train(self):
        
        state = self.environment.get_state()
        action = self.agent.get_action_soft(state)

        while self.t <= self.budget:

            s_prime, reward = self.environment.step(action)
            a_prime = self.agent.get_action_soft(s_prime)

            self.agent.update_policy(state,action,reward,s_prime,a_prime,self.alpha,self.gamma)

            state = s_prime
            action = a_prime

            if self.environment.reach_goal or self.environment.steps > 5000:
                
                self.N_episodes += 1
                self.environment.reset()
                self.run_test()

                self.episodes += [self.N_episodes]
                self.length_episodes += [self.environment.steps]
                self.environment.reset()

                state = self.environment.get_state()
                action = self.agent.get_action_soft(state)
                
            if self.t % self.plotInterval == 0:
                self.visualizer.draw_policy(self.agent.pi)
                self.visualizer.draw_reward(self.episodes,self.length_episodes)

                print("Time step %i" %(self.t))
            
            self.t += 1