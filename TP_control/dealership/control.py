import matplotlib.pyplot as plt

class TDControl():

    def __init__(self, agent, environment, visualizer, budget=20000,gamma=1,alpha=0.1,plotInterval=200):

        self.budget = budget
        self.t = 0
        self.N_episodes = 0
        self.episodes = []
        self.time_steps = []
        self.length_episodes = []
        self.returns = []
        self.G = 0
        self.alpha = alpha
        self.gamma = gamma
        self.plotInterval = plotInterval

        self.agent = agent
        self.environment = environment
        self.visualizer = visualizer

    def run_test(self):
        
        G = 0
        
        state = self.environment.get_state()
        action = self.agent.get_action_greedy(state)

        while self.environment.steps < self.plotInterval:
            s_prime, reward = self.environment.step(action)
            a_prime = self.agent.get_action_greedy(s_prime)

            self.agent.update_policy(state,action,reward,s_prime,a_prime,self.alpha,self.gamma)

            state = s_prime
            action = a_prime

            G += reward

        return G

    def train(self):
    
        state = self.environment.get_state()
        action = self.agent.get_action_soft(state)

        while self.t <= self.budget:

            s_prime, reward = self.environment.step(action)
            a_prime = self.agent.get_action_soft(s_prime)

            self.agent.update_policy(state,action,reward,s_prime,a_prime,self.alpha,self.gamma)

            state = s_prime
            action = a_prime

            self.time_steps += [self.t]

            if self.environment.steps > self.plotInterval:
                
                self.N_episodes += 1
                self.environment.reset()
                G = self.run_test()

                self.episodes += [self.N_episodes]
                self.length_episodes += [self.environment.steps]
                self.returns += [G]
                self.environment.reset()

                state = self.environment.get_state()
                action = self.agent.get_action_soft(state)
                
                self.visualizer.draw(self.agent.pi,self.episodes,self.returns)

                print("Time step %i" %(self.t))
            
            self.t += 1
