import numpy as np
from environment import Environment
from agent import MCAgent, TDAgent
from visualizer import Visualizer
from control import TDControlOnPolicy,TDControlOffPolicy,MCControl
import matplotlib.pyplot as plt

if __name__ == "__main__":

    lb_x=0; ub_x=9; lb_y=0; ub_y=6
    lx = ub_x - lb_x + 1; ly = ub_y - lb_y + 1

    plot_data = np.zeros((lx,ly))
    plot_data[7,3] = 2
    plot_data[0,3] = 1

    vis = Visualizer(plot_data,lx,ly)
    vis.setup()
    # vis.visualize_gridworld()

    ####################################
    # TD control on-policy
    # state_0 = (0,3)
    # wind=[0,0,0,1,1,1,2,2,1,0]
    # # wind=[0,0,0,0,0,0,0,0,0,0]
    # env = Environment(state_0,wind=wind,wind_type="stochastic")
    # agent = TDAgent(epsilon=0.1,atype="8D")

    # contoller = TDControlOnPolicy(agent,env,vis,budget=80000,alpha=0.1,plotInterval=2000)
    # contoller.train()
    # plt.show()

    # contoller.environment.reset()
    # contoller.run_test(plotSteps=True)

    ####################################
    # TD control off-policy
    state_0 = (0,3)
    wind=[0,0,0,1,1,1,2,2,1,0]
    # wind=[0,0,0,0,0,0,0,0,0,0]
    env = Environment(state_0,wind=wind,wind_type="stochastic")
    agent = TDAgent(epsilon=0.1,atype="8D")

    contoller = TDControlOffPolicy(agent,env,vis,budget=160000,alpha=0.1,plotInterval=2000)
    contoller.train()
    plt.show()

    contoller.environment.reset()
    contoller.run_test(plotSteps=True)

    ####################################
    # MC control
    # state_0 = (0,3)
    # wind=[0,0,0,1,1,1,2,2,1,0]
    # # wind=[0,0,0,0,0,0,0,0,0,0]
    # env = Environment(state_0,wind=wind,wind_type="stochastic")
    # agent = MCAgent(epsilon=0.01,atype="4D")

    # contoller = MCControl(agent,env,vis,gamma=0.99,budget=640000,plotInterval=2000,episodeLength=80,decay=1.0)
    # contoller.train()
    # plt.show()

    # contoller.environment.reset()
    # contoller.run_test(plotSteps=True)