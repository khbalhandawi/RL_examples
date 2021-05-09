import numpy as np
from environment import Environment
from agent import Agent
from visualizer import Visualizer
from control import TDControlOnPolicy
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

    state_0 = (0,3)
    env = Environment(state_0)
    agent = Agent(epsilon=0.1,atype="8D")

    contoller = TDControlOnPolicy(agent,env,vis,budget=800000,alpha=0.1,plotInterval=2000)
    contoller.train()
    plt.show()

    contoller.environment.reset()
    contoller.run_test(plotSteps=True)