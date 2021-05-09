import numpy as np
from environment import Environment
from agent import Agent
from visualizer import Visualizer
from control import TDControl

if __name__ == "__main__":

    lb_A=0; ub_A=20; lb_B=0; ub_B=20; lb_a=-5; ub_a=5
    lA = ub_A - lb_A + 1; lB = ub_B - lb_B + 1

    vis = Visualizer(lb_A,ub_A,lb_B,ub_B)
    vis.create_window()
    # vis.visualize_gridworld()

    state_0 = (10,10)
    env = Environment(state_0)
    agent = Agent(lb_A,ub_A,lb_B,ub_B,lb_a,ub_a,epsilon=0.5)

    contoller = TDControl(agent,env,vis,gamma=0.9,budget=2000000,alpha=1.0,plotInterval=1000)
    contoller.train()