import numpy as np
import matplotlib.pyplot as plt
import math
import time

class Visualizer:
    
    #HELPFUL FUNCTIONS
    
    def create_window(self):
        '''
        Creates window and assigns self.display variable
        '''

        self.fig = plt.figure(figsize=(10,10))

        self.ax_1 = self.fig.add_subplot(2, 1, 1)
        self.ax_1.set_xlabel('episodes',fontsize=14)
        self.ax_1.set_ylabel('reward',fontsize=14)

        # Fitting curve
        self.ax_2 = self.fig.add_subplot(2, 1, 2)
        self.ax_2.set_xlabel('s_A',fontsize=14)
        self.ax_2.set_ylabel('s_B',fontsize=14)

    def close_window(self):
        plt.close(self.fig)


    def clear_lines(self):
        '''
        remove all lines and collections
        input:      ax: plt axis object
        '''
        for artist in self.ax_1.lines + self.ax_1.collections:
            artist.remove()

    def plot_policy(self,pi,ns=21):

        plt.cla()
        n_states = len(pi.keys())
        ns = ns # legnth of state vector
        # Create an empty numpy array with the right dimensions
        x = np.zeros((n_states, 1))
        y = np.zeros((n_states, 1))
        z = np.zeros((n_states, 1))
        i = 0
        for keys, values in pi.items():
            # print(keys)
            # print(values)
            x[i] = keys[0]
            y[i] = keys[1]
            z[i] = values

            i += 1

        X = np.reshape(x,(ns,ns)); Y = np.reshape(y,(ns,ns))
        Z = np.reshape(z,np.shape(X))

        c = self.ax_2.contourf(X,Y,Z, cmap=plt.cm.jet,levels=11)

        # Make a colorbar for the ContourSet returned by the contourf call.
        self.ax_2.set_xlabel('State_A')
        self.ax_2.set_ylabel('State_B')

    def draw(self, pi, time, rewards):

        self.clear_lines()
        self.ax_1.plot(time,rewards,'-b')
        self.plot_policy(pi,ns=((self.ub_A - self.lb_A)+1))

        plt.draw()
        plt.pause(0.001)
                
        return None

    def __init__(self,lb_A,ub_A,lb_B,ub_B):

        self.lb_A = lb_A
        self.ub_A = ub_A