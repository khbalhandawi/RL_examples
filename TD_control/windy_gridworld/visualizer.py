import numpy as np
import pygame
import math
import time
import matplotlib.pyplot as plt

class Visualizer:
    
    #HELPFUL FUNCTIONS
    
    def create_window(self):
        '''
        Creates window and assigns self.display variable
        '''
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("gridworld")

    def create_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(10,6))
        self.ax.set_xlabel('episodes',fontsize=14)
        self.ax.set_ylabel('episode length',fontsize=14)
    
    def setup(self):
        '''
        Does things which occur only at the beginning
        '''
        self.cell_edge = 90
        self.width = self.lx*self.cell_edge
        self.height = self.ly*self.cell_edge
        self.create_window()
        self.create_figure()
        self.window = True

    def clear_lines(self):
        '''
        remove all lines and collections
        input:      ax: plt axis object
        '''
        for artist in self.ax.lines + self.ax.collections:
            artist.remove()

    def close_window(self):
        self.window = False
        pygame.quit()
        plt.close(self.fig)

    def draw_reward(self,x,y):

        self.clear_lines()
        self.ax.plot(x,y,'-b')

        plt.draw()
        plt.pause(0.001)

    def draw_policy(self,pi,state = np.array([])):

        self.display.fill(0)
        for i in range(self.lx):
            for j in range(self.ly):
                if self.data[i,j] != -1:
                    if self.data[i,j] == 0:
                        color = (255,0,0)
                    elif self.data[i,j] == 1:
                        color = (255,255,0)
                    elif self.data[i,j] == 2:
                        color = (0,255,0)
                    pygame.draw.rect(self.display,color,((i*self.cell_edge,j*self.cell_edge),(self.cell_edge,self.cell_edge)),1)

                if len(state)>0:
                    pygame.draw.rect(self.display,(0,0,255),((state[0]*self.cell_edge,state[1]*self.cell_edge),(self.cell_edge,self.cell_edge)),1)

                coord = (i,j)
                action = pi[coord]
                s = self.cell_edge
                l = 0.3
                if action == "U":
                    self.draw_arrow(self.display,(255,255,255),(i*s + s/2, (j+l)*s),(i*s + s/2, (j+1-l)*s))
                elif action == "D":
                    self.draw_arrow(self.display,(255,255,255),(i*s + s/2, (j+1-l)*s),(i*s + s/2, (j+l)*s))
                elif action == "L":
                    self.draw_arrow(self.display,(255,255,255),((i+1-l)*s, j*s + s/2),((i+l)*s, j*s + s/2))
                elif action == "R":
                    self.draw_arrow(self.display,(255,255,255),((i+l)*s, j*s + s/2),((i+1-l)*s, j*s + s/2))

                elif action == "LD":
                    self.draw_arrow(self.display,(255,255,255),((i+1-l)*s, (j+1-l)*s),((i+l)*s, (j+l)*s))
                elif action == "LU":
                    self.draw_arrow(self.display,(255,255,255),((i+1-l)*s, (j+l)*s),((i+l)*s, (j+1-l)*s))
                elif action == "RD":
                    self.draw_arrow(self.display,(255,255,255),((i+l)*s, (j+1-l)*s),((i+1-l)*s, (j+l)*s))
                elif action == "RU":
                    self.draw_arrow(self.display,(255,255,255),((i+l)*s, (j+l)*s),((i+1-l)*s, (j+1-l)*s))

        pygame.display.update()
        time.sleep(0.01)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.loop = False
                self.close_window()
                return 'stop'
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.loop = False
                
        return None

    def draw_arrow(self,screen, colour, start, end):
        pygame.draw.line(screen,colour,start,end,2)
        rotation = math.degrees(math.atan2(start[1]-end[1], end[0]-start[0]))+90
        pygame.draw.polygon(screen, (255, 255, 255), ((end[0]+20*math.sin(math.radians(rotation)), end[1]+20*math.cos(math.radians(rotation))), (end[0]+20*math.sin(math.radians(rotation-120)), end[1]+20*math.cos(math.radians(rotation-120))), (end[0]+20*math.sin(math.radians(rotation+120)), end[1]+20*math.cos(math.radians(rotation+120)))))

    def visualize_gridworld(self, state = np.array([])):
        '''
        Draws gridworld in a pygame window
        '''
        if self.window == False:
            self.setup()
        self.loop = True
        while(self.loop):
            ret = self.draw(state)
            if ret!=None:
                return ret
    
    #CONSTRUCTOR
    def __init__(self,data,lx=10,ly=10):
        self.lx = lx; self.ly = ly
        self.data = data
        self.window = False