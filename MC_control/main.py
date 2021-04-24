import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import pickle

def dealership_model(state,lb=0,ub=20):
    '''
    input:      s: dict {loc_A:, loc_B}
                a: dict {loc_A:, loc_B}

    returns:    r: float
                s_prime: dict {loc_A:, loc_B}
    '''

    cars_requested_A = np.random.choice(np.arange(lb, ub+1), p=poisson(3,lb=lb,ub=ub))
    cars_requested_B = np.random.choice(np.arange(lb, ub+1), p=poisson(4,lb=lb,ub=ub))

    cars_returned_A = np.random.choice(np.arange(lb, ub+1), p=poisson(3,lb=lb,ub=ub))
    cars_returned_B = np.random.choice(np.arange(lb, ub+1), p=poisson(2,lb=lb,ub=ub))

    # Next state function
    s_A = state[0] - cars_requested_A + cars_returned_A
    s_B = state[1] - cars_requested_B + cars_returned_B

    s_prime_A = np.clip(s_A, lb, ub) # clamp values between 0 and 20
    s_prime_B = np.clip(s_B, lb, ub) # clamp values between 0 and 20

    # Reward function
    r_A = (min(cars_requested_A,state[0]) * 10.0)
    r_B = (min(cars_requested_B,state[1]) * 10.0)
    r = r_A + r_B
    
    s_prime = (s_prime_A, s_prime_B)

    return s_prime, r

def poisson(lamda,lb=0,ub=20):
    '''
    Compute the probability of value in range [lb,ub]
    '''
    p_values = np.zeros(len(range(lb,ub+1)))
    for i in range(lb,ub+1):
        p_i = ((lamda**i)/np.math.factorial(i))*np.exp(-lamda)
        p_values[i] = p_i

    return p_values

def init(lb=0,ub=20,lb_a=-5,ub_a=5):
    '''
    Initializes the set of actions and states
    returns:    pi: list of dicts
    '''

    actions = np.arange(lb_a,ub_a+1)
    states = np.arange(lb,ub+1)

    # Generate initial policy pi
    pi = {}; Q = {}; C = {}; a = {}
    for state_A in states:
        for state_B in states:
            
            state = (state_A,state_B)
            a[state] = get_possible_acts(state,actions,lb=lb,ub=ub)

            Q_a = []
            for possible_action in a[state]:

                q_init = np.random.random()
                Q_a += [q_init]
                Q[(state,possible_action)] = q_init
                Q[(state,possible_action)] = 0
                C[(state,possible_action)] = 0

            pi[state] = a[state][np.argmax(Q_a)]
            pi[state] = 0

    return Q,C,pi,a

def get_possible_acts(state,actions,lb=0,ub=20):
    # decide on action
    possible_actions = []
    for action in actions:
        if state[0] - action >= lb and state[1] + action >= lb and action + state[1] <= ub and - action + state[0] <= ub:
            possible_actions += [action]

    return possible_actions

def get_action_b(best_action,state,a,epsilon=0.1):

    if np.random.rand() > epsilon and best_action in a[state]:
        soft_action = best_action
    else:
        soft_action = np.random.choice(a[state])

    num_actions = len(a[state])

    # Update behavioral policy
    if best_action in a[state]:

        if soft_action == best_action:
            prob = 1 - epsilon + epsilon/num_actions
        else:
            prob = epsilon/num_actions
    else:
        prob = 1/num_actions

    return prob,soft_action

def get_action_pi(best_action,state,a):

    if best_action in a[state]:
        greedy_action = best_action
    else:
        greedy_action = np.random.choice(a[state])

    return greedy_action

def clear_lines(ax):
    '''
    remove all lines and collections
    input:      ax: plt axis object
    '''
    for artist in ax.lines + ax.collections:
        artist.remove()

def plot_policy(ax,pi,ns=21):

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

    c = ax.contourf(X,Y,Z, cmap=plt.cm.jet,levels=11)

    # Make a colorbar for the ContourSet returned by the contourf call.
    ax.set_xlabel('State_A')
    ax.set_ylabel('State_B')

    plt.draw()
    plt.pause(0.001)

def episode(pi,state_0,a,T=100,lb=0,ub=20):

    state = state_0
    s = []; A = []; R = [0]; probs = []
    for i in range(T):

        best_action = pi[state]
        prob,soft_action = get_action_b(best_action,state,a)

        # print("state A: %2d | state B: %2d | a: %2d | p: %3f" %(state_A,state_B,soft_action,prob))
        s += [state] # starting from S_0

        state_action = (state[0] - soft_action, state[1] + soft_action)

        A += [soft_action]
        r_move = abs(soft_action) * 2.0

        # print("best action: %2d | soft action: %2d" %(best_action,soft_action))
        # print("state A: %2d | state B: %2d" %(state_A,state_B))

        state,r_sell = dealership_model(state_action,lb=lb,ub=ub)

        r = r_sell - r_move

        R += [r] # starting from R1
        probs += [prob]

    return s, A, R, probs

def train(Q,C,pi,state_0,a,N,gamma=1,lb=0,ub=20):

    [s, A, R, probs] = episode(pi,state_0,a,lb=lb,ub=ub)
    G = 0
    W = 1

    T = len(s)

    for t in range(T-1,-1,-1):

        S_t = s[t]; A_t = A[t]

        # print("state A: %2d | state B: %2d | r: %2f" %(Sa_t,Sb_t,R[t+1]))

        G = gamma*G + R[t+1]
        C[(S_t,A_t)] += W
        Q[(S_t,A_t)] += ( W * (G - Q[(S_t,A_t)]) ) / C[(S_t,A_t)]

        Q_S_a = []
        for action in a[S_t]:
            Q_S_a += [Q[(S_t,action)]]

        # maxs = np.argwhere(Q_S_a == np.amax(Q_S_a)).flatten().tolist()
        # for max_arg in maxs:
        #     if a[S_t][max_arg] != pi[S_t]:
        #         pi[S_t] = max_arg
        #         break

        pi[S_t] = a[S_t][np.argmax(Q_S_a)]
        # if A_t != pi[S_t]:
        #     break

        W /= probs[t]

    return Q,C,pi,sum(R)

if __name__ == "__main__":

    lb = 0; ub = 20
    Q,C,pi,a = init(lb=lb,ub=ub)
    state_0 = (10, 10)

    # fig_1, ax_1 = plt.subplots(figsize=(10,6))
    fig = plt.figure(figsize=(10,10))

    ax_1 = fig.add_subplot(2, 1, 1)
    ax_1.set_xlabel('episodes',fontsize=14)
    ax_1.set_ylabel('reward',fontsize=14)

    # Fitting curve
    ax_2 = fig.add_subplot(2, 1, 2)
    ax_2.set_xlabel('s_A',fontsize=14)
    ax_2.set_ylabel('s_B',fontsize=14)

    plot_interval = 20

    episodes = []; r_plot = []

    for N in range(100000):

        Q,C,pi,G = train(Q,C,pi,state_0,a,N,gamma=1,lb=lb,ub=ub)

        if N % plot_interval == 0:
            print("Episode %i" %(N))
            episodes += [N]; r_plot += [G]

            clear_lines(ax_1)
            ax_1.plot(episodes,r_plot,'-b')
            plot_policy(ax_2,pi,ns=((ub - lb)+1))

            plt.draw()
            plt.pause(0.001)