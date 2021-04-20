import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import pickle

def p_calc(args,params):
    '''
    input:      args: list of states
                params: list of state vectors, action vectors, 
                    prior distribution, and lower and upper bounds
    '''
    [state_A,state_B] = args
    [states,actions,lb,ub,prior] = params

    print("State A: %02d | State B: %02d" %(state_A,state_B))
    P = {}
    for action in actions: # Action index
        
        temp = {}
        if action <= state_A and -action <= state_B and action + state_B <= ub and -action + state_A <= ub:
            for request_A in range(len(states)): # State A request index
                for return_A in range(len(states)): # State A return index
                    for request_B in range(len(states)): # State B request index
                        for return_B in range(len(states)): # State B return index

                                # Next state function
                                s_A = state_A - request_A + return_A - action
                                s_B = state_B - request_B + return_B + action

                                s_prime_A = np.clip(s_A, lb, ub) # clamp values between 0 and 20
                                s_prime_B = np.clip(s_B, lb, ub) # clamp values between 0 and 20
                                
                                # Reward function
                                r_A = (max(state_A - request_A,0) * 10.0)
                                r_B = (max(state_B - request_B,0) * 10.0)
                                r = r_A + r_B - (abs(action) * 2.0)

                                # print("     State A: %02d | State B: %02d | r: %04d" %(s_prime_A,s_prime_B,r))
                                temp[((s_prime_A, s_prime_B),r)] = temp.get(((s_prime_A, s_prime_B),r),0)
                                temp[((s_prime_A, s_prime_B),r)] += prior[(request_A, return_A, request_B, return_B)]

            P[action] = temp

    with open('P' + str(state_A)+str('_')+str(state_B), 'wb') as f:
        pickle.dump(P, f, protocol=-1)

def parallel_sampling(grid,params):
    '''
    input:      grid: list of all possible states
                params: list of state vectors, action vectors, 
                    prior distribution, and lower and upper bounds
    '''
    from joblib import Parallel, delayed
    import multiprocessing
    from multiprocessing import Process, Pool
    import subprocess
    
    num_threads = int(6)

    args = []
    for arg in grid:
        args += [(arg,params)]

    with Pool(num_threads) as pool:
        pool.starmap(p_calc, args)

def serial_sampling(grid,params):
    '''
    input:      grid: list of all possible states
                params: list of state vectors, action vectors, 
                    prior distribution, and lower and upper bounds
    '''
    for arg in grid:
        p_calc(arg,params)

def dealership_model(s,a,lb=0,ub=20):
    '''
    input:      s: dict {loc_A:, loc_B}
                a: dict {loc_A:, loc_B}

    returns:    r: float
                s_prime: dict {loc_A:, loc_B}
    '''

    cars_requested_A = np.random.choice(np.arange(lb, ub+1), p=poisson(3))
    cars_requested_B = np.random.choice(np.arange(lb, ub+1), p=poisson(4))

    cars_returned_A = np.random.choice(np.arange(lb, ub+1), p=poisson(3))
    cars_returned_B = np.random.choice(np.arange(lb, ub+1), p=poisson(2))

    # Next state function
    s_A = s['loc_A'] - cars_requested_A + cars_returned_A - a
    s_B = s['loc_B'] - cars_requested_B + cars_returned_B + a

    s_prime_A = np.clip(s_A, lb, ub+1) # clamp values between 0 and 20
    s_prime_B = np.clip(s_B, lb, ub+1) # clamp values between 0 and 20
    s_prime = {'loc_A': s_prime_A, 'loc_B': s_prime_B}

    # Reward function
    r_A = (max(s['loc_A'] - cars_requested_A,0) * 10.0)
    r_B = (max(s['loc_B'] - cars_requested_B,0) * 10.0)
    r = r_A + r_B - (abs(a) * 2.0)

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

def policy_evaluation(pi,V,gamma=0.9,Theta=0.00001,lb=0,ub=20):
    '''
    Computes the state value for a given policy
    inputs:     S: list of dicts
                pi: list of dicts
                gamma: discounting rate
    returns:    V: estimate of state values
    '''
    
    states = np.arange(lb,ub+1)

    counter = 1
    while True:
        Delta = 0
        print("Calculating loop " + str(counter))

        Delta = 0
        # loop over all states
        for i in range(len(states)): # State A index
            for j in range(len(states)): # State B index

                with open('P' + str(i)+str('_')+str(j), 'rb') as f:
                    P = pickle.load(f)

                a = pi[(i, j)]
                temp = P.get(a,{})
                old_value = V[(i, j)]
                V[(i, j)] = 0

                for keys, values in temp.items():
                    (states, reward) = keys
                    probability = values
                    V[(i, j)] += (reward + gamma * V[states]) * probability
                
                Delta = max(Delta, abs(V[(i, j)] - old_value))

        print("Delta = " + str(Delta))
        if Delta < Theta:
            return V
        
        counter += 1

def policy_improvement(pi,V,gamma=0.9,Theta=0.00001,lb=0,ub=20,lb_a=-5,ub_a=5):
    '''
    Computes the state value for a given policy
    inputs:     S: list of dicts
                pi: list of dicts
                gamma: discounting rate
    returns:    V: estimate of state values
    '''
    
    len_states = len(np.arange(lb,ub+1))
    actions = np.arange(lb_a,ub_a+1)

    counter = 1
    while True:
        policy_stable = True
        print("Calculating policy loop " + str(counter))

        # loop over all states
        for i in range(len_states): # State A index
            for j in range(len_states): # State B index
                
                old_action = pi[(i, j)]
                with open('P' + str(i)+str('_')+str(j), 'rb') as f:
                    P = pickle.load(f)

                # objective function to be maximized
                action_i = 0; action_values = [0]*len(actions)
                for action in actions:
                    
                    if action <= i and -action <= j and action + j <= 20 and -action + i <= 20:
                        temp = P.get(action,{})

                        for keys, values in temp.items():
                            (states, reward) = keys
                            probability = values
                            action_values[action_i] += (reward + gamma * V[states]) * probability
                    else:
                        action_values[action_i] = -999

                    action_i += 1
                    
                # update policy
                pi[(i, j)] = actions[np.argmax(action_values)]
                
                print("     state A: %4d | state B: %4d | optimal action %1d" %(i,j,pi[(i, j)]))

                if pi[(i, j)] != old_action:
                    policy_stable = False

        if policy_stable:
            return pi

        counter += 1

def init(lb=0,ub=20,lb_a=-5,ub_a=5):
    '''
    Initializes the set of actions and states
    returns:    pi: list of dicts
    '''

    actions = np.arange(lb_a,ub_a+1)
    states = np.arange(lb,ub+1)

    # Generate initial policy pi
    pi = {}; V = {}
    for state_A in states:
        for state_B in states:
            pi[(state_A, state_B)] = np.random.randint(lb_a,ub_a) # random policy
            # pi[(state_A, state_B)] = 0
            V[(state_A, state_B)] = 10 * np.random.random()

    return pi,V
            
def compute_prior_P(lb=0,ub=20, save=True):
    '''
    Initializes the set of actions and states
    returns:    P: dict: prior probabilities of events
    '''

    if save:
        states = np.arange(lb,ub+1)

        # Generate four argument probability function P
        p_cars_requested_A = poisson(3)
        p_cars_returned_A = poisson(3)

        p_cars_requested_B = poisson(4)
        p_cars_returned_B = poisson(2)

        p_A = {}; p_B = {}
        for i in range(len(states)): # State request index
            for j in range(len(states)): # State return index
                    p_A[(i,j)] = p_cars_requested_A[i] * p_cars_returned_A[j] # probability of this state A_k occuring
                    p_B[(i,j)] = p_cars_requested_B[i] * p_cars_returned_B[j] # probability of this state B_k occuring


        P = {}; 
        for i in range(len(states)): # State A request index
            for j in range(len(states)): # State A return index
                for m in range(len(states)): # State B request index
                    for n in range(len(states)): # State B return index
                        P[(i, j, m, n)] = p_A[i, j] * p_B[m, n]

        with open('prior_distribution.pkl', 'wb') as filename:
            pickle.dump(P, filename, protocol=-1)

    else:
        with open('prior_distribution.pkl', 'rb') as filename:
            P = pickle.load(filename)

    return P

def compute_posterior_P(lb=0,ub=20,lb_a=-5,ub_a=5):
    '''
    Initializes the set of actions and states
    returns:    P_a: dict: Posterior probabilities of events
    '''

    prior = compute_prior_P(lb=lb,ub=ub, save=True)

    states = np.arange(lb,ub+1)
    actions = np.arange(lb_a,ub_a+1)

    S = np.zeros((len(states)*len(states),2))

    k = 0
    for i in range(len(states)): # State A index
        for j in range(len(states)): # State B index

            S[k,0] = i
            S[k,1] = j

            k += 1

    params = [states,actions,lb,ub,prior]
    parallel_sampling(S.round().astype(int),params)

def clear_lines(ax):
    '''
    remove all lines and collections
    input:      ax: plt axis object
    '''
    for artist in ax.lines + ax.collections:
        artist.remove()

def train(pi,V,save=True):
    '''
        inputs:     pi: Initial policy
                    V: initial state value function
        return:     pi: optimal policy
                    V: final state value function
    '''
    policies = []; value_functions = []
    for q in range(5):
        print("Big loop "+str(q))

        if save:
            V = policy_evaluation(pi,V)
            pi = policy_improvement(pi,V)
            
            with open('pi'+str(q), 'wb') as f:
                pickle.dump(pi, f, protocol=-1)
            with open('V'+str(q), 'wb') as v:
                pickle.dump(V, v, protocol=-1)
        else:
            with open('pi'+str(q), 'rb') as f:
                pi = pickle.load(f)
            with open('V'+str(q), 'rb') as v:
                V = pickle.load(v)

        plot_policy(pi,V)
        policies += [pi]
        value_functions += [V]

    return policies,value_functions

def plot_policy(pi,V):

    n_states = len(pi.keys())
    ns = 21 # legnth of state vector
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

    fig, ax = plt.subplots(figsize=(10,6))
    ax.contourf(X,Y,Z, cmap=plt.cm.jet,levels=11)

    plt.show()


if __name__ == "__main__":

    pi,V = init()
    policies,value_functions = train(pi,V,save=True)
    
    pi = policies[0]; V = value_functions[4]

    # for keys, values in V.items():
    #     print(keys)
    #     print(values)

    # # run the model
    # s = {'loc_A': 10, 'loc_B': 10}
    # a = {'loc_A': 0, 'loc_B': 0}

    # fig, ax = plt.subplots(figsize=(10,6))

    # r_cumulative = 0.0; time = []; s_A = []; r_plot = []
    # for i in range(365):
    #     s,r = dealership_model(s,a,lb=0,ub=20)

    #     time += [i]
    #     s_A += [s['loc_A']]
    #     r_cumulative += r
    #     r_plot += [r_cumulative]

    #     clear_lines(ax)

    #     ax.plot(time,r_plot,'-b')

    #     plt.draw()
    #     plt.pause(0.001)