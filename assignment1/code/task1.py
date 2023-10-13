"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""
import numpy as np
np.seterr(all="ignore")
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

# START EDITING HERE
# You can use this space to define any helper functions that you need
def KL(x ,y):
    # print(x,y)
    if x == 0 :
        return math.log(1/(1 - y))
    elif x == 1 :
        return math.log(1/y)
    return (x * math.log(x / y) + (1 - x) * math.log((1 - x) / (1 - y)))

def KL_ucb(p, u_a, t, c = 3, tol = 1e-3) : 
    l = p
    u = 1
    q = (l + u)/2   
    target = (math.log(t) + c * math.log(math.log(t)))/u_a
    while (u - l > tol):
        q = (l + u)/2
        current = KL(p,q)
        if current < target :
            l = q
        elif current > target :
            u = q
        else :
            # print(p, u_a, t, KL(p,q) - target, "returning q")
            return q
    # print(p, u_a, t, KL(p,q) - target, target, q , "returning q")
    return q        
# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        self.ucbs = np.zeros(num_arms)
        self.u = np.zeros(num_arms)
        self.p_hat = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        t = sum(self.u)
        self.ucbs = self.p_hat + np.sqrt(2 * np.log(t) / self.u)
        # print(self.ucbs, t)
        return np.argmax(self.ucbs)
        # END EDITING HERE  
        
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.u[arm_index] += 1
        n = self.u[arm_index]
        mean = self.p_hat[arm_index]
        new_mean = ((n - 1) / n) * mean + (reward / n)
        self.p_hat[arm_index] = new_mean
        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.first_pull = True
        self.kl_ucbs = np.zeros(num_arms)
        self.u = np.zeros(num_arms)
        self.p_hat = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        if self.first_pull :
            arm = int(sum(self.u))
            if arm == (len(self.kl_ucbs) - 1) :
                self.first_pull = False
            return arm
        t = sum(self.u)
        for i in range(len(self.kl_ucbs)):
            self.kl_ucbs[i] = KL_ucb(self.p_hat[i], self.u[i], t, c=0)
        arm = np.argmax(self.kl_ucbs)
        # print(self.p_hat, t)
        return arm
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.u[arm_index] += 1
        n = self.u[arm_index]
        mean = self.p_hat[arm_index]
        new_mean = ((n - 1) / n) * mean + (reward / n)
        self.p_hat[arm_index] = new_mean
        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.sa = np.zeros(num_arms)
        self.fa = np.zeros(num_arms)
        self.t_samples = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        for i in range(len(self.t_samples)) :
            self.t_samples[i] = np.random.beta(self.sa[i] + 1, self.fa[i] + 1)
        return np.argmax(self.t_samples)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.sa[arm_index] += reward
        self.fa[arm_index] += 1 - reward
        # END EDITING HERE
