"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the FaultyBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon, fault): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

class FaultyBanditsAlgo:
    def __init__(self, num_arms, horizon, fault):
        # You can add any other variables you need here
        self.num_arms = num_arms
        self.horizon = horizon
        self.fault = fault # probability that the bandit returns a faulty pull
        # START EDITING HERE
        self.sa = np.zeros(num_arms)
        self.fa = np.zeros(num_arms)
        self.t_samples = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        # for i in range(len(self.t_samples)) :
        #     self.t_samples[i] = np.random.beta(self.sa[i] + 1, self.fa[i] + 1)
        # return np.argmax(self.t_samples)
        for i in range(len(self.t_samples)):
            N = self.sa[i] + self.fa[i]
            Sa = (self.sa[i] - N*self.fault/2) * (1/(1 - self.fault))
            Fa = (self.fa[i] - N*self.fault/2) * (1/(1 - self.fault))
            Sa = Sa if Sa > -1 else self.sa[i] 
            Fa = Fa if Fa > -1 else self.fa[i] 
            self.t_samples[i] = np.random.beta(Sa + 1, Fa + 1)
        return np.argmax(self.t_samples)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.sa[arm_index] += reward
        self.fa[arm_index] += 1 - reward
        #END EDITING HERE

