# BernoulliArm and BernoulliBandit
# do not modify!

import numpy as np
import matplotlib.pyplot as plt
from bernoulli_bandit import BernoulliArm

class FaultyBandit:
  def __init__(self, probs=[0.3, 0.5, 0.7], fault=0.2):
    self.__arms = [BernoulliArm(p) for p in probs]
    self.__reward = 0
    self.__faulty_arm_prob = 0.5
    self.__fault = fault # probability of a faulty pull

  def pull(self, index):
    correct_pull = np.random.binomial(1, 1 - self.__fault)
    reward = 0
    if correct_pull == 1:
      reward = self.__arms[index].pull()
    else:
      reward = np.random.binomial(1, self.__faulty_arm_prob)
    self.__reward += reward
    return reward

  def reward(self):
    return self.__reward

  def num_arms(self):
    return len(self.__arms)
