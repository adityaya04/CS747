# BernoulliArm and BernoulliBandit
# do not modify!

import numpy as np
import matplotlib.pyplot as plt
from bernoulli_bandit import BernoulliArm

class MultiBandit:
  def __init__(self, probs=[[0.3, 0.5, 0.7]]*2):
    self.__arms = [[BernoulliArm(p) for p in probs[j]] for j in range(len(probs))]
    self.__reward = 0
    self.__set_probs = [0.5, 0.5]
    self.__num_sets = len(self.__set_probs)
    if self.__num_sets != len(probs):
      raise Exception("MultiBandit only supports 2 sets of arms. Check `probs`.")

  def pull(self, index):
    set_chosen = np.random.choice(range(self.__num_sets), 1, False, self.__set_probs)[0]
    reward = self.__arms[set_chosen][index].pull()
    self.__reward += reward
    return reward, set_chosen

  def reward(self):
    return self.__reward

  def num_arms(self):
    return len(self.__arms)
