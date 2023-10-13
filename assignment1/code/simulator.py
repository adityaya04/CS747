# simulator
# do not modify! (except final few lines)

from bernoulli_bandit import *
from faulty_bandit import *
from multi_bandit import *
from task1 import Algorithm, Eps_Greedy, UCB, KL_UCB, Thompson_Sampling
from task2 import task2
from task3 import FaultyBanditsAlgo 
from task4 import MultiBanditsAlgo 
from multiprocessing import Pool
import time

def single_sim(seed=0, ALGO=Algorithm, PROBS=[0.3, 0.5, 0.7], HORIZON=1000):
  np.random.seed(seed)
  np.random.shuffle(PROBS)
  bandit = BernoulliBandit(probs=PROBS)
  algo_inst = ALGO(num_arms=len(PROBS), horizon=HORIZON)
  for t in range(HORIZON):
    arm_to_be_pulled = algo_inst.give_pull()
    reward = bandit.pull(arm_to_be_pulled)
    algo_inst.get_reward(arm_index=arm_to_be_pulled, reward=reward)
  return bandit.regret()

def single_sim_faulty(seed=0, ALGO=Algorithm, PROBS=[0.3, 0.5, 0.7], FAULT=0.2, HORIZON=1000):
  np.random.seed(seed)
  np.random.shuffle(PROBS)
  bandit = FaultyBandit(probs=PROBS, fault=FAULT)
  algo_inst = ALGO(num_arms=len(PROBS), horizon=HORIZON, fault=FAULT)
  for t in range(HORIZON):
    arm_to_be_pulled = algo_inst.give_pull()
    reward = bandit.pull(arm_to_be_pulled)
    algo_inst.get_reward(arm_index=arm_to_be_pulled, reward=reward)
  return bandit.reward()

def single_sim_multi(seed=0, ALGO=Algorithm, PROBS=[[0.3, 0.5, 0.7]]*2, HORIZON=1000):
  np.random.seed(seed)
  # PROBS = np.random.shuffle(np.array(PROBS), axis=1)
  bandit = MultiBandit(probs=PROBS)
  algo_inst = ALGO(num_arms=len(PROBS[0]), horizon=HORIZON)
  for t in range(HORIZON):
    arm_to_be_pulled = algo_inst.give_pull()
    reward, set_chosen = bandit.pull(arm_to_be_pulled)
    algo_inst.get_reward(arm_index=arm_to_be_pulled, set_pulled=set_chosen, reward=reward)
  return bandit.reward()

def simulate(algorithm, probs, horizon, num_sims=50):
  """simulates algorithm of class Algorithm
  for BernoulliBandit bandit, with horizon=horizon
  """
  
  def multiple_sims(num_sims=50):
    with Pool(10) as pool:
      sim_out = pool.starmap(single_sim,
        [(i, algorithm, probs, horizon) for i in range(num_sims)])
    return sim_out 

  sim_out = multiple_sims(num_sims)
  regrets = np.mean(sim_out)

  return regrets

def simulate_faulty(algorithm, probs, fault, horizon, num_sims=50):
  """simulates algorithm of class Algorithm
  for BernoulliBandit bandit, with horizon=horizon
  """
  
  def multiple_sims(num_sims=50):
    with Pool(10) as pool:
      sim_out = pool.starmap(single_sim_faulty,
        [(i, algorithm, probs, fault, horizon) for i in range(num_sims)])
    return sim_out 

  sim_out = multiple_sims(num_sims)
  rewards = np.mean(sim_out)

  return rewards 

def simulate_multi(algorithm, probs, horizon, num_sims=50):
  """simulates algorithm of class Algorithm
  for BernoulliBandit bandit, with horizon=horizon
  """
  
  def multiple_sims(num_sims=50):
    with Pool(10) as pool:
      sim_out = pool.starmap(single_sim_multi,
        [(i, algorithm, probs, horizon) for i in range(num_sims)])
    return sim_out 

  sim_out = multiple_sims(num_sims)
  rewards = np.mean(sim_out)

  return rewards

def task1(algorithm, probs, num_sims=50):
  """generates the plots and regrets for task1
  """
  horizons = [2**i for i in range(10, 19)]
  regrets = []
  for horizon in horizons:
    regrets.append(simulate(algorithm, probs, horizon, num_sims))

  print(regrets)
  plt.plot(horizons, regrets)
  plt.title("Regret vs Horizon")
  plt.savefig("task1-{}-{}.png".format(algorithm.__name__, time.strftime("%Y%m%d-%H%M%S")))
  plt.clf()

def task3(algorithm, probs, fault, num_sims=50):
  """generates the plots and rewards for task3
  """
  horizons = [2**i for i in range(10, 19)]
  rewards = []
  for horizon in horizons:
    rewards.append(simulate_faulty(algorithm, probs, fault, horizon, num_sims))

  print(rewards)

def task4(algorithm, probs, num_sims=50):
  """generates the plots and rewards for task4
  """
  horizons = [2**i for i in range(10, 19)]
  rewards = []
  for horizon in horizons:
    rewards.append(simulate_multi(algorithm, probs, horizon, num_sims))

  print(rewards)
  

if __name__ == '__main__':
  ### EDIT only the following code ###

  # TASK 1 STARTS HERE
  # Note - all the plots generated for task 1 & 3 will be for the following 
  # bandit instance:
  # 20 arms with uniformly distributed means

  task1probs = [i/20 for i in range(20)]
  # task1(Eps_Greedy, task1probs, 1)
  # task1(UCB, task1probs)
  # task1(KL_UCB, task1probs)
  # task1(Thompson_Sampling, task1probs)
  # TASK 1 ENDS HERE

  # TASK 3 STARTS HERE
  # Note - all the results generated for task 3 shall use a fault 
  # probability of 0.2

  # task3probs = [i/20 for i in range(20)]
  # fault = 0.2
  # task3(FaultyBanditsAlgo, task3probs, fault)
  # TASK 3 ENDS HERE

  # TASK 4 STARTS HERE
  # Note - all the results generated for task 4 will be for the following 
  # bandit instance:
  # 2 bandits having 20 arms with uniformly distributed means 

  task4probs = [[0.15, 0.05, 0.5, 0.6, 0.35, 0.85, 0.75, 0.3, 0.1, 
  0.45, 0.0, 0.55, 0.9, 0.2, 0.8, 0.65, 0.95, 0.4, 0.7, 0.25], [0.3, 
  0.55, 0.5, 0.0, 0.2, 0.25, 0.95, 0.1, 0.8, 0.6, 0.05, 0.45, 0.7, 
  0.65, 0.35, 0.4, 0.15, 0.85, 0.75, 0.9]]
  task4(MultiBanditsAlgo, task4probs)
  #TASK 4 ENDS HERE

