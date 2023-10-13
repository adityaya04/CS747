import argparse, time
from simulator import simulate, simulate_faulty, simulate_multi
from task1 import Algorithm, Eps_Greedy, UCB, KL_UCB, Thompson_Sampling
from task3 import FaultyBanditsAlgo 
from task4 import MultiBanditsAlgo

class Testcase:
    def __init__(self, task, probs, horizon, fault):
        self.task = task
        self.probs = probs
        self.horizon = horizon
        self.fault = fault
        self.ucb = 0
        self.kl_ucb = 0
        self.thompson = 0
        self.other = 0

def read_tc(path):
    tc = None
    with open(path, 'r') as f:
        lines = f.readlines()
        task = int(lines[0].strip())
        horizon = int(lines[1].strip())
        if task == 1:
            probs = [float(p) for p in lines[2].strip().split()]
            ucb, kl_ucb, thompson = [float(x) for x in lines[3].strip().split()]
            tc = Testcase(task, probs, horizon, fault=None)
            tc.ucb = ucb
            tc.kl_ucb = kl_ucb
            tc.thompson = thompson
        elif task == 3:
            probs = [float(p) for p in lines[2].strip().split()]
            reference = [float(p) for p in lines[3].strip().split()]
            fault = float(lines[4].strip())
            tc = Testcase(task, probs, horizon, fault)
            tc.other = reference
            tc.fault = fault
        elif task == 4:
            probs1 = [float(p) for p in lines[2].strip().split()]
            probs2 = [float(p) for p in lines[3].strip().split()]
            probs = [probs1, probs2]
            reference = [float(p) for p in lines[4].strip().split()]
            tc = Testcase(task, probs, horizon, fault=None)
            tc.other = reference
            
    return tc

def grade_task1(tc_path, algo):
    algo = algo.lower()
    tc = read_tc(tc_path)
    regrets = {}
    scores = {}
    if algo == 'ucb' or algo == 'all':
        regrets['UCB'] = simulate(UCB, tc.probs, tc.horizon)
        scores['UCB'] = 1 if regrets['UCB'] <= tc.ucb else 0
    if algo == 'kl_ucb' or algo == 'all':
        regrets['KL-UCB'] = simulate(KL_UCB, tc.probs, tc.horizon, num_sims=20)
        scores['KL-UCB'] = 1 if regrets['KL-UCB'] <= tc.kl_ucb else 0
    if algo == 'thompson' or algo == 'all':
        regrets['Thompson Sampling'] = simulate(Thompson_Sampling, tc.probs, tc.horizon)
        scores['Thompson Sampling'] = 1 if regrets['Thompson Sampling'] <= tc.thompson else 0
    
    return scores, regrets

def grade_task3(tc_path):
    tc = read_tc(tc_path)
    # reward = simulate_faulty(FaultyBanditsAlgo, tc.probs, tc.fault, tc.horizon)
    reward = simulate_faulty(FaultyBanditsAlgo, tc.probs, tc.fault, tc.horizon)
    upper_ref = tc.other[0]
    lower_ref = tc.other[1]
    score = 0
    if upper_ref >= reward > lower_ref:
        score = 0.5
    if reward > upper_ref:
        score = 1
    return score, reward 

def grade_task4(tc_path):
    tc = read_tc(tc_path)
    # reward = simulate_multi(MultiBanditsAlgo, tc.probs, tc.horizon)
    reward = simulate_multi(MultiBanditsAlgo, tc.probs, tc.horizon)
    upper_ref = tc.other[0]
    lower_ref = tc.other[1]
    score = 0
    if upper_ref >= reward > lower_ref:
        score = 0.5
    if reward > upper_ref:
        score = 1
    return score, reward 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='The task to run. Valid values are: 1, 2, 3, all')
    parser.add_argument('--algo', type=str, required=False, help='The algo to run (for task 1 only). Valid values are: ucb, kl_ucb, thompson, all')
    args = parser.parse_args()
    pass_fail = ['FAILED', 'PARTIALLY PASSED', 'PASSED']

    start = time.time()
    if args.task == '1' or args.task == 'all':
        if args.task == 'all':
            args.algo = 'all'
        if args.algo is None:
            print('Please specify an algorithm for task 1')
            exit(1)
        if args.algo.lower() not in ['ucb', 'kl_ucb', 'thompson', 'all']:
            print('Invalid algorithm')
            exit(1)

        print("="*18+" Task 1 "+"="*18)
        for i in range(1, 4):
            print(f"Testcase {i}")
            scores, regrets = grade_task1(f'testcases/task1-{i}.txt', args.algo)
            for algo, score in scores.items():
                print("{:18}: {}. Regret: {:.2f}".format(algo, pass_fail[int(score * 2)], regrets[algo]))
            print("")
    
    if args.task == '3' or args.task == 'all':
        print("="*18+" Task 3 "+"="*18)
        for i in range(1, 4):
            print(f"Testcase {i}")
            score, reward = grade_task3(f'testcases/task3-{i}.txt')
            print("Faulty Bandit Algorithm: {}. Reward: {:.2f}".format(pass_fail[int(score * 2)], reward))
            print("")

    if args.task == '4' or args.task == 'all':
        print("="*18+" Task 4 "+"="*18)
        for i in range(1, 4):
            print(f"Testcase {i}")
            score, reward = grade_task4(f'testcases/task4-{i}.txt')
            print("MultiBandit Algorithm: {}. Reward: {:.2f}".format(pass_fail[int(score * 2)], reward))
            print("")

    end = time.time()

    print("Time elapsed: {:.2f} seconds".format(end-start))
