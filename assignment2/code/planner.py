import numpy as np
import pulp
import random
import sys
import argparse
parser = argparse.ArgumentParser()

class MDP:
    def __init__(self, gamma, T, R, numStates, numActions, continuing):
        self.gamma = gamma
        self.T = T
        self.R = R
        self.numStates = numStates
        self.numActions = numActions
        self.continuing = continuing
    
    def B_star(self, Vt):
        T = self.T 
        R = self.R 
        gamma = self.gamma        
        avals = np.sum(T * (R + gamma * Vt), axis=2)
        V = np.max(avals, axis=1)
        Pi = np.argmax(avals, axis=1)
        return V, Pi

    def vi(self):
        tol = 1e-7
        V_old = np.zeros(self.numStates)
        V_new, Pi = self.B_star(V_old)
        count = 0
        delta = np.linalg.norm(V_new - V_old)
        while delta > tol :
            V_old = V_new
            V_new, Pi = self.B_star(V_old)
            count += 1
            delta = np.linalg.norm(V_new - V_old)
        return V_new, Pi

    def evaluate_value(self, Pi):
        b = np.zeros(self.numStates)
        A = np.zeros((self.numStates, self.numStates))
        for i in range(self.numStates):
            b[i] = -np.sum(self.T[i, Pi[i], :] * self.R[i, Pi[i], :])
            A[i, :] = self.T[i, Pi[i], :] * self.gamma
            A[i, i] -= 1
        V_new = np.linalg.solve(A, b)
        return V_new
    
    def action_value(self, s, a, V):
        T_sa = self.T[s, a, :]
        R_sa = self.R[s, a, :]
        q = np.sum(T_sa * (R_sa + self.gamma * V))
        return q
    
    def get_pi_star(self, V):
        Pi = np.zeros(self.numStates, dtype=np.int32)
        for i in range(self.numStates):
            avals = np.zeros(self.numActions)
            for j in range(self.numActions):
                avals[j] = self.action_value(i, j, V)
            Pi[i] = np.argmax(avals)
        return Pi
    
    def lp(self):
        prob = pulp.LpProblem('OptimalPolicyFinder', pulp.LpMaximize)
        variables = [pulp.LpVariable('V' + str(i)) for i in range(self.numStates)]
        cost = -pulp.lpSum(variables)
        prob += cost
        for i in range(self.numStates):
            for j in range(self.numActions):
                sum1 = np.sum(self.T[i, j, :] * self.R[i, j, :])
                string = pulp.lpSum(self.T[i, j, k] * self.gamma * variables[k] for k in range(self.numStates)) + sum1
                prob += variables[i] >= string
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        V_dict = {v.name[1:]: v.varValue for v in prob.variables()}
        V_star = np.array([V_dict[str(i)] for i in range(self.numStates)])
        return V_star, self.get_pi_star(V_star)
    
    def improving_actions(self, V_pi, s):
        action_values = np.array([self.action_value(s, i, V_pi) for i in range(self.numActions)])
        IA = np.where(action_values - V_pi[s] > 1e-9)[0]
        return IA.tolist()
    
    def improvable_states(self, Pi):
        V_pi = self.evaluate_value(Pi)
        improving_actions_all = [self.improving_actions(V_pi, i) for i in range(self.numStates)]
        IS_indices = np.where(np.array([len(ia) > 0 for ia in improving_actions_all]))[0]
        IS = {i: improving_actions_all[i] for i in IS_indices}
        return IS


    def hpi(self):
        Pi = np.zeros(self.numStates, dtype=np.int32)
        IS = self.improvable_states(Pi)
        
        while IS:
            improvable_states_indices = list(IS.keys())
            random_actions = np.array([random.choice(IS[i]) for i in improvable_states_indices])
            Pi[improvable_states_indices] = random_actions
            IS = self.improvable_states(Pi)
        
        Vpi = self.evaluate_value(Pi)
        return Vpi, Pi


def read_mdp(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    for line in lines:
        arr = line.split()
        if arr[0] == 'numStates' :
            numStates = int(arr[1])
        elif arr[0] == 'numActions' :
            numActions = int(arr[1])
            T = np.zeros((numStates, numActions, numStates))
            R = np.zeros((numStates, numActions, numStates))
        elif arr[0] == 'end' :
            endStates = [int(arr[i]) for i in range(1,len(arr))]
        elif arr[0] == 'discount' :
            gamma = float(arr[1])
        elif arr[0] == 'mdptype' :
            continuing = True if arr[1] == 'continuing' else False
        elif arr[0] == 'transition' :
            s1 = int(arr[1])
            s2 = int(arr[3])
            a = int(arr[2])
            T[s1,a,s2] = float(arr[5])
            R[s1,a,s2] = float(arr[4])
    return MDP(gamma, T, R, numStates, numActions, continuing)

def read_policy(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    arr = []
    for line in lines :
        arr.append(int(line[0]))
    arr = np.array(arr)
    return arr

def write_Vstar(V, Pi):
    for i in range(len(V)):
        print(f"{V[i]} {int(Pi[i])}")


if __name__ == '__main__':
    parser.add_argument('--mdp', type = str)
    parser.add_argument("--algorithm",type=str,default="vi")
    parser.add_argument("--policy",type=str)
    args = parser.parse_args()
    mdp_file = args.mdp
    mdp = read_mdp(mdp_file)
    if args.policy != None :
        policy = read_policy(args.policy)
        Vpi = mdp.evaluate_value(policy)
        write_Vstar(Vpi, policy)
        sys.exit()
    if args.algorithm == 'vi':
        V, Pi = mdp.vi()
        write_Vstar(V, Pi)
    elif args.algorithm == 'lp':
        V, Pi = mdp.lp()
        write_Vstar(V, Pi)
    elif args.algorithm == 'hpi':
        V, Pi = mdp.hpi()
        write_Vstar(V, Pi)