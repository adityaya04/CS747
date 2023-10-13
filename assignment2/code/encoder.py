import numpy as np
import argparse
parser = argparse.ArgumentParser()

coordinates = {'01':[0,3], '02':[1,3], '03':[2,3], '04':[3,3],
               '05':[0,2], '06':[1,2], '07':[2,2], '08':[3,2],
               '09':[0,1], '10':[1,1], '11':[2,1], '12':[3,1],
               '13':[0,0], '14':[1,0], '15':[2,0], '16':[3,0] }
cells_dict = {tuple(i): j for j, i in coordinates.items()}
def cells(arr):
    return cells_dict[(arr[0],arr[1])]

def move_player(a, x):
    x1 = x[0]
    y1 = x[1]
    if a%4 == 0:
        x1 -= 1
    elif a%4 == 1:
        x1 += 1
    elif a%4 == 2:
        y1 += 1
    else :
        y1 -= 1
    return [x1,y1]

def vertical_inline(B1, B2, R):
    if(B1[0] == B2[0]) and (B1[0] == R[0]):
        if (R[1] <= max(B1[1],B2[1])) and (R[1] >= min(B1[1],B2[1])):
            return True
    return False
def horizontal_inline(B1, B2, R):
    if(B1[1] == B2[1]) and (B1[1] == R[1]):
        if (R[0] <= max(B1[0],B2[0])) and (R[0] >= min(B1[0],B2[0])):
            return True
    return False
def diag_inline(B1,B2,R):
    deltax = B1[0] - B2[0]
    deltay = B1[1] - B2[1]
    deltar1x = B1[0] - R[0]
    deltar1y = B1[1] - R[1]
    if deltax == deltay :
        if deltar1x == deltar1y :
            if (R[0] <= max(B1[0],B2[0])) and (R[0] >= min(B1[0],B2[0])):
                return True
    elif deltax == -deltay :
        if deltar1x == -deltar1y :
            if (R[0] <= max(B1[0],B2[0])) and (R[0] >= min(B1[0],B2[0])):
                return True
    return False


def check_inline(B1, B2, R):
    if vertical_inline(B1,B2,R):
        return True
    elif horizontal_inline(B1,B2,R):
        return True
    elif diag_inline(B1,B2,R):
        return True
    return False

class Game:
    def __init__(self):
        self.numStates = 8194
        self.numActions = 10
        self.q = 0
        self.p = 1
        self.states = {}
        self.state_indices = None
        self.R_policy = {}
        self.T = None
        self.R = None

    def read_R_policy(self, filename):
        f = open(filename, 'r')
        _ = f.readline()
        lines = f.readlines()
        for i in range(8192):
            arr = lines[i].split()
            self.states[i] = arr[0]
            self.R_policy[arr[0]] = [float(arr[i]) for i in range(1,5)]
        self.states[8192] = 'END'
        self.states[8193] = 'WIN'
        self.state_indices = {i : j for j, i in self.states.items()}
        self.T = np.zeros((self.numStates, self.numActions, self.numStates))
        self.R = np.zeros((self.numStates, self.numActions, self.numStates))

    def possible_transitions(self, state, action):
        R = coordinates[state[4:6]]
        B1 = coordinates[state[0:2]]
        B2 = coordinates[state[2:4]]
        R_action = self.R_policy[state]
        for i in range(4):
            if R_action[i] > 0 :
                R_new = move_player(i, R)
                if R_new[0] > 3 or R_new[0] < 0 or R_new[1] > 3 or R_new[1] < 0:
                    self.T[self.state_indices[state], action, self.state_indices['END']] += R_action[i] 
                else :
                    if action < 4 :
                        B1_new = move_player(action, B1)
                        if B1_new[0] > 3 or B1_new[0] < 0 or B1_new[1] > 3 or B1_new[1] < 0:
                            self.T[self.state_indices[state], action, self.state_indices['END']] = 1
                            continue
                        s1 = cells(B1_new)+cells(B2)+cells(R_new)+state[6]
                        if state[6] == '2':
                            p_move = 1 - self.p
                            self.T[self.state_indices[state], action, self.state_indices[s1]] += p_move*R_action[i]
                            self.T[self.state_indices[state], action, self.state_indices['END']] += (1 - p_move)*R_action[i]
                        else :
                            tackle = False
                            if (R_new[0] == B1_new[0]) and (R_new[1] == B1_new[1]):
                                tackle = True
                            elif (R_new[0] == B1[0]) and (R_new[1] == B1[1]) and (B1_new[0] == R[0]) and (B1_new[1] == R[1]):
                                tackle = True
                            if tackle :
                                p_move = (0.5 - self.p)
                                self.T[self.state_indices[state], action, self.state_indices[s1]] += p_move*R_action[i]
                                self.T[self.state_indices[state], action, self.state_indices['END']] += (1 - p_move)*R_action[i]
                            else :
                                p_move = 1 - 2*self.p
                                self.T[self.state_indices[state], action, self.state_indices[s1]] += p_move*R_action[i]
                                self.T[self.state_indices[state], action, self.state_indices['END']] += (1 - p_move)*R_action[i]
                    elif action < 8 :
                        B2_new = move_player(action, B2)
                        if B2_new[0] > 3 or B2_new[0] < 0 or B2_new[1] > 3 or B2_new[1] < 0:
                            self.T[self.state_indices[state], action, self.state_indices['END']] = 1
                            continue    
                        s1 = cells(B1)+cells(B2_new)+cells(R_new)+state[6]
                        if state[6] == '1':
                            p_move = 1 - self.p
                            self.T[self.state_indices[state], action, self.state_indices[s1]] += p_move*R_action[i]
                            self.T[self.state_indices[state], action, self.state_indices['END']] += (1 - p_move)*R_action[i]
                        else :
                            tackle = False
                            if (R_new[0] == B2_new[0]) and (R_new[1] == B2_new[1]):
                                tackle = True
                            elif (R_new[0] == B2[0]) and (R_new[1] == B2[1]) and (B2_new[0] == R[0]) and (B2_new[1] == R[1]):
                                tackle = True
                            if tackle :
                                p_move = (0.5 - self.p)
                                self.T[self.state_indices[state], action, self.state_indices[s1]] += p_move*R_action[i]
                                self.T[self.state_indices[state], action, self.state_indices['END']] += (1 - p_move)*R_action[i]
                            else :
                                p_move = 1 - 2*self.p
                                self.T[self.state_indices[state], action, self.state_indices[s1]] += p_move*R_action[i]
                                self.T[self.state_indices[state], action, self.state_indices['END']] += (1 - p_move)*R_action[i]
                    elif action == 8 :
                        p_pass = self.q - 0.1*max(abs(B1[0]-B2[0]),abs(B1[1]-B2[1]))
                        inLine = check_inline(B1, B2, R_new)
                        if inLine:
                            p_pass /= 2
                        self.T[self.state_indices[state], action, self.state_indices['END']] += (1 - p_pass)*R_action[i]
                        s1 = state[0:4]+cells(R_new)
                        if state[6] == '1':
                            s1 += '2'
                        else :
                            s1 += '1'
                        self.T[self.state_indices[state], action, self.state_indices[s1]] += p_pass*R_action[i]
                    else :
                        if state[6] == '1':
                            p_shoot = self.q - 0.2*(3-B1[0])
                        else :
                            p_shoot = self.q - 0.2*(3-B2[0])
                        if R_new == [3,1] or R_new == [3,2]:
                            p_shoot /= 2
                        self.T[self.state_indices[state], action, self.state_indices['END']] += (1 - p_shoot)*R_action[i]
                        self.T[self.state_indices[state], action, self.state_indices['WIN']] += p_shoot*R_action[i]
                        self.R[self.state_indices[state], action, self.state_indices['WIN']] = 1
            else :
                continue
        return

    def create_mdp(self):
        self.T[8192,:,8192] = 1
        self.T[8193,:,8193] = 1
        for i in range(self.numStates-2):
            for j in range(self.numActions):
                self.possible_transitions(self.states[i], j)

    def write_mdp(self):
        print(f"numStates {self.numStates}")
        print(f"numActions {self.numActions}")
        print(f"end 8192 8193")
        T = self.T
        R = self.R
        non_zero_ts = np.nonzero(T)
        for i in zip(*non_zero_ts):
            print(f"transition {i[0]} {i[1]} {i[2]} {R[i[0],i[1],i[2]]} {T[i[0],i[1],i[2]]}")
        print("mdptype episodic")
        print("discount 1.0")

if __name__ == '__main__':
    parser.add_argument('--opponent', type = str)
    parser.add_argument("--p",type=float)
    parser.add_argument("--q",type=float)
    args = parser.parse_args()
    # args.opponent = 'data/football/test-1.txt'
    # p = 0.2
    # q = 0.8
    game = Game()
    game.p = args.p
    game.q = args.q
    game.read_R_policy(args.opponent)
    game.create_mdp()
    game.write_mdp()
