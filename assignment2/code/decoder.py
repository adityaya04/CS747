import numpy as np
import argparse
parser = argparse.ArgumentParser()

if __name__ == '__main__':
    parser.add_argument('--opponent', type = str)
    parser.add_argument("--value-policy",type= str)
    args = parser.parse_args()
    value = np.loadtxt(args.value_policy)
    f = open(args.opponent, 'r')
    _ = f.readline()
    lines = f.readlines()
    for i in range(8192):
        print(f"{lines[i][:7]} {int(value[i,1])} {value[i,0]}")


