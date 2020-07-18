# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 23:37:32 2020

@author: hyu
"""
import numpy as np
from cvxopt import matrix, solvers

def maxmin(A):
    num_vars = len(A)
    
    # minimize matrix c
    c = [-1] + [0 for i in range(num_vars)]
    c = np.array(c, dtype="float")
    c = matrix(c)
    
    # constraints G*x <= h
    G = np.matrix(A, dtype="float").T # reformat each variable is in a row
    # G before inverse sign to get the standard form
    print("G matrix:", G)
    G *= -1 # minimization constraint
    G = np.vstack([G, np.eye(num_vars) * -1]) # > 0 constraint for all vars
    new_col = [1 for i in range(num_vars)] + [0 for i in range(num_vars)]
    G = np.insert(G, 0, new_col, axis=1) # insert utility column for simplex tableau
    
    G = matrix(G)
    h = ([0 for i in range(num_vars)] + 
         [0 for i in range(num_vars)])
    h = np.array(h, dtype="float")
    h = matrix(h)
    
    # contraints Ax = b, sum of pi_rock, pi_paper, pi_scissors equal 1
    A = [0] + [1 for i in range(num_vars)]
    A = np.matrix(A, dtype="float")
    A = matrix(A)
    b = np.matrix(1, dtype="float")
    b = matrix(b)
    print("c matrix:", c)
    print("G matrix:", G)
    print("h matrix:", h)
    print("A matrix:", A)
    print("b matrix:", b)
    
    sol = solvers.lp(c=c, G=G, h=h, A=A, b=b)
    return sol

# R = [[0,1,-1], [-1,0,1],[1,-1,0]]
R = [[0, 2, -1], [-2, 0, 1], [1, -1, 0]]
R = [[0.0, 1.66, -0.52], [-1.66, 0.0, 1.0], [0.52, -1.0, 0.0]]
R = [[0.0, 0.46, -1.0], [-0.46, 0.0, 1.0], [1.0, -1.0, 0.0]]
R = [[0.0, 3.04, -1.0], [-3.04, 0.0, 0.37], [1.0, -0.37, 0.0]]
R = [[0.0, 4.92, -0.32], [-4.92, 0.0, 1.66], [0.32, -1.66, 0.0]]
R = [[0.0, 1.0, -0.39], [-1.0, 0.0, 1.0], [0.39, -1.0, 0.0]]
R = [[0.0, 1.0, -1.12], [-1.0, 0.0, 2.71], [1.12, -2.71, 0.0]]
R = [[0.0, 1.62, -4.18], [-1.62, 0.0, 1.92], [4.18, -1.92, 0.0]]
R = [[0.0, 0.17, -4.69], [-0.17, 0.0, 1.0], [4.69, -1.0, 0.0]]
R = [[0.0, 1.0, -1.31], [-1.0, 0.0, 3.88], [1.31, -3.88, 0.0]]
R = [[0.0, 2.58, -3.29], [-2.58, 0.0, 1.0], [3.29, -1.0, 0.0]]
sol = maxmin(A=R)
probs = sol["x"]
ans = list(probs)[1:]
ans = sum([a**2 for a in ans])
print(ans)