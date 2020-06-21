# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 00:30:07 2020

@author: hyu
"""

import numpy as np

def getTD1(probToState1, valueEstimates, rewards):
    VS6 = 0
    VS5 = VS6 + rewards[6]
    VS4 = VS5 + rewards[5]
    VS3 = VS4 + rewards[4]
    
    VS2 = VS3 + rewards[3]
    VS1 = VS3 + rewards[2]
    
    VS0 = probToState1 * (VS1 + rewards[0]) + (1 - probToState1) * (VS2 + rewards[1])
    
    return VS0

def getE1(probToState1, valueEstimates, rewards):
    VS1 = valueEstimates[1]
    VS2 = valueEstimates[2]
    
    VS0 = probToState1 * (VS1 + rewards[0]) + (1 - probToState1) * (VS2 + rewards[1])
    
    return VS0

def getE2(probToState1, valueEstimates, rewards):
    VS3 = valueEstimates[3]
    VS1 = rewards[2] + VS3
    VS2 = rewards[3] + VS3
    
    VS0 = probToState1 * (VS1 + rewards[0]) + (1 - probToState1) * (VS2 + rewards[1])
    
    return VS0

def getE3(probToState1, valueEstimates, rewards):
    VS4 = valueEstimates[4]
    VS3 = rewards[4] + VS4
    VS1 = rewards[2] + VS3
    VS2 = rewards[3] + VS3
    
    VS0 = probToState1 * (VS1 + rewards[0]) + (1 - probToState1) * (VS2 + rewards[1])
    
    return VS0

def getE4(probToState1, valueEstimates, rewards):
    VS5 = valueEstimates[5]
    VS4 = rewards[5] + VS5
    VS3 = rewards[4] + VS4
    VS1 = rewards[2] + VS3
    VS2 = rewards[3] + VS3
    
    VS0 = probToState1 * (VS1 + rewards[0]) + (1 - probToState1) * (VS2 + rewards[1])
    
    return VS0

def getE5(probToState1, valueEstimates, rewards):
    VS6 = valueEstimates[6]
    VS5 = rewards[6] + VS6
    VS4 = rewards[5] + VS5
    VS3 = rewards[4] + VS4
    VS1 = rewards[2] + VS3
    VS2 = rewards[3] + VS3
    
    VS0 = probToState1 * (VS1 + rewards[0]) + (1 - probToState1) * (VS2 + rewards[1])
    
    return VS0


def getE6(probToState1, valueEstimates, rewards):
    VS6 = 0 + 0 # no reward no value from "S7"
    VS5 = rewards[6] + VS6
    VS4 = rewards[5] + VS5
    VS3 = rewards[4] + VS4
    VS1 = rewards[2] + VS3
    VS2 = rewards[3] + VS3
    
    VS0 = probToState1 * (VS1 + rewards[0]) + (1 - probToState1) * (VS2 + rewards[1])
    
    return VS0

def getEstimators(probToState1, valueEstimates, rewards):
    E1 = getE1(probToState1, valueEstimates, rewards)
    E2 = getE2(probToState1, valueEstimates, rewards)
    E3 = getE3(probToState1, valueEstimates, rewards)
    E4 = getE4(probToState1, valueEstimates, rewards)
    E5 = getE5(probToState1, valueEstimates, rewards)
    E6 = getE6(probToState1, valueEstimates, rewards)
    
    print((E1, E2, E3, E4, E5, E6))
    return (E1, E2, E3, E4, E5, E6)

def findLambda(probToState1, valueEstimates, rewards):
    E = getEstimators(probToState1, valueEstimates, rewards)

    coeffs = [E[5] - E[4], E[4] - E[3], E[3] - E[2], E[2] - E[1], E[1] - E[0], E[0] - E[5]]

    print(np.roots(coeffs))
    
p=0.42

V=[22.2,11.9,4.4,11.4,0,24.7,0.0]

rewards=[8.5,0,1.9,-0.8,9.9,-2.6,4.8]

lamb = findLambda(p,V,rewards)
