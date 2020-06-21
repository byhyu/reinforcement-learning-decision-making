ss_# -*- coding: utf-8 -*-
"""
Created on Sun May 31 20:38:11 2020

@author: hyu
"""
states = [ i for i in range(16)]
values = [0 for _ in range(16)]
actions = ['n','e','s','w']
ds_actions = {'n':-4,'e':1,'s':4,'w':-1}
gamma = 1.0
def get_next_state(s, a):
    next_state = s
    if (s%4 == 0 and a=='w') or (s<4 and a=='n') or \
        ((s+1)%4==0 and a=='e') or (s>11 and a=='s'):
            pass
    else:
        ds = ds_actions[a]
        next_state = s + ds
    return next_state

def get_reward(s):
    return 0 if s in [0,15] else -1

def is_terminate_state(s):
    return s in [0,15]

def get_successors(s):
    successors = []
    if is_terminate_state(s):
        return successors
    for a in actions:
        next_state = get_next_state(s,a)
        successors.append(next_state)
    return successors

def update_value(s):
    successors = get_successors(s)
    new_value = 0
    num = 4
    reward = get_reward(s)
    for next_state in successors:
        new_value += 1.0/num*(reward + gamma * values[next_state])
    return new_value

def perform_one_interation():
    new_values = [0 for _ in  range(16)]
    for s in states:
        new_values[s] = update_value(s)
    global values
    values = new_values
    print_value(values)
    
def print_value(v):
    for i in range(16):
        print(f'{v[i]}')
        if (i+1)%4 == 0:
            print("")
    print()
    
def main():
    max_iterations = 160
    cur_iteration = 0
    while cur_iteration <= max_iterations:
        print(f'iterate No.{cur_iteration}')
        perform_one_interation()
        cur_iteration += 1
    print_value(values)
    
if __name__ == '__main__':
    main()
    
    

