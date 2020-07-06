from itertools import permutations
# KWIK Learner
# http://icml2008.cs.helsinki.fi/papers/627.pdf
# Algorithm 2 -- Enumeration Algorithm
# Enumerate all the possible hypothesis space, in the problem, enumerate all possible
# permutations of instigator and peacemaker pairs

# the function takes in an episode of patrons, no.of total patrons, the hypothesis space H
# and true label whether fight occurred

# the function returns the updated H and the label it predicts (return -1 if it learns from true label)
def pred_or_learn(patrons, num_patrons, H, fight):

    # if all patrons present, no fight will occur for sure
    if sum(patrons) == num_patrons:
        return (H, 0)

    # store the opinions from each hypothesis h
    votes = []

    # itearte each h in Hypothesis space H
    for h in H:
        # from h, compute logics whether instigator/peacemaker present in the current episode
        instigator_ind, peacemaker_ind = h[0], h[1]
        instigator_presence = (patrons[instigator_ind] == 1)
        peacemaker_presence = (patrons[peacemaker_ind] == 1)

        # fight only occurs if instigator presents while peacemake does not
        if (instigator_presence and not peacemaker_presence):
            votes.append(1)
        else:
            votes.append(0)

    # check whether all remaining h have unaminous opinions on fight or no-fight
    # if Yes |L| = 1, the algo knows the proper output
    if (sum(votes) == len(votes) or sum(votes) == 0):
        return (H, votes[0])

    # if different opinion is in the vote, present the true label, remove all hypothesis that
    # give the wrong opinion
    else:
        remove_indices = [i for i, x in enumerate(votes) if x == ( 1 -fight)]
        for ind in sorted(remove_indices, reverse=True):
            del H[ind]
        return (H, -1)



at_establishment = [[0,0,0,1],[1,1,1,1],[0,1,1,1],[0,0,0,1],[0,1,1,1],[1,1,1,1],[0,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[0,0,0,1],[1,1,1,1],[1,1,1,1],[0,1,1,1],[1,1,1,1],[0,0,0,1],[0,1,1,1],[0,0,1,1],[1,1,1,1],[0,0,1,1],[0,1,1,1],[0,0,0,1],[0,0,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]
fight_occurred = [0,0,1,0,1,0,1,0,0,0,0,0,0,1,0,0,1,1,0,1,1,0,1,0,0,0,0,0,0,0,0,0]
#
# at_establishment = [[1,1], [1,0], [0,1], [1,1], [0,0], [1,0], [1,1]]
# fight_occurred = [0, 1, 0, 0, 0, 1, 0]
#p2:
at_establishment = [[1,1,1],[0,0,1],[1,1,1],[1,1,1],[1,1,1],[0,1,1],[0,0,1],[0,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[0,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]]
fight_occurred = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

#p3:
at_establishment = [[0,0,0,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[0,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[0,0,1,1],[0,1,1,1],[1,1,1,1],[1,1,1,1],[0,0,0,1],[0,0,0,1],[1,1,1,1],[0,1,1,1],[1,1,1,1],[0,1,1,1],[1,1,1,1],[0,0,0,1],[1,1,1,1],[0,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[0,0,1,1],[0,0,0,1],[0,1,1,1],[1,1,1,1],[0,0,0,1],[1,1,1,1]]
fight_occurred = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

#p4
at_establishment = [[0,1,1,1],[1,1,1,1],[1,1,1,1],[0,1,1,1],[1,1,1,1],[1,1,1,1],[0,1,1,1],[0,0,0,1],[0,0,0,1],[0,0,1,1],[1,1,1,1],[0,1,1,1],[1,1,1,1],[1,1,1,1],[0,1,1,1],[0,1,1,1],[1,1,1,1],[0,1,1,1],[0,1,1,1],[0,0,1,1],[1,1,1,1],[0,0,1,1],[0,0,0,1],[0,0,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[0,1,1,1],[1,1,1,1],[0,1,1,1]]
fight_occurred = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#p5:
at_establishment = [[0,0,1],[0,0,1],[0,1,1],[1,1,1],[0,1,1],[0,1,1],[0,0,1],[0,0,1],[0,1,1],[0,1,1],[1,1,1],[0,1,1],[0,1,1],[0,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]]
fight_occurred = [0,0,1,0,1,1,0,0,1,1,0,1,1,1,0,0,0,0]
#p6:
at_establishment = [[1,1],[0,1],[0,1],[0,1],[0,1],[1,1],[0,1],[0,1]]
fight_occurred = [0,1,1,1,1,0,1,1]
#p7
at_establishment = [[0,1,1],[0,0,1],[0,0,1],[1,1,1],[1,1,1],[1,1,1],[0,0,1],[0,0,1],[0,1,1],[1,1,1],[0,1,1],[1,1,1],[0,1,1],[1,1,1],[0,1,1],[0,1,1],[0,1,1],[1,1,1]]
fight_occurred = [1,1,1,0,0,0,1,1,1,0,1,0,1,0,1,1,1,0]
#8
at_establishment = [[0,0,1],[1,1,1],[0,1,1],[0,0,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[0,1,1],[0,0,1],[0,1,1],[1,1,1],[1,1,1],[0,1,1],[1,1,1],[1,1,1],[0,1,1],[1,1,1]]
fight_occurred = [1,0,1,1,0,0,0,0,1,1,1,0,0,1,0,0,1,0]
#9
at_establishment = [[1,1],[1,1],[1,1],[1,1],[0,1],[1,1],[1,1],[1,1]]
fight_occurred = [0,0,0,0,1,0,0,0]
#10
at_establishment = [[1,1,1,1],[0,1,1,1],[0,1,1,1],[0,1,1,1],[0,1,1,1],[0,0,0,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[0,1,1,1],[1,1,1,1],[0,1,1,1],[0,1,1,1],[1,1,1,1],[0,1,1,1],[0,1,1,1],[1,1,1,1],[0,1,1,1],[0,1,1,1],[0,0,0,1],[0,0,0,1],[1,1,1,1],[0,1,1,1],[0,1,1,1],[0,0,1,1],[0,1,1,1],[0,0,0,1],[0,0,1,1],[0,1,1,1],[0,0,1,1],[1,1,1,1]]
fight_occurred = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0]


num_patrons = len(at_establishment[0])
# create a list of patrons index
patrons = list(range(num_patrons))
# init the original hypothesis space
H = list(permutations(patrons, 2))

pred_results = []
pred_string_results = []

# iterate each episode
for episode in range(len(at_establishment)):
    print("\repisode:", episode, end=" ")
    H_prime, pred = pred_or_learn(at_establishment[episode], num_patrons, H, fight_occurred[episode])

    pred_results.append(pred)
    H = H_prime
    if (pred == 1):
        pred_string_results.append('1')
    elif (pred == -1):
        pred_string_results.append('2')
    else:
        pred_string_results.append('0')
print(''.join(pred_string_results))
# print("Final H:", H)
#
# class Kwik:
#
#     df
#
#
#     def learn(self, at_establishment, fight_occurred):
#         pass
#
