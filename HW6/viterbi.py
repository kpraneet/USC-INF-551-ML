import numpy as np
from scipy.spatial import distance
from decimal import *

#calculates the distance from all the towers
def distance_from_towers(x,y):
    coords = [(x, y), (0,0), (0,9), (9,0), (9,9)]
    distance_array =  np.array(distance.cdist(coords, coords, 'euclidean'))[0]
    distance_array = np.delete(distance_array, 0)
    return distance_array

#function to prepare the emission probability matrix - given a state what is the probability of observation
def prepare_emission_probability(states_array, observations):
    emission_prob = [{}]
    for count,observation in enumerate(observations):
        temp_list = []
        for state in states_array:
            distance_array = distance_from_towers(state[0], state[1])
            if (observation.item(0) >= distance_array[0]*0.7 and observation.item(0) <= distance_array[0]*1.3) and (observation.item(1) >= distance_array[1]*0.7 and observation.item(1) <= distance_array[1]*1.3) and (observation.item(2) >= distance_array[2]*0.7 and observation.item(2) <= distance_array[2]*1.3) and (observation.item(3) >= distance_array[3]*0.7 and observation.item(3) <= distance_array[3]*1.3):
                temp_list.append(state)
        temp_emission_probability = 1/float(len(temp_list))
        emission_prob.append({})
        for temp in temp_list:
            emission_prob[count][temp] = temp_emission_probability
    print (emission_prob)
    return emission_prob

# function to prepare the transition probability matrix - the probability of transition from one state to another
def prepare_transition_probability(states_array):
    transition_probability = {}
    count = 0
    for state in states_array:
        count = 0
        if (state[0]+1, state[1]) in states_array:
            count += 1
        if (state[0], state[1]+1) in states_array:
            count += 1
        if (state[0]-1, state[1]) in states_array:
            count += 1
        if (state[0], state[1]-1) in states_array:
            count += 1
        temp_probability = float(1/Decimal(count))
        temp_dictionary = {}
        if (state[0]+1, state[1]) in states_array:
            temp_dictionary[(state[0]+1, state[1])] = temp_probability
        if (state[0], state[1]+1) in states_array:
            temp_dictionary[(state[0], state[1]+1)] = temp_probability
        if (state[0]-1, state[1]) in states_array:
            temp_dictionary[(state[0]-1, state[1])] = temp_probability
        if (state[0], state[1]-1) in states_array:
            temp_dictionary[(state[0], state[1]-1)] = temp_probability
        transition_probability[state] = temp_dictionary
    print (transition_probability)
    return transition_probability




#Prepares data as mentioned in the assignment description
def prepare_data():
    # TODO : check if input can be directly added or read from input file
    states = np.matrix('1 1 1 1 1 1 1 1 1 1; 1 1 1 1 1 1 1 1 1 1; 1 1 0 0 0 0 0 1 1 1; 1 1 0 1 1 1 0 1 1 1; 1 1 0 1 1 1 0 1 1 1; 1 1 0 1 1 1 0 1 1 1; 1 1 0 1 1 1 0 1 1 1; 1 1 1 1 1 1 1 1 1 1; 1 1 1 1 1 1 1 1 1 1; 1 1 1 1 1 1 1 1 1 1')
    observations = np.matrix('6.3 5.9 5.5 6.7; 5.6 7.2 4.3 6.8; 7.6 9.4 4.3 5.4; 9.5 10.0 3.7 6.6; 6.0 10.7 2.8 5.8; 9.3 10.2 2.6 5.4; 8.0 13.1 1.9 9.4; 6.4 8.2 3.9 8.8; 5.0 10.3 3.6 7.2; 3.8 9.8 4.4 8.8; 3.3 7.6 4.3 8.5')
    states_array = []
    for x in range(states.shape[0]):
        for y in range(states.shape[1]):
            if states.item(x,y) != 0:
                states_array.append((x,y))
    towers = np.matrix('0 0; 0 9; 9 0; 9 9')
    print (states_array)
    emission_probability = prepare_emission_probability(states_array, observations)
    transition_probability = prepare_transition_probability(states_array)
    return observations, states_array, emission_probability, transition_probability

#Performs viterbi algorithm to determine the best possible hidden state sequence for the given observations
def viterbi(observations, states_array, emission_probability, transition_probability):
    V = [{}]
    for st in states_array:
        V[0][st] = {"probability": float(1/Decimal(87)) * emission_probability[0].get(st, 0), "previous" : None}
    for t in range(1, len(observations)):
        V.append({})
        for st in states_array:
            max_tr_prob = 0
            for prev_st in states_array:
                if V[t - 1][prev_st]["probability"] * transition_probability.get(prev_st, 0).get(st, 0) > max_tr_prob:
                    max_tr_prob = V[t - 1][prev_st]["probability"] * transition_probability.get(prev_st, 0).get(st, 0)
            for prev_st in states_array:
                if V[t - 1][prev_st]["probability"] * transition_probability.get(prev_st, 0).get(st, 0) == max_tr_prob:
                    max_prob = max_tr_prob * emission_probability[t].get(st, 0)
                    V[t][st] = {"probability": max_prob, "previous": prev_st}
                    break
    opt = []
    max_prob = max(value["probability"] for value in V[-1].values())
    print (max_prob)
    previous = None

    for st, data in V[-1].items():
        if data["probability"] == max_prob:
            opt.append(st)
            previous = st

    final_opt = []
    min_x = 10
    min_y = 10
    min_coordinate = ()
    #tie breaking0
    print (opt)
    for op in opt:
        if op[0] < min_x:
            min_x = op[0]
            min_coordinate = op
        elif op[0] == min_x and op[1] < min_y:
            min_y = op[1]
            min_coordinate = op

    final_opt.append(min_coordinate)

    for t in range(len(V) - 2, -1, -1):
        final_opt.insert(0, V[t + 1][previous]["previous"])
        previous = V[t + 1][previous]["previous"]

    print(final_opt)
    print ('The steps of states are ' + ' '.join(str(final_opt)) + ' with highest probability of %s' % max_prob)

def main():
    observations, states_array, emission_probability, transition_probability = prepare_data()
    viterbi(observations, states_array, emission_probability, transition_probability)

if __name__ == '__main__':
    main()