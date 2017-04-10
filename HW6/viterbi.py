import numpy as np
from scipy.spatial import distance

#Prepares data as mentioned in the assignment description
def prepare_data():
    # TODO : check if input can be directly added or read from input file
    states = np.matrix('1 1 1 1 1 1 1 1 1 1; 1 1 1 1 1 1 1 1 1 1; 1 1 0 0 0 0 0 1 1 1; 1 1 0 1 1 1 0 1 1 1; 1 1 0 1 1 1 0 1 1 1; 1 1 0 1 1 1 0 1 1 1; 1 1 0 1 1 1 0 1 1 1; 1 1 1 1 1 1 1 1 1 1; 1 1 1 1 1 1 1 1 1 1; 1 1 1 1 1 1 1 1 1 1')
    observations = np.matrix('6.3 5.9 5.5 6.7; 5.6 7.2 4.3 6.8; 7.6 9.4 4.3 5.4; 9.5 10.0 3.7 6.6; 6.0 10.7 2.8 5.8; 9.3 10.2 2.6 5.4; 8.0 13.1 1.9 9.4; 6.4 8.2 3.9 8.8; 5.0 10.3 3.6 7.2; 3.8 9.8 4.4 8.8; 3.3 7.6 4.3 8.5')
    states_array = []
    for x in range(states.shape[0]):
        for y in range(states.shape[1]):
            if states.item(x,y) != 0 and (x !=0 and y!=0) and (x !=0 and y!=9) and (x !=9 and y!=0) and (x !=9 and y!=9):
                states_array.append((x,y))
    towers = np.matrix('0 0; 0 9; 9 0; 9 9')
    return observations, states, towers, states_array

#Performs viterbi algorithm to determine the best possible hidden state sequence for the given observations.
def distance_from_towers(x,y):
    coords = [(x, y), (0,0), (0,9), (9,0), (9,9)]
    distance_array =  np.array(distance.cdist(coords, coords, 'euclidean'))[0]
    distance_array = np.delete(distance_array, 0)
    return distance_array

def random_emission_probability(x, y):
    array = []
    random_factor = 0.7
    tower_distance_array = distance_from_towers(x,y)
    while random_factor < 1.4:
        for x in tower_distance_array:
            array.append(x*random_factor)
        random_factor += 0.1
    choice = np.random.choice(array, 1)[0]
    return choice

def viterbi(observations, states, towers, states_array):
    V = [{}]
    for st in states_array:
        V[0][st] = {"probability": 1/100 * random_emission_probability(st[0],st[1]), "previous" : None}

    for t in range(1, len(observations)):
        V.append({})
        for st in states_array:
            max_tr_prob = max(V[t - 1][prev_st]["probability"] * 0.25 for prev_st in states_array)
            for prev_st in states_array:
                if V[t - 1][prev_st]["probability"] * 0.25 == max_tr_prob:
                    max_prob = max_tr_prob * random_emission_probability(prev_st[0],prev_st[1])
                    V[t][st] = {"probability": max_prob, "previous": prev_st}
                    break

    opt = []
    max_prob = max(value["probability"] for value in V[-1].values())
    previous = None

    for st, data in V[-1].items():
        if data["probability"] == max_prob:
            opt.append(st)
            previous = st

    final_opt = []
    min_x = 10
    min_coordinate = ()
    #tie breaking
    for op in opt:
        if op[0] < min_x:
            min = op[0]
            min_coordinate = op

    final_opt.append(min_coordinate)

    for t in range(len(V) - 2, -1, -1):
        final_opt.insert(0, V[t + 1][previous]["previous"])
        previous = V[t + 1][previous]["previous"]

    print(final_opt)
    # print ('The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob)

def main():
    observations, states, towers, states_array = prepare_data()
    viterbi(observations, states, towers, states_array)
    random_emission_probability(1,1)
if __name__ == '__main__':
    main()