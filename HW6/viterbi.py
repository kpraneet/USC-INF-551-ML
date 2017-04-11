# Authors: Karthik Ravindra Rao - raokarth@usc.edu & Praneet Kalluri - pkalluri@usc.edu
import numpy as np
from scipy.spatial import distance
from decimal import *

#-------------------------------------calculates the distance from all the towers---------------------------------------

def distance_from_towers(x,y):
    coordinates = [(x, y), (0,0), (0,9), (9,0), (9,9)]
    distance_array =  np.array(distance.cdist(coordinates, coordinates, 'euclidean'))[0]
    distance_array = np.delete(distance_array, 0)
    return distance_array

#------function to prepare the emission probability matrix - given a state what is the probability of observation-------

def prepare_emission_probability(states_array, observations):
    emission_probability = np.zeros((len(observations), len(states_array)))
    for observation_index,observation in enumerate(observations):
        for state_index,state in enumerate(states_array):
            distance_array = distance_from_towers(state[0], state[1])
            if (observation.item(0) >= distance_array[0]*0.7 and observation.item(0) <= distance_array[0]*1.3) and (observation.item(1) >= distance_array[1]*0.7 and observation.item(1) <= distance_array[1]*1.3) and (observation.item(2) >= distance_array[2]*0.7 and observation.item(2) <= distance_array[2]*1.3) and (observation.item(3) >= distance_array[3]*0.7 and observation.item(3) <= distance_array[3]*1.3):
                emission_probability[observation_index][state_index] = float(1/Decimal(7)) #hard coded - there can possibly be only 7 valid observations for the distances to towers including the error
    return emission_probability

#---function to prepare the transition probability matrix - the probability of transition from one state to another-----

def prepare_transition_probability(states_array):
    transition_probability = np.zeros((len(states_array), len(states_array)))
    for index,state in enumerate(states_array):
        count = 0
        if (state[0]+1, state[1]) in states_array:
            count += 1
        if (state[0], state[1]+1) in states_array:
            count += 1
        if (state[0]-1, state[1]) in states_array:
            count += 1
        if (state[0], state[1]-1) in states_array:
            count += 1
        temporary_probability = float(1/Decimal(count))
        if (state[0]+1, state[1]) in states_array:
            transition_probability[index][states_array.index((state[0]+1, state[1]))] = temporary_probability
        if (state[0], state[1]+1) in states_array:
            transition_probability[index][states_array.index((state[0], state[1]+1))] = temporary_probability
        if (state[0]-1, state[1]) in states_array:
            transition_probability[index][states_array.index((state[0]-1, state[1]))] = temporary_probability
        if (state[0], state[1]-1) in states_array:
            transition_probability[index][states_array.index((state[0], state[1]-1))] = temporary_probability
    return transition_probability




#-----------------------------------Prepares data as mentioned in the assignment description----------------------------

def prepare_data():
    # TODO : check if input can be directly added or read from input file
    states = np.matrix('1 1 1 1 1 1 1 1 1 1; 1 1 1 1 1 1 1 1 1 1; 1 1 0 0 0 0 0 1 1 1; 1 1 0 1 1 1 0 1 1 1; 1 1 0 1 1 1 0 1 1 1; 1 1 0 1 1 1 0 1 1 1; 1 1 0 1 1 1 0 1 1 1; 1 1 1 1 1 1 1 1 1 1; 1 1 1 1 1 1 1 1 1 1; 1 1 1 1 1 1 1 1 1 1')
    observations = np.matrix('6.3 5.9 5.5 6.7; 5.6 7.2 4.3 6.8; 7.6 9.4 4.3 5.4; 9.5 10.0 3.7 6.6; 6.0 10.7 2.8 5.8; 9.3 10.2 2.6 5.4; 8.0 13.1 1.9 9.4; 6.4 8.2 3.9 8.8; 5.0 10.3 3.6 7.2; 3.8 9.8 4.4 8.8; 3.3 7.6 4.3 8.5')
    states_array = []
    for x in range(states.shape[0]):
        for y in range(states.shape[1]):
            if states.item(x,y) != 0:
                states_array.append((x,y))
    emission_probability = prepare_emission_probability(states_array, observations)
    transition_probability = prepare_transition_probability(states_array)
    return observations, states_array, emission_probability, transition_probability

#-------Performs viterbi algorithm to determine the best possible hidden state sequence for the given observations------

def viterbi(observations, states_array, emission_probability, transition_probability):
    V = [{}]
    viterbi_probability_matrix = np.zeros((len(observations), len(states_array)))
    viterbi_previous_state_matrix = np.zeros((len(observations), len(states_array)))
    viterbi_previous_state_matrix = viterbi_previous_state_matrix.astype(int)
    for state_count in range(len(states_array)):
        viterbi_probability_matrix[0][state_count] = float(1/Decimal(87)) * emission_probability[0][state_count]
    for observation_index in range(1, len(observations)):
        for current_state_index in range(len(states_array)):
            maximum_probability = 0
            for previous_state_index in range(len(states_array)):
                if viterbi_probability_matrix[observation_index - 1][previous_state_index] * transition_probability[current_state_index][previous_state_index] * emission_probability[observation_index - 1][previous_state_index] > maximum_probability:
                    maximum_probability = viterbi_probability_matrix[observation_index - 1][previous_state_index] * transition_probability[current_state_index][previous_state_index] * emission_probability[observation_index - 1][previous_state_index]
                    viterbi_probability_matrix[observation_index][current_state_index] =  maximum_probability
                    viterbi_previous_state_matrix[observation_index][current_state_index] = previous_state_index

    # tie is automatically taken care of because states_array is indexed in such a way that x and y is in ascending order

    temporary_maximum_probability = -1.0
    maximum_state = ()
    final_optimal_path_of_robot = []
    previous_state_index = 0
    for state_index,probability_value in enumerate(viterbi_probability_matrix[-1]):
        if probability_value > temporary_maximum_probability:
            temporary_maximum_probability = probability_value
            maximum_state = states_array[state_index]
            previous_state_index = state_index

    final_optimal_path_of_robot.append(maximum_state)

    for t in range(len(viterbi_probability_matrix) - 1, 0, -1):
        previous_state_index = viterbi_previous_state_matrix[t][previous_state_index]
        final_optimal_path_of_robot.insert(0,states_array[previous_state_index])

    print ("predicted optimal state of robot from 0th observation to 11th observation")
    print(final_optimal_path_of_robot)

#---------------------------------------------------------main function-------------------------------------------------

def main():
    observations, states_array, emission_probability, transition_probability = prepare_data()
    viterbi(observations, states_array, emission_probability, transition_probability)

#------------------------------------function to inform compiler of the main function-----------------------------------

if __name__ == '__main__':
    main()