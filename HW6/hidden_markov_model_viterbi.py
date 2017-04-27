# Authors: Karthik Ravindra Rao - raokarth@usc.edu & Praneet Kalluri - pkalluri@usc.edu
import numpy as np
from scipy.spatial import distance
from decimal import *

#---------------------------------helper function to truncate values to nth decimal place-------------------------------

def truncate(f, n):
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return float('.'.join([i, (d+'0'*n)[:n]]))

#-------------------------------------calculates the distance from all the towers---------------------------------------

def distance_from_towers(x,y):
    coordinates = [(x, y), (0,0), (0,9), (9,0), (9,9)]
    distance_array =  np.array(distance.cdist(coordinates, coordinates, 'euclidean'))[0]
    distance_array = np.delete(distance_array, 0)
    return distance_array

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

#--------------------helper function to calculate the probability of observation given state ---------------------------

def calculate_emission_probability(distance_array):
    temporary_array = []
    for distance in distance_array:
        count = 1
        start_value = truncate(distance*0.7, 1)
        end_value = truncate(distance*1.3, 1)
        while(start_value < end_value):
            start_value += 0.1
            start_value = round(start_value, 1)
            count += 1
        temporary_array.append(count)
        temporary_probability = 1
        for temporary_value in temporary_array:
            temporary_probability = temporary_probability*float(1/Decimal(temporary_value))
    return temporary_probability

#------function to prepare the emission probability matrix - given a state what is the probability of observation-------

def prepare_emission_probability(states_array, observations):
    emission_probability = np.zeros((len(observations), len(states_array)))
    for state_index, state in enumerate(states_array):
        distance_array = distance_from_towers(state[0], state[1])
        for observation_index,observation in enumerate(observations):
            counter = 0
            for distance_index,distance in enumerate(distance_array):
                start_value = truncate(distance * 0.7, 1)
                end_value = truncate(distance * 1.3, 1)
                if observation.item(distance_index)>= start_value and observation.item(distance_index) <= end_value:
                    counter += 1
                if counter == 4:
                    emission_probability[observation_index][state_index] = calculate_emission_probability(distance_array)
    return emission_probability

#-----------------------------------Prepares data as mentioned in the assignment description----------------------------

def prepare_data():
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

#------------------------helper function to consider precision only until the 24th decimal place ------------------------

def correct_error(value1, value2):
     if value1!=0 and value2!=0 and abs(value2-value1) < 1e-24:
         return 0
     else:
        return value1 > value2

#-------Performs viterbi algorithm to determine the best possible hidden state sequence for the given observations------

def viterbi(observations, states_array, emission_probability, transition_probability):
    viterbi_probability_matrix = np.zeros((len(observations), len(states_array)))
    viterbi_previous_state_matrix = np.zeros((len(observations), len(states_array)))
    viterbi_previous_state_matrix = viterbi_previous_state_matrix.astype(int)
    for state_count in range(len(states_array)):
        viterbi_probability_matrix[0][state_count] = float(1/Decimal(87)) * emission_probability[0][state_count]
    for observation_index in range(1, len(observations)):
        for current_state_index in range(len(states_array)):
            maximum_probability = 0
            for previous_state_index in range(len(states_array)):
                if correct_error(viterbi_probability_matrix[observation_index - 1][previous_state_index] * transition_probability[previous_state_index][current_state_index] * emission_probability[observation_index - 1][previous_state_index],maximum_probability):
                    maximum_probability = viterbi_probability_matrix[observation_index - 1][previous_state_index] * transition_probability[previous_state_index][current_state_index] * emission_probability[observation_index - 1][previous_state_index]
                    viterbi_probability_matrix[observation_index][current_state_index] = maximum_probability
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

    print ("predicted optimal state of robot from 1st observation to 11th observation")
    print(final_optimal_path_of_robot)

#---------------------------------------------------------main function-------------------------------------------------

def main():
    observations, states_array, emission_probability, transition_probability = prepare_data()
    viterbi(observations, states_array, emission_probability, transition_probability)

#------------------------------------function to inform compiler of the main function-----------------------------------

if __name__ == '__main__':
    main()
