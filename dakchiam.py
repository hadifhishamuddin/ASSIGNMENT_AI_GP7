# 3BENRS1 GROUP 2
# Using Genetic Algorithm to Find a Best Path (Travelling Salesman Problem) #
# Problem : A travel man wants to travel a list of city in Malaysia in a minimum distance #

# Error may occurs, please try to simulate again !

# After simulation, error maybe happens because of the distance is not below 2300
# which means the distance is not the optimize one. You can try to simulate again
# since this project is used less number of population size and less number of generation
# which do not consume more time on simulation

import matplotlib.pyplot as plt
import numpy as np
import random

# Total number of city the travel man wants to visit
n_cities = 9

# Population Size
n_population = 50

# Number of generation
n_iteration = 500

# Probability select the parents
selectivity = 0.15

################
#list of cities#
################

# Coordinate in latitude and longitude form
coordinates_list = [
    (103.7300402, 1.480024637), (102.566597, 2.033737609), (101.6999833, 3.166665872),
    (113.9845048, 4.399923929), (100.3293679, 5.413613156), (102.2464615, 2.206414407),
    (101.9400203, 2.710492166), (100.3729325, 6.113307718), (102.2299768, 6.119973978)
    ]
# Name of city list
name_list = [
    'Johor Bahru','Muar','Kuala Lumpur',
    'Miri','George Town', 'Malacca',
    'Seremban', 'Alor Setar', 'Kota Baharu'
    ]
# Create a dictionary by using name of city as key and coordinate as value
cities_dict = { x:y for x,y in zip(name_list,coordinates_list)}

layout = "{0:<20}{1:<4}{2:<5}"
# Display each city with respect to their coordinates
print("City list with coordinates (latitude, longitude)")
for key, value in cities_dict.items():
    print(layout.format(str(key), ":", str(value)))
print()

####################################
#Step 1 : Create Initial Population#
####################################
# Chromosome/individual
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    #return a particular length list of items, e.g(len()cityList) choosen from the sequence, citylist
    return route

# Population
def initialPopulation(popSize, cityList):
    population_set = []

    for i in range(0, popSize):
        population_set.append(createRoute(cityList))

    list_index = []
    for i in range(len(population_set)):
        list_index.append(i)

    popRouteIndex_dict = { x:y for x,y in zip(list_index, population_set)}

    return population_set

# Endong / Decoding
# Assign each city with a number
def getListIndex(population_set):
    list_index = []
    for i in range(len(population_set)):
        list_index.append(i)

    popRouteIndex_dict = { x:y for x,y in zip(list_index, population_set)}

    return list_index, popRouteIndex_dict

###################################
#Step 2: Compute the Fitness Value#
###################################

def compute_city_distance_coordinates(a,b):
    # calculate distance between two city
    # Multiply by 111km because of 1 degree (lat, longitude) = 111 km
    return (np.sqrt( (a[0]-b[0]) ** 2 + (a[1]-b[1]) ** 2)) * 111

def compute_city_distance_names(city_a, city_b, cities_dict):
    #passing two cities to calculate the distance
    return compute_city_distance_coordinates(cities_dict[city_a], cities_dict[city_b])

#calculate the total distance for one complete route and find the fitness for the route
def fitness_eval(city_list, cities_dict):
    total = 0
    for i in range(n_cities-1):
        a = city_list[i]
        b = city_list[i+1]
        total += compute_city_distance_names(a,b, cities_dict)
    #the shorter the distance, the higher the fitness
    inverseTotal = 1/ total
    return inverseTotal

#calculate each of the total inverse distance for the all possible routes and return in a list
def get_all_fitnes(population_set, cities_dict):
    #original the fitness for all route is 0
    fitnes_list = np.zeros(n_population)

    #Looping over all solutions computing the fitness for each solution
    for i in  range(n_population):
        fitnes_list[i] = fitness_eval(population_set[i], cities_dict)
    return fitnes_list

#######################################################
#Step 3: Selection - Method : Roulette Wheel Selection#
#######################################################

def rouletteWheelSelection(list_index, fitnes_list):
    # Calculate the total fitness based on all chromosome
    totalFit = sum(fitnes_list)
    prob_list = []

    # Calculate the probability of each chromosome to be selected
    for each in fitnes_list:
        prob_fit = each / totalFit
        prob_list.append(prob_fit)

    # Generate a list of probability list with respect to their route/path
    prob_dict = { x:y for x,y in zip(list_index,prob_list)}

    # Sort the list in ascending order of the probability to be selected with respect
    # to their route
    sorted_prob_dict = dict(sorted(prob_dict.items(), key=lambda x:x[1]))

    # Calculate the culmulative probability
    cumsum_list = np.cumsum(prob_list)

    # initialization so that all route to be selected is 0
    sumSelectEachRoute = np.zeros(n_population)

    for i in range(10000):

        # uniformly and randomly select probability value between 0-1
        temp_prob = random.uniform(0.0, 1)

        k = 0
        while(temp_prob > 0):
            temp_prob = temp_prob - prob_list[k]
            k+=1

        # Record when a route is selected
        sumSelectEachRoute[k-1]+=1

    # Take out the city list in the sorted way based on probability
    sortedCity = sorted_prob_dict.keys()

    # Record the sum of route being selected
    sumSelect_sortedRoute = { x:y for x,y in zip(sumSelectEachRoute,sortedCity) }

    # Sort the dictionay in descending order of the sum of route being selected
    sorted_sumSelect_sortedRoute = dict(reversed(sorted(sumSelect_sortedRoute.items())))

    return sorted_sumSelect_sortedRoute

#select k number of chromosome based on the probability of being selected in roulette wheel selection
def select(list_index, popRouteIndex_dict, fitnes_list, k=9):
    listSelectionParentsIndex = []
    index = 0
    sorted_sumSelect_sortedRoute = rouletteWheelSelection(list_index, fitnes_list)
    for key, bestNRoute in sorted_sumSelect_sortedRoute.items():
        if index < k:
            listSelectionParentsIndex.append(bestNRoute)
            index += 1
        else:
            break

    listSelectionParentsRoute = []
    for index in listSelectionParentsIndex:
        listSelectionParentsRoute.append(popRouteIndex_dict[index])
    return listSelectionParentsRoute

###################
#Step 4: Crossover#
###################

# multipoint crossover
def crossover(listSelectionParentsRoute, p_cross=0.3):
    children = []
    parents = np.asarray(listSelectionParentsRoute)
    count, size = parents.shape
    for _ in range(len(population_set)):
        if np.random.rand() > p_cross:
            children.append(
                list(parents[np.random.randint(count, size=1)[0]])
            )
        else:
            #if cross rate less than the default one, crossover occur
            parent1, parent2 = parents[
                np.random.randint(count, size=2), :
            ]
            idx = np.random.choice(range(size), size=2, replace=False)
            start, end = min(idx), max(idx)
            child = [None] * size
            for i in range(start, end + 1, 1):
                child[i] = parent1[i]
            pointer = 0
            for i in range(size):
                if child[i] is None:
                    while parent2[pointer] in child:
                        pointer += 1
                    child[i] = parent2[pointer]
            children.append(child)
    return children

###############################################3
#Step 5: Mutation - Method : Inversion Mutation#
################################################

#swap any two element in chromosome
def swap(chromosome):
    a, b = np.random.choice(len(chromosome), 2)
    chromosome[a], chromosome[b] = (
        chromosome[b],
        chromosome[a],
    )
    return chromosome

###############################
#Step 6: Create New Generation#
###############################

def mutate(listSelectionParentsRoute, p_cross=0.3, p_mut=0.05):
    next_pop = []
    # Perform crossover
    children = crossover(listSelectionParentsRoute, p_cross)
    for child in children:
        # If mutation happens (less than mutation probability)
        if np.random.rand() < p_mut:
            # generate a chromosome using swap mutation method
            next_pop.append(swap(child))
        else:
            # else, only crossover happen
            next_pop.append(child)
    return next_pop

#########################################################################################################
#                                         MAIN PROGRAM                                                  #
#########################################################################################################

# Create a population set
population_set = initialPopulation(n_population, name_list)

# Encoding and Decoding, assign each city with a number
list_index, popRouteIndex_dict = getListIndex(population_set)

# Find the fitness of population
fitnes_list = get_all_fitnes(population_set, cities_dict)

# Parent Selection
listSelectionParentsRoute = select(list_index, popRouteIndex_dict, fitnes_list)

# Create the next population set
new_pop_set = mutate(listSelectionParentsRoute)

# Calculate the mimumum distance from the maximum value of fitness
# because fitness is the inverse of distance
minDistance = 1 / np.max(fitnes_list)

# locate the index of the minimum distance in the fitness list
index = np.argmax(np.array(fitnes_list))

# Find the best route with respect to the minimum distance
bestRoute = popRouteIndex_dict[index]

layout1 = "{0:<5}{1:>5}{2:<5}{3:<10}{4:>5}"
print(layout1.format("Gen", "0", ":", minDistance, "km"))

###########################
#Step 7: Stopping Criteria#
###########################

history = []
value = []
i=1

# Terminate the process when number of generation reahes 500
# else, continue the process from step 2 until step 6
while (n_iteration > 0):

    prev_minDistance = minDistance
    prev_bestRoute = bestRoute

    list_index, popRouteIndex_dict = getListIndex(new_pop_set)
    fitnes_list = get_all_fitnes(new_pop_set, cities_dict)
    listSelectionParentsRoute = select(list_index, popRouteIndex_dict, fitnes_list)
    new_pop_set = mutate(listSelectionParentsRoute)

    minDistance = 1 / np.max(fitnes_list)
    index = np.argmax(np.array(fitnes_list))
    bestRoute = popRouteIndex_dict[index]

    # If distance is less than 2200, consider as 'Short' (1) class,
    # else consider as 'Long' (0) class
    if (minDistance < 2300):
        value.append(1)
    else:
        value.append(0)

    # Compare the current minumum distance with the previous generation's minimum distance
    # and record the shortest with respect to its route
    if(prev_minDistance < minDistance):
        minDistance = prev_minDistance
        bestRoute = prev_bestRoute

    # Storing the minimum distance for each generation
    history.append(minDistance)

    # Diplay the current minimum distance every 100 generaions
    if (i % 100 == 0):
        print(layout1.format("Gen", i, ":", minDistance, "km"))
    i = i + 1

    # Minus the number of generation from 500 until 0 to stop the generation
    n_iteration = n_iteration - 1

print("\nBest Route: ",bestRoute)

####################
#Performance Metric#
####################

# Import all the required modules
from numpy import array
from sklearn.model_selection import KFold
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Use the history (distance) and value (is the distance consider short or long) as dataset
# And assign them to x and y variables
x = np.array(history)
y = np.array(value)

# Split the dataset into training and test dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, train_size=0.9, random_state=1, shuffle=False, stratify=None)

# Reshape all the data
x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Create a Logistic Regression object
lr = LogisticRegression()

# Perform Logistic Regression
lr.fit(x_train, y_train)

# View the coefficient and intercept
print("\nCoefficient: ", lr.coef_)
print("Intercept: ", lr.intercept_)

# Make prediction
y_pred = lr.predict(x_test)

# Diplay the performance matrix report which showing accuracy, precision, F1-score
print("\nPerfomance Metric:")
print(classification_report(y_test, y_pred))

# Display the confusion matrix
conf_matrix = confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

###########
#Plot path#
###########

point_plot = []
for city in bestRoute:
    coordinates = cities_dict[city]
    point_plot.append(coordinates)

for xlongt, ylat in coordinates_list:
    plt.scatter(xlongt, ylat)

plt.xlim(100, 120)
plt.ylim(1,7)
for key, value in cities_dict.items():
    plt.annotate(key, xy=value, xytext=value)
plt.plot(*zip(*point_plot))
plt.grid()
plt.show()

###################
#Optimization Plot#
###################

plt.plot(range(len(history)), history, color="skyblue")
plt.show()

