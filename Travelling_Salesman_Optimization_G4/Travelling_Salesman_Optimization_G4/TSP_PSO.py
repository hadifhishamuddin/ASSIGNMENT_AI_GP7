## The Traveling Salesman Problem (TSP) refers to the challenge of determining the shortest yet most efficient route for a travelling salesman to take to visit a list of specific destinations / cities ##
import random                      # import random module for random data generation
import math                        # import math module to use mathematical functions esp to calculate cost
import matplotlib.pyplot as plt    # import matplotlib.pyplot module to make matplotlib works like MATLAB
import numpy                       # import numpy to perform a wide variety of mathematical operations on arrays
from sklearn import metrics        # to measure classification performance to create the confusion matrix
from sklearn.metrics import classification_report # to measure the quality of predictions from a classification algorithm

actual = numpy.random.binomial(1,.9,size = 300)              # draw samples from a binomial distribution for actual values
predicted = numpy.random.binomial(1,.9,size = 300)           # draw samples from a binomial distribution for predicted values

# Different measures include: Accuracy, Precision, Sensitivity (Recall), Specificity, and the F-score, that help us to evaluate out the classification model
Accuracy = metrics.accuracy_score(actual, predicted)         # to show the score of accuracy for both actual and prodeicted values measures how often the model is correct
Precision = metrics.precision_score(actual, predicted)       # to show the score of
Sensitivity_recall = metrics.recall_score(actual, predicted) #  to show the score of sensitivity (sometimes called Recall) measures how good the model is at predicting positive results which are positives that have been incorrectly predicted as negative
Specificity = metrics.recall_score(actual, predicted, pos_label=0) # to show the score of specificity measures how good the model is at predicting negative results which are negatives that have been incorrectly predicted as positive
F1_score = metrics.f1_score(actual, predicted)                     # to show the  F-score which is the "harmonic mean" of precision and sensitivity


confusion_matrix = metrics.confusion_matrix(actual, predicted) # use the confusion matrix function on our actual and predicted values once the metrices is imported

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True]) # to create a more interpretable visual display we need to convert the table into a confusion matrix display

cm_display.plot()  # to display the plot
plt.title('Confusion Matrix of TSP')   # to label the title of figure as 'Confusion Matrix of TSP'
plt.show()         # show the plot

from util import City, read_cities, write_cities_and_return_them, generate_cities, path_cost  # allow a Python file or a Python module (pso.py) to access the script from another Python file or module (util.py) by importing

# In this PSO consists of two classes, the first is the class of the particle, in this class the costs are updated, get the cost of the route, and update the costs of each route 
class Particle:  #class that represents a particle
# This is the initialiser, basically initialize_particle
# pbest stands for personal Best, best solution so far by that particle    
    def __init__(self, route, cost=None):                               # use the __init__() function to assign values for route and cost                    
        self.route = route                                              # create an attribute called route and assigns to it the value of the route parameter
        self.pbest = route                                              # create an attribute called pbest and assigns to it the value of the route parameter
        self.current_cost = cost if cost else self.path_cost()          # self.current_cost is equals to cost if cost else self.path_cost()
        self.pbest_cost = cost if cost else self.path_cost()            # self.pbest_cost is equals to cost if cost else self.path_cost()
        self.velocity = []                                              # list of velocity              

    def clear_velocity(self):                                           # define method of clear_velocity to clear the list velocity   
        self.velocity.clear()                                           # remove all elements of the list velocity

    def update_costs_and_pbest(self):                                   # define instance method of update_costs_and_pbest to set or get details about current_cost and pbest_cost
        self.current_cost = self.path_cost()                            # currrent_cost is equals to pbest.cost
        if self.current_cost < self.pbest_cost:                         # if current_cost is less than pbest.cost
            self.pbest = self.route                                     # the best position of the particle among its all positions visited so far is equals to the route travelled by the salesman
            self.pbest_cost = self.current_cost                         # the best and cheap solutions (pbest) is equals to the current cost travelled by the salesman 

    def path_cost(self):                                                # define instance method of cost of the path travellled by the salesman
        return path_cost(self.route)                                    # return the cost of path travelled by the salesman

# The second is the class where the PSO is performed, this class generates the routes and based on the global best which finds the best solution within a solution set based on position and velocity updates of the particles
class PSO:                                                              # class that represents PSO
# gbest stands for Global Best, best value so far by any particle in its neighborhood
    def __init__(self, iterations, population_size, gbest_probability=1.0, pbest_probability=1.0, cities=None):  # use the __init__() function to assign values for iterations, population_size, gbest_probablility, pbest_probability and cities
        self.cities = cities                                            # create an attribute called cities and assigns to it the value of the cities parameter
        self.gbest = None                                               # need to break cycles when an exception occurs
        self.gcost_iter = []                                            # list of gcost_iteration
        self.iterations = iterations                                    # create an attribute called iterations and assigns to it the value of the iterations parameter (maximum of iterations)
        self.population_size = population_size                          # create an attribute called population size and assigns to it the value of the population size parameter
        self.particles = []                                             # list of particles
        self.gbest_probability = gbest_probability                      # create an attribute called gbest probability and assigns to it the value of the gbest probability parameter
        self.pbest_probability = pbest_probability                      # create an attribute called pbest probability and assigns to it the value of the pbest probability parameter

        solutions = self.initial_population()                                    # the best solutions is referring to the initial population of the cities
        self.particles = [Particle(route=solution) for solution in solutions]    # the shortest route travelled by the salesman to the many cities is the solution of to decrease the cost travelled 

    def random_route(self):                                                      # define instance method of random_route travelled by the salesman
        return random.sample(self.cities, len(self.cities))                      # return the value of cities and length of the cities that travelled by the salesman

    def initial_population(self):                                                # define instance method of initial population 
        random_population = [self.random_route() for _ in range(self.population_size - 1)] # randomly start the route by the salesman without focusing on finding the most effective route
        greedy_population = [self.greedy_route(0)]                               # to direct the travelsalesman to travel by finding the nearest city for greedy population 
        return [*random_population, *greedy_population]                          # return the  results for salesman starts in random population and in greedy population
        # return [*random_population]

    def greedy_route(self, start_index):                                        # define the instance method of greedy_route for the start index
        unvisited = self.cities[:]                                              # unvisited represents a shallow copy of the cities array
        del unvisited[start_index]                                              # delete the unvisited city which act as the start index
        route = [self.cities[start_index]]                                      # starting point (cities) for the salesman to travel 
        while len(unvisited):                                                   # used to run a specific code until a certain condition to compare the distance of untraveled cities in order to find out the nearest city to reach the destination / cities 
            index, nearest_city = min(enumerate(unvisited), key=lambda item: item[1].distance(route[-1])) # finding the nearest point with the latest point in path
            route.append(nearest_city)                                          # append the nearest city to the end of the list for the salesman to travel between the cities
            del unvisited[index]                                                # delete the unvisited index
        return route                                                            # return the routes

    def run(self):                                                              # run all the attributes and methods with self
        self.gbest = min(self.particles, key=lambda p: p.pbest_cost)            # to take the minimum cost to start from a particular one and returning to the same destination
        print(f"initial cost is {self.gbest.pbest_cost}")                       # print out the initial cost required by the traveling salesman to complete the journey 
        plt.ion()                                                               # turn interactive mode on
        plt.draw()                                                              # to redraw the current figure
        for t in range(self.iterations):                                        # time recorded in the range of iterations
            self.gbest = min(self.particles, key=lambda p: p.pbest_cost)        # to take the minimum cost to start from a particular one and returning to the same destination
            if t % 20 == 0:                                                     # if time divided by 20, the remaider is equals to 0
                plt.figure(0)                                                   # figure 0 is created 
                plt.plot(pso.gcost_iter, 'g')                                   # plot the figure named pso.gcost_iter that connect with green colour
                plt.ylabel('Distance')                                          # plot y-axis label as Distance
                plt.xlabel('Generation')                                        # plot x-axis label as Generation
                fig = plt.figure(0)                                             # figure 0 is created 
                fig.suptitle('pso iter')                                        # figure with title named 'pso iter'
                x_list, y_list = [], []                                         # create empty list for  x_list and y_list  which is not null, it's just empty
                for city in self.gbest.pbest:                                   # iterate using for to find city in self.gbest.pbest
                    x_list.append(city.x)                                       # for x_list we append() is the list method for adding  city.x to the end of x_list
                    y_list.append(city.y)                                       # for y_list we append() is the list method for adding  city.x to the end of y_list
                x_list.append(pso.gbest.pbest[0].x)                             # x_list is the name given to the list, then .append() is the list method for adding an item to the end of pso.gbest.pbest[0].x is the specified item that we want to add
                y_list.append(pso.gbest.pbest[0].y)                             # y_list is the name given to the list, then .append() is the list method for adding an item to the end of pso.gbest.pbest[0].y is the specified item that we want to add
                fig = plt.figure(1)                                             # figure 1 is created
                fig.clear()                                                     # to clear the figures
                fig.suptitle(f'pso TSP iter {t}')                               # adding title PSO TSP to the graph with the iteration


                plt.plot(x_list, y_list, 'ro')                                  # to plot individual points as red circles ('r' specifies the color, and 'o' the shape of marker)
                plt.plot(x_list, y_list, 'g')                                   # to plot x and y using green lines that connected to each cities travel by the salesman
                plt.draw()                                                      # to redraw the current figure
                plt.pause(.001)                                                 # used to pause for 0.001s 
            self.gcost_iter.append(self.gbest.pbest_cost)                       # append the best value so far by any particle in its neighborhood with the minimum spend of cost by the salesman

            for particle in self.particles:                                     # iterate using for to find particles in particles
                particle.clear_velocity()                                       # clear the velocity of the particle
                temp_velocity = []                                              # create a empty array for temporary velocity
                gbest = self.gbest.pbest[:]                                     # the array in the gbest of each coordinates travelled by the salesman
                new_route = particle.route[:]                                   # new route is determined by the route travelled by the particles

                for i in range(len(self.cities)):                               # the length of cities list to see the updated values below the for loop
                    if new_route[i] != particle.pbest[i]:                       # if new route not equals to the pbest of the particle
                        swap = (i, particle.pbest.index(new_route[i]), self.pbest_probability) # swap in terms of the index of pbest particle which include new route and pbest probability
                        temp_velocity.append(swap)                              # swapping in the temporary velocity
                        # swapping between new_route[0] and new roue[1] / swapping between new_route[1]  and new_route[0] : line 129 & line 130
                        new_route[swap[0]], new_route[swap[1]] = \
                            new_route[swap[1]], new_route[swap[0]]

                for i in range(len(self.cities)):                               # the length of cities list to see the updated values below the for loop
                    if new_route[i] != gbest[i]:                                # if new_route[i] is not equals to particle.pbest[i]
                        swap = (i, gbest.index(new_route[i]), self.gbest_probability) # # swap in terms of the index of gbest particle which include new route and gbest probability
                        temp_velocity.append(swap)                              # applying swappping in the temporary velocity of the particle
                        gbest[swap[0]], gbest[swap[1]] = gbest[swap[1]], gbest[swap[0]] # gbest[swap[0]]=gbest[swap[1]]

                particle.velocity = temp_velocity                               # the velocity of the particle same as the temporary velocity of the particle     

                for swap in temp_velocity:                                      # swapping in the temporary velocity of the particle
                    if random.random() <= swap[2]:                              # if new_route[i] is not equals to particle.pbest[i]
                         #swapping between new_route[0] and new roue[1] / swapping between new_route[1]  and new_route[0] : line 143 & line 144
                        new_route[swap[0]], new_route[swap[1]] = \
                            new_route[swap[1]], new_route[swap[0]]

                particle.route = new_route                                     # new route of the particle is updated and equals to the route of the particle 
                particle.update_costs_and_pbest()                              # updating the which is most cheapest and effective in the travelling salesman problem

if __name__ == "__main__":                                                     # to store code that should only run when the file is executed as a script
    cities = read_cities(64)                                                   # travel total number of 64 cities by the salesman according to their respective size of the city
    pso = PSO(iterations=1200, population_size=300, pbest_probability=0.9, gbest_probability=0.02, cities=cities)  # creates a PSO instance with these values
    pso.run()                # to run the PSO algorithm
    print(f'cost: {pso.gbest.pbest_cost}\t| gbest: {pso.gbest.pbest}')         # to create an f-string by showing the global best particle

    x_list, y_list = [], []                                                    # create empty list for  x_list and y_list  which is not null, it's just empty
    for city in pso.gbest.pbest:                                               # iterate using for to find city in pso.gbest.pbest
        x_list.append(city.x)                                                  # for x_list we append() is the list method for adding  city.x to the end of x_list
        y_list.append(city.y)                                                  # for y_list we append() is the list method for adding  city.x to the end of y_list
    x_list.append(pso.gbest.pbest[0].x)                                        # x_list is the name given to the list, then .append() is the list method for adding an item to the end of pso.gbest.pbest[0].x is the specified item that we want to add
    y_list.append(pso.gbest.pbest[0].y)                                        # y_list is the name given to the list, then .append() is the list method for adding an item to the end of pso.gbest.pbest[0].y is the specified item that we want to add
    fig = plt.figure(1)                                                        # figure 1 is created
    fig.suptitle('pso TSP')                                                    # figure 1 with title named 'pso TSP'

    plt.plot(x_list, y_list, 'ro')                                             # to plot individual points as red circles ('r' specifies the color, and 'o' the shape of marker)
    plt.plot(x_list, y_list)                                                   # to plot x-axis and y-axis
    plt.show(block=True)                                                       # to show the output when there is no interactive plot

## The two algorithms (particles and PSO) are taken and tests are carried out by modifying the different parameters and have to determine which is most effective in solving the travelling salesman problem ##

#metrics:
print("\nDifferent measures of classification model:")                         # to print the specified message which is "Different measures of classification model:" to the screen 
print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"Specificity":Specificity,"F1_score":F1_score}) # to print all the calculations of accurancy, precision, sensitivity_recall, specificity and F1_score

#print the classification_report:
print("\nPerformance report of the model is :")                                # to print the specified message which is "Performance report of the model is :" to the screen 
print(classification_report(actual, predicted))                                # to print the classification report of the actual and predicted values           