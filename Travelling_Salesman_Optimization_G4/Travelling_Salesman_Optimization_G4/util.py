import math                       # import math module to use mathematical functions esp to calculate cost
import random                     # import random module for random data generation
import matplotlib.pyplot as plt   # import matplotlib.pyplot module  to make matplotlib works like MATLAB


class City:                       # define a City class which allow us to create and handle our cities.
    def __init__(self, x, y):     # init method or constructor with self parameter to refer to instance attributes with x and y attributes(coordinates)
        self.x = x                # parameter x passed at init time
        self.y = y                # parameter y passed at init time

    def distance(self, city):     # distance() function with self and city argument 
        k = math.hypot(self.x - city.x, self.y - city.y) # performs distance calculation (by using Pythagorean theorem) 
        return k                  # ends the function call and returns k value to the caller

    def __repr__(self):                 # pass an instance of the City class to the repr(), Python will call the __repr__ method automatically to return the string representation of the object x and y
        return f"({self.x}, {self.y})"  # cleaner way to output the cities as coordinates with __repr__


def read_cities(size):                                        # read_cities function with size argument
    cities = []                                               # create a empty array for cities
    with open(f'test_data/cities_{size}.data', 'r') as handle:# opening test data file for the total of cities/ size of cities involved as handle to perform reading
        lines = handle.readlines()                            # readlines used to read all test data's of the city involved
        for line in lines:                                    # iterate using for to find each line in lines
            x, y = map(float, line.split())                   # map() takes 2 arguments: float and line.split , line.split() method split the line into a list
            cities.append(City(x, y))                         # cities.append() used to add a new x and y at the end of the cities
    return cities                                             # returns the cities

def write_cities_and_return_them(size):                        # write_cities_and_return_them function with passing the size argument
    cities = generate_cities(size)                             # cities represent newly generate_cities according to their respective size of the city
    with open(f'test_data/cities_{size}.data', 'w+') as handle:# opening test data file for the total of cities/ size of cities involved as handle to open it for reading and writing
        for city in cities:                                    # iterate using for to find city in cities
            handle.write(f'{city.x} {city.y}\n')               # write x and y value of the city to the handle file by using the write() method
    return cities                                              # returns the cities


def generate_cities(size):                                                                           # generate_cities() function with size argument
    return [City(x=int(random.random() * 1000), y=int(random.random() * 1000)) for _ in range(size)] # random.random( ) returns a random float in the range [0.0, 1.0) for both x and y in the range of the city size


def path_cost(route):                                                                 # path_cost() function with route argument to calculate the cost
    return sum([city.distance(route[index - 1]) for index, city in enumerate(route)]) # sum up the cost of travel from one city to another city by the travel salesman until reach the destination


def visualize_tsp(title, cities):                              # visualize_tsp() function passed with title and cities argument
    fig = plt.figure()                                         # used plt.figure in pyplot module of matplotlib library used to create a figure object
    fig.suptitle(title)                                        # suptitle() method figure module of matplotlib library is used to add a centered title to the figure
    x_list, y_list = [], []                                    # create empty list for  x_list and y_list  which is not null, it's just empty
    for city in cities:                                        # again Iterate using for to find city in cities
        x_list.append(city.x)                                  # for x_list we append() is the list method for adding  city.x to the end of x_list
        y_list.append(city.y)                                  # for y_list we append() is the list method for adding  city.y to the end of y_list
    x_list.append(cities[0].x)                                 # x_list is the name given to the list, then .append() is the list method for adding an item to the end of x_list.cities[0].x is the specified item that we want to add
    y_list.append(cities[0].y)                                 # y_list is the name given to the list, then .append() is the list method for adding an item to the end of y_list.cities[0].y is the specified item that we want to add
                                               
    plt.plot(x_list, y_list, 'ro')                             # plot x and y using red circle markers to indicate position of the cities
    plt.plot(x_list, y_list, 'g')                              # plot x and y using green lines that connected to each cities travel by the salesman
    plt.show(block=True)                                       # to display the plot of the figure, use show() method with block=True