import matplotlib.pyplot as plt
import argparse
import timeit
import random
import math
from copy import deepcopy
import csv
from shapely.affinity import translate
from shapely.geometry import Point
import matplotlib.pyplot as plt
from geopy import distance

"""
__________________________________ the helper funtion ______________________________
"""
def read_csv(filename):
    """
    Reads the content of a csv file.
    Returns a list of dictionaries, where each dictionary represents a row from the csv file.
    """
    with open(filename, mode="r", encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = [row for row in csv_reader]
        return rows


def get_coord_cities(cities_list):
    """
    Creates a dictionary where the key is the city address and the value is another dictionary
    containing the city's geographical coordinates.
    """
    cities = {}
    for city in cities_list:
        cities[city["address"]] = {"geo": [float(city["geo_lat"]), float(city["geo_lon"])]}
    return cities


def get_most_populated_cities(num, cities_list):
    """
    Returns a list of the n most populated cities from the given list.
    """
    cities = {}
    for city in cities_list:
        cities[city["address"]] = [int(city["population"]), deepcopy(city)]
    sorted_cities = sorted(cities.items(), key=lambda x: x[1][0], reverse=True)
    filtered_cities = []
    for _, city in sorted_cities[:num]:
        filtered_cities.append(city[1])
    return filtered_cities

def get_xy_gps(cities):
    """
    Adds the x and y coordinates to the cities dictionary based on their geographic coordinates.
    """
    new_cities = {}
    for city_name, city in cities.items():
        geo = city["geo"]
        xy = translate(Point(geo))
        y, x = xy.x, xy.y
        xy = {"xy": [x, y]}
        geo = {"geo": geo}
        new_cities[city_name] = {**xy, **geo}
    return new_cities

def get_distances(cities):
    """
    Adds to each city in the cities dictionary the distances to all the other cities in the same dictionary.
    """
    new_cities = {}
    for city_name, city in cities.items():
        distances = {}
        for city_name2, city2 in cities.items():
            if city_name == city_name2:
                continue
            city_distance = distance.distance(city["geo"], city2["geo"]).km
            distances[city_name2] = city_distance
        new_cities[city_name] = {**city, **{"distances": distances}}
    return new_cities

def plot_cities(fig, ax, cities):
    """
    Plots the cities in a scatter plot.
    """
    x, y, n = [], [], []
    for city_name, city in cities.items():
        xy = city["xy"]
        x.append(xy[0])
        y.append(xy[1])
        n.append(city_name)
    ax.scatter(x, y)
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))
    return ax


def clear_plot(ax):
    """
    clears a plot
    """
    ax.clear()


def plot_lines(fig, ax, cities):
    """
    Plots the lines between the cities in a line plot.
    """
    x, y, n = [], [], []

    for i in range(len(cities)):
        x.append(cities[i][0])
        y.append(cities[i][1])
    ax.plot(x, y, "-o")

    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))
    return ax


def plot_animation(fig, ax, cities, lines, title=None, pause=0.1):
    """
    plot the animation of finding the solution.
    """
    clear_plot(ax)
    ax = plot_cities(fig, ax, cities)
    ax = plot_lines(fig, ax, lines)
    if (title is not None):
        ax.set_title(title)
    plt.draw()
    plt.pause(pause)


def to_list(cities):
    cities_list = []
    for city_name in cities.keys():
        city = cities[city_name]
        new_city = {**{"name": city_name}, **city}
        cities_list.append(new_city)
    return cities_list


def to_dict(cities_list):
    cities_dict = {}
    for i in range(len(cities_list)):
        cities_dict[cities_list[i]["name"]] = cities_list[i]
    return cities_dict


def generate_path(cities_list):
    xy = []
    for i in range(len(cities_list)):
        xy.append(cities_list[i]["xy"])
    xy.append(cities_list[0]["xy"])
    return xy


def compute_cost(cities):
    """
    Calculate the distance starting from the city with index 0 to the next city in the index
    """
    cost = 0
    for i in range(1, len(cities)):
        city0 = cities[i-1]
        city1 = cities[i]
        cost += city0["distances"][city1["name"]]
    cost += cities[-1]["distances"][cities[0]["name"]]
    return cost

def generate_solution(cities, iter=3):
    """
    pick two cities in the path and exchange their positions in the path
    return the new proposed path
    """
    new_cities = deepcopy(cities)
    for i in range(iter):
        new_idx = random.sample(range(0, len(cities)), 2)
        tmp_city = deepcopy(new_cities[new_idx[0]])
        new_cities[new_idx[0]] = new_cities[new_idx[1]]
        new_cities[new_idx[1]] = tmp_city
    return new_cities

def SA( cities, initial_temp=10000, cooling_rate=0.95,
        visualize=False, visualization_rate=0.01, fig=None, ax=None):
    """
    The core implementation of SA algorithm itself 
    """
    cities_dict = cities
    cities = to_list(cities)
    time = 0
    temp = initial_temp
    costs = []
    temps = []
    new_solution_cost = 0
    current_solution = cities
    current_solution_cost = compute_cost(cities)
    new_solution = None
    while temp > 0.1:
        # Get a new solution
        new_solution = generate_solution(current_solution)
        # Calculate the cost for the new solution
        new_solution_cost = compute_cost(new_solution)
        # Calculate p
        p = safe_exp((current_solution_cost - new_solution_cost)/temp)#(math.exp()

        # if new solution is better or random less than p
        if(new_solution_cost < current_solution_cost or random.uniform(0,1) < p):
            current_solution = new_solution
            current_solution_cost = new_solution_cost
        if(visualize):
            path = generate_path(current_solution)
            title = f"Temp={temp:.3f}, Cost={current_solution_cost:.3f}\nInitial temp={initial_temp}, Cooling rate={cooling_rate}"
            plot_animation(fig, ax, cities_dict, path, pause=visualization_rate, title=title)

        temp *= cooling_rate
        costs.append(current_solution_cost)
        temps.append(temp)
        time += 1
    return costs, temps, new_solution, new_solution_cost


def safe_exp(v):
    try:
        return math.exp(v)
    except:
        return 0

"""
__________________________________ the main script ______________________________
"""
city_csv_file_path = "./city.csv"

start = timeit.default_timer()
plt.figure(figsize=(15, 8))
parser = argparse.ArgumentParser()
parser.add_argument('--cooling', type=float, default=0.5)
parser.add_argument('--temp', type=int, default=1000)
args = parser.parse_args()

visualize = True 
visualization_rate = 0.001

initial_temp = args.temp 
cooling_rate = args.cooling

csv_data = read_csv(city_csv_file_path)
cities = get_most_populated_cities(30, csv_data)
cities = get_coord_cities(cities)
cities = get_xy_gps(cities)
cities = get_distances(cities)


fig, ax = None, None
if(visualize):
    plt.ion()
    plt.show(block=True)
    fig, ax = plt.subplots()


start = timeit.default_timer()
costs, temps, new_solution, new_solution_cost = SA(cities, initial_temp, cooling_rate, visualize=visualize, 
                                                   visualization_rate=visualization_rate, fig=fig, ax=ax)
stop = timeit.default_timer()

print('Time: ', stop - start)
print(f"Final Cost: {costs[-1]}")
print(f"Number of steps: {len(costs)}")

path = generate_path(new_solution)
title = f"Final Path, Cost={costs[-1]:.3f}"
plot_animation(fig, ax, cities, path, pause=5, title=title)
clear_plot(ax)
plt.plot([i for i in range(len(costs))], [c/costs[0]for c in costs], c = "green",label="Cost")
plt.legend()
plt.plot([i for i in range(len(temps))], [t/temps[0] for t in temps], c= "purple",label="Temp")
plt.legend()
plt.title(f"Initial temp={initial_temp}, Cooling rate={cooling_rate}")
plt.draw()
plt.pause(10)

"""
Refrences:
https://gist.github.com/MNoorFawi/4dcf29d69e1708cd60405bd2f0f55700
https://realpython.com/python-csv/
https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point

"""