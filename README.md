Ant Colony Optimization (ACO) for Traveling Salesman Problem

Author: Panagiotis Georgiadis

This Python code implements the Ant Colony Optimization algorithm to solve the Traveling Salesman Problem (TSP) for a set of locations. The TSP aims to find the shortest possible route that visits a given set of locations exactly once and returns to the starting location.

Problem Description:

The problem is described as follows:

A set of locations is defined with their coordinates.
The objective is to find the shortest possible route that visits all locations exactly once and returns to the starting location.
The Ant Colony Optimization algorithm is used to find an optimal or near-optimal solution.

Dependencies:

The code relies on several Python libraries and modules:

xml.etree.ElementTree: Used to parse KML files containing location data.
pprint: Used to pretty-print data.
math: Provides mathematical functions and constants.
mpl_toolkits.mplot3d: Offers 3D plotting capabilities.
matplotlib: Used for creating plots and visualizations.
numpy: Provides support for numerical operations.
sys: Used to adjust the print options for large numpy arrays.

Functions:

The code includes several key functions:

read_data(file_path: str) -> dict: Reads location data from a KML file, parses it, and returns a dictionary with location names as keys and their coordinates as values.

distance(coords1: list, coords2: list) -> float: Calculates the Euclidean distance between two sets of coordinates.

plot_map(points: dict): Plots the locations on a 2D graph, showing their relative positions.

make_distance_array(data: dict) -> np.ndarray: Creates a distance matrix that stores the distances between all pairs of locations.

ac(data: dict, n_ants: int, n_iterations: int, decay: float, alpha: int, beta: int) -> Tuple: Implements the Ant Colony Optimization algorithm to find the optimal or near-optimal solution for the TSP. It returns the best path and its distance.

How to Use:

To use this code for solving a TSP problem with the ACO algorithm, follow these steps:

Provide your location data in a KML file and specify the file path in the read_data function.

Adjust the ACO parameters such as the number of ants, the number of iterations, decay rate, alpha, and beta as needed.

Run the code using a Python interpreter.

The code will calculate the best path and its distance, and then display the results.

Example:
An example KML file with location data is provided in the code. You can use this file to test the ACO algorithm. This example is taking locations from bins in Agios Athanasios Kavalas as you can see. You can try with your locations

License:
This code is provided under a license that specifies the terms and conditions for its use and distribution.
