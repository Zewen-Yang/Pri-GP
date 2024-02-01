
# Copyright (c) by Zewen Yang under GPL-3.0 license
# Last modified: Zewen Yang 02/2024

import numpy as np
def minmaxScaling(values, _new_min = 0, _new_max =1):
    # values = [0, 0.0001, 10000]
    # Calculate the minimum and maximum values in the list
    min_value = min(values)
    max_value = max(values)
    # Define the new range for scaling
    new_min = _new_min
    new_max = _new_max
    # Initialize an empty list to store the scaled values
    scaled_values = []
    # Iterate through the original values and scale them
    for value in values:
        scaled_value = (value - min_value) / (max_value - min_value) * (new_max - new_min) + new_min
        scaled_values.append(scaled_value)
    return scaled_values

def getProportions(values):
        # Calculate the sum of all elements in the list
        total = sum(values)
        # Calculate the proportions by dividing each element by the total
        proportions = [value / total for value in values]
        return proportions

def equalProportions(n):
    if n <= 0:
        raise ValueError("Input 'n' must be a positive integer.")
    # Calculate the proportion value
    proportion_value = 1 / n
    # Create a list with n elements, each set to the proportion_value
    proportions = [proportion_value] * n
    return proportions
