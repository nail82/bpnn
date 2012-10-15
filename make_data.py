"""
Ted Satcher
CS 640
Fall 2012

Assignment 2

File: make_data.py

This module is for creating test data for Assingment 2.
"""

import numpy as np

def circle_data(n):
    """This function creates the data for the assignment
    with omega 1 bounded inside a cirlce of radius 1 and
    omega 2 outside a circle of radius 4."""
    b = 1.0
    a = -.99
    points = []
    one = np.array(1.)
    two = np.array(-1.)
    stop = n/2
    while len(points) < stop:
        p = (b-a) * np.random.random(2) + a
        d = np.sqrt(p[0]**2 + p[1]**2)
        if d < 1.0:
            p = np.hstack((one, p))
            points.append(p)

    b = 10.0
    a = -9.99
    while len(points) < n:
        p = (b-a) * np.random.random(2) + a
        d = np.sqrt(p[0]**2 + p[1]**2)
        if d > 4.0:
            p = np.hstack((two, p))
            points.append(p)

    data = np.matrix(points[0])
    for i in range(1,len(points)):
        data = np.vstack((data, points[i]))

    return (data)

def write_data1(fnm, points):
    """Write that data to a csv."""
    fh = open(fnm, 'w')
    for p in points:
        p.tofile(fh, sep=',')
        fh.write('\n')

def read_data(fnm):
    """Read data from the csv."""
    fh = open(fnm, 'r')
    data = np.genfromtxt(fh, dtype=float, delimiter=',')
    return np.matrix(data)

def convert_data(data):
    """Changes the omega 2 tagged data elements to have a
    tag of -1."""
    for d in data:
        if d[0,0] == 2:
            d[0,0] = -1.
    return data

def test():
    data = read_data('homework_data.csv')

if __name__ == '__main__':
    test()
