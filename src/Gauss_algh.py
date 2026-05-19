import numpy as np
from math import sqrt


def interval_search_for_x(y, function):
    sigma = 0.0001
    x0 = 0.0
    k = 1
    if function(x0, y) > function(x0+sigma, y):
        x1 = x0 + sigma
        h = sigma
    else:
        x1 = x0 - sigma
        h = -sigma
    flag = True
    while flag:
        h *= 2
        x1 += h
        k += 1
        if function(x0, y) > function(x1, y):
            x0 = x1
        else:
            flag = False
            
    if x0 < x1:
        return np.array([x0-h/2, x1])
    else:
        return np.array([x1, x0-h/2])

def interval_search_for_y(x, function):
    sigma = 0.0001
    y0 = 0.0
    k = 1
    if function(x, y0) > function(x, y0 + sigma):
        y1 = y0 + sigma
        h = sigma
    else:
        y1 = y0 - sigma
        h = -sigma
    flag = True
    while flag:
        h *= 2
        y1 += h
        k += 1
        if function(x, y0) > function(x, y1):
            y0 = y1
        else:
            flag = False
            
    if y0 < y1:
        return np.array([y0-h/2, y1])
    else:
        return np.array([y1, y0-h/2])

def golden_section_method_for_x(y, function, a, b, eps=1e-3):
    a1 = a
    b1 = b

    coeff_1 = (3 - sqrt(5)) / 2
    coeff_2 = (sqrt(5) - 1) / 2

    x1 = a + coeff_1 * (b - a)
    x2 = a + coeff_2 * (b - a)

    i = 1
    while b1 - a1 > eps:

        a2 = a1
        b2 = b1
        func_value_1 = function(x1, y)
        func_value_2 = function(x2, y)
        if func_value_1 < func_value_2:
            b2 = x2
            x2 = x1
            func_value_2 = func_value_1
            x1 = a2 + coeff_1 * (b2 - a2)
            func_value_1 = function(x1, y)
        else:
            a2 = x1
            x1 = x2
            func_value_1 = func_value_2
            x2 = a2 + coeff_2 * (b2 - a2)
            func_value_2 = function(x2, y)
        i += 1
        b1 = b2
        a1 = a2
        
    return (a1 + b1) / 2

def golden_section_method_for_y(x, function, a, b, eps=1e-7):
    a1 = a
    b1 = b

    coeff_1 = (3 - sqrt(5)) / 2
    coeff_2 = (sqrt(5) - 1) / 2

    y1 = a + coeff_1 * (b - a)
    y2 = a + coeff_2 * (b - a)

    i = 1
    while b1 - a1 > eps:

        a2 = a1
        b2 = b1
        func_value_1 = function(x, y1)
        func_value_2 = function(x, y2)
        if func_value_1 < func_value_2:
            b2 = y2
            y2 = y1
            func_value_2 = func_value_1
            y1 = a2 + coeff_1 * (b2 - a2)
            func_value_1 = function(x, y1)
        else:
            a2 = y1
            y1 = y2
            func_value_1 = func_value_2
            y2 = a2 + coeff_2 * (b2 - a2)
            func_value_2 = function(x, y2)
        i += 1
        b1 = b2
        a1 = a2
        
    return (a1 + b1) / 2

def find_x(y, function):
    x_interval = interval_search_for_x(y, function)
    x = golden_section_method_for_x(y, function, x_interval[0], x_interval[1])                          
    return x

def find_y(x, function):
    y_interval = interval_search_for_y(x, function)
    y = golden_section_method_for_y(x, function, y_interval[0], y_interval[1])                          
    return y

def gauss_alghoritm(x0, y0, function, eps=1e-7):
    
    coords = np.zeros(2)
    coords[0] = x0
    coords[1] = y0
    coords_tmp = np.zeros(2)

    delta_func = 1.0
    
    i = 0
    while abs(delta_func) > eps:
        coords[0] = find_x(coords[1], function)
        coords[1] = find_y(coords[0], function)
        delta_func = function(coords[0], coords[1]) - function(coords_tmp[0], coords_tmp[1])
        coords_tmp = coords.copy()
        i += 1

    return coords