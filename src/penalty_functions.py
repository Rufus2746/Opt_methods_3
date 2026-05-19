import numpy as np
import Gauss_algh as ga

func_counter = 0

def reset_counter():
    global func_counter
    func_counter = 0

def my_func(x, y):
    global func_counter
    func_counter += 1
    return (x - y)**2 + 10 * (x + 5)**2

def restriction_g(x, y):
    return -(x + y)

def restriction_h(x, y):
    return x + y - 1

def penalty_func_H(x, y, alpha=2):
    return abs(restriction_h(x, y)) ** alpha

def penalty_func_G(x, y, alpha=2):
    restrict = restriction_g(x, y)
    if restrict > 0:
        return restrict ** alpha
    else:
        return 0

def penalty_functions_method(x0, y0, r0, penalty_func, alpha=2, beta=2, eps=1e-7):
    coords_prev = np.zeros(2)
    coords_next = np.zeros(2)
    coords_prev[0] = x0
    coords_prev[1] = y0
    r = r0
    print("ro = ", r, end="\t")
    print("x0, y0 = ", coords_prev, end="\t")
    print("alpha = ", alpha, end="\t")
    print("beta = ", beta, end="\t")
    print("eps = ", eps)
    
    def modified_func_Q(x, y):
        return my_func(x, y) + r * penalty_func(x, y, alpha)
    
    flag = True
    i = 0
    while flag:
        coords_next = ga.gauss_alghoritm(coords_prev[0], coords_prev[1], modified_func_Q)
        
        if abs(penalty_func(coords_next[0], coords_next[1], alpha)) < eps:
             flag = False
        else:
             r = beta * r
        i += 1
        coords_prev = coords_next.copy()
 
    print("iter num = ", i)
    print("function calls = ", func_counter)
    print("founded coords = ", coords_next)
    print("func value  = ", my_func(coords_next[0], coords_next[1]))
    print("\n")
    reset_counter()
    return coords_next

def test2():
    x = -2.
    y = 3.
    pf = penalty_func_G
    penalty_functions_method(x,y,2.,pf)
    penalty_functions_method(x,y,5.,pf)
    penalty_functions_method(x,y,10.,pf)
    penalty_functions_method(x,y,15.,pf)
    penalty_functions_method(x,y,20.,pf)

def test3():
    x = -2.
    y = 3.
    r = 2.
    pf = penalty_func_H
    penalty_functions_method(x,y,r,pf,2,2)
    penalty_functions_method(x,y,r,pf,2,4)
    penalty_functions_method(x,y,r,pf,2,6)
    penalty_functions_method(x,y,r,pf,2,8)
    penalty_functions_method(x,y,r,pf,2,10)

def test4():
    r = 2.
    pf = penalty_func_G
    penalty_functions_method(-4.,2.,r,pf,2)
    penalty_functions_method(2.,-5.,r,pf,2)
    penalty_functions_method(-8.,3.,r,pf,2)
    penalty_functions_method(-26.,25.,r,pf,2)
    penalty_functions_method(-400.,200.,r,pf,2)

def test5():
    x = -2.
    y = 3.
    r = 2.
    pf = penalty_func_H
    penalty_functions_method(x,y,r,pf,2.,2.,1e-03)
    penalty_functions_method(x,y,r,pf,2.,2.,1e-05)
    penalty_functions_method(x,y,r,pf,2.,2.,1e-07)
    penalty_functions_method(x,y,r,pf,2.,2.,1e-09)

def main():
    test5()

if __name__ == "__main__":
    main()