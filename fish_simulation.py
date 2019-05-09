'''
# Round two, except better this time.
'''
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
#import pandas as pd

'''
# Global Variables
'''
NUM = 100 # if no NUM in scope, global NUM is used.
AT_W = 5 # attraction weight
RE_W = 0.5 # repulsion weights
OR_W = 0.3 # orientation weight
RA_W = 0.2 # random weight
R_MAX = 1 # max random v
P_MAX = 10 # max starting position per component
V_MAX = 5 # max starting component wise velocity
DELTA = 1 # time step for movement
p = np.ones(3) #
v = np.ones(3) #

# move all the fish
def move_1():
    global p, v
    for i in range(1):#NUM):
        print("i {0}".format(i))
        p_i = np.vstack([p[i]]*NUM) # duplicate current position NUM times

        r = p - p_i # get current vector to every other fish
        mag = np.stack([np.sum(np.multiply(r, r), axis=1)]*3, axis=1) # add in quadrature for mag squared

        v1 = r * (AT_W) # attraction
        v1 = np.sum(v1, axis=0)
        print('V1: {0}'.format(v1))

        with np.errstate(invalid='ignore', divide='ignore'):v2 = (-r * (RE_W))/mag # repulsion
        v2 = np.nan_to_num(v2)
        v2 = np.sum(v2, axis=0)
        print('V2: {0}'.format(v2))

        v3 = (np.sum(v, axis=0)/NUM)*OR_W # orientation
        print('V3: {0}'.format(v3))
        v4 = 2*np.random.rand(3) - 1
        print('V4: {0}'.format(v4))

        dv = v1 + v2 + v3 + v4
        print('Old velocity: {0}'.format(v[i]))
        print('Old position: {0}'.format(p[i]))
        print('dv: {0}'.format(dv))
        v[i] = v1 + v2 + v3 + v4
        print('New velocity: {0}'.format(v[i]))

        p[i] = p[i] + v[i]*DELTA
        print('New position: {0}'.format(p[i]))
        del v1, v2, v3, v4

#
def move_2():
    global p, v
    beta = np.stack([p]*NUM, axis=0) # 3D array containing all positions stacked vertically
    alpha = np.stack([p]*NUM, axis=1) # 3D array containing all positions stacked horizontally

    # r[a][b] contains the vector from fish a to b
    #r = beta - alpha # every vector from every fish to every other fish.
    r = alpha - beta
    mag = np.stack([np.sum(np.multiply(r[:][:], r[:][:]), axis=2)]*3, axis=2)# the magnitude squared of every vector

    # now make the 4 movement vectors
    v1 = np.sum((AT_W/NUM)*r, axis=0) # attraction
    v2a = np.divide(-r*(RE_W/NUM), mag)
    v2a[mag==0] = 0
    v2 = np.sum(v2a, axis=0) # repulsion
    v3 = np.vstack([np.sum(p, axis=0)/NUM]*NUM)*OR_W # average v
    v4 = np.random.rand(NUM,3)*RA_W # random

    # get new velocity and position
    v = v1 + v2 + v3 + v4
    p = p + (v* DELTA) # new position

    return #p, v

def plot(p, v):
    fig = plt.figure() # set up the fig
    ax = p3.Axes3D(fig) # and the axes

    # get x, y, z
    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]

    ax.set_xlim(auto=True)
    ax.set_ylim(auto=True)
    ax.set_zlim(auto=True)
    ax.scatter(x, y, z)

    plt.show()

def simulate(n=1000, steps=500, method=2):
    # grab and set the global number to n
    global NUM, p, v
    NUM = n # set NUM

    p = np.random.randint(0, high=P_MAX, size=(NUM,3), dtype=int) # create the position vectors
    v = np.random.randint(0, high=V_MAX, size=(NUM,3)) # create the velocity vectors
    plot(p, v)

    # do all of the steps
    if(method==2):
        for i in range(steps):
            move_2()
    elif(method==1):
        for i in range(steps):
            move_1()
    plot(p, v)

    return

# main method
def main():
    simulate(n=100, steps=1, method=1)
    return
if __name__=="__main__":main()
