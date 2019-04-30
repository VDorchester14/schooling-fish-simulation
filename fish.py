'''
Vail Dorchester
Final Project
'''

import numpy as np
import random
import math
import operator
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import pandas as pd
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import normalize
from matplotlib import animation, rc
from IPython.display import HTML, Image
import os

'''
# This here is my fishy
'''
class fish():

    # various variables
    radii = [] # radius of repulsion, orientation, and attraction
    noise = [0.02, 0.035] # velocity and angle noise
    weights = [1.2, 0.7, 0.9, 1.0] # Attraction & repulsion weights, orientation weights, self

    # physical properties position and velocity
    pos = [] # componentwise position
    r = 0.0 # spherical r
    vel = [] # componentwise velocity
    ang = [] # angles theta, phi using spherical (phi is azimuthal)
    dt = 0.25 # the timestep

    # init boundary conditions
    max_init_pos = 150 # can start randomly in a 200x200 box (unitless)

    '''
    # This is the init function. In this, we just set some basic variables, init
    # the position and velocity, solve for the spherical unit r, and initialize
    # the angle.
    '''
    def __init__(self, r=[12,20,150], v=5, noise = [0.02, 0.035], weights = [1.5, 0.7, 0.9, 1.0]):
        # set constraints on movement
        self.radii = r # init the radii
        self.noise = noise # setting the random noises
        self.weights = weights # setting the weights

        # init pos and velocity
        self.pos = [((random.random()*2*self.max_init_pos) - self.max_init_pos) for i in range(3) ]
        self.vel = v

        # now init the angle. times i cause theta is from 0 to pi and phi
        # is from 0 to 2*pi where i just conveniently works for the 2
        self.ang = [random.random(), random.random(), random.random()]

        # return
        return

    '''
    # This function will just wrap all of the other functions that make the fish move.
    # The function will simply take the school as a variable. There's also a verbose
    # (ve) option. The function will basically calculate new angles and then move.
    '''
    def move(self, school, ve=False):
        lx, ly, lz = self.calculate_angle(school, ve) # calculate the new angles to move in
        self.update_angles(lx, ly, lz) # update with new angle
        x, y, z = self.calculate_pos(lx, ly, lz) # calculate new position with local x y and z angles
        self.update_pos(x, y, z) # move to new position
        return

    '''
    # just updates angles
    '''
    def update_angles(self, x, y, z):
        self.ang[0] = x # update theta
        self.ang[1] = y # update phi
        self.ang[2] = z
        return
    '''
    # updates position
    '''
    def update_pos(self, x, y, z):
        self.pos[0] = x
        self.pos[1] = y
        self.pos[2] = z
        return
    '''
    # Update the position. This will take the new angles as arguments, and maybe other stuff, not sure yet,
    # but it'll actually move the fish. I didn't want to call the calculate angle in here so that it's more
    # modular and general.
    '''
    def calculate_pos(self, vx, vy, vz, ve=False):
        if(ve):print(vx,vy,vz) # verbose
        #print(vx, vy, vz)

        # calculate the new position in each of the principal directions.
        # the equation is initial position plus the quantity of the
        # timestep multiplied by the directional component by the velocity
        # constant by the noise.
        dx = (self.dt * vx * self.vel ) * ( 1 + random.random()*self.noise[0]) # dx
        dy = (self.dt * vy * self.vel ) * ( 1 + random.random()*self.noise[0]) # dy
        dz = (self.dt * vz * self.vel ) * ( 1 + random.random()*self.noise[0]) # dz
        x = self.pos[0] + dx # x
        y = self.pos[1] + dy # y
        z = self.pos[2] + dz # z

        #print(vx, vy, vz, dx, dy, dz)

        # now update the positions
        x = round(x, 4)
        y = round(y, 4)
        z = round(z, 4)

        return x, y, z

    '''
    # Calculating the angle. Turns out this is the bulk of the work because
    # this is what determines where each fish goes. School is a list containing
    # all the fishies.
    '''
    def calculate_angle(self, school, ve=False):
        xs = []
        ys = []
        zs = []
        count = 0
        # iterate over each of the possible neighbors
        for fish in school:
            other_pos = fish.get_pos() # gonna be using this so may as well store it
            other_ang = fish.get_ang() # same here

            # get distance to this fish
            d = round(math.sqrt(sum([(a - b) ** 2 for a, b in zip(other_pos, self.pos)])), 3)
            if(ve):print('Self: {0}'.format(self.pos))
            if(ve):print('Other: {0}'.format(other_pos))

            # now do things with that distance
            if(d == 0): # if it's this fish
                v_norm = self.get_local_xyz() # get own angle
                v_norm = [m * self.weights[3] for m in v_norm] # self weight
                count += self.weights[3]
            elif(d < self.radii[0]): # if it's in the radius of repulsion
                v = list(map(operator.sub, self.pos, other_pos)) # from other to self
                vs = sum([i**2 for i in v])**0.5 # getting square of quad
                v_norm = [i/vs for i in v] # normalizing
                v_norm = [m * self.weights[1] for m in v_norm] # repulsion weight
                count += self.weights[1]
            elif(d < self.radii[1]): # if it's in the radius of orientation
                # vector pointing from self to other
                v = list(map(operator.add, other_ang, self.ang)) # add both vectors
                v_norm = [i/2 for i in v] # average them
                v_norm = [m * self.weights[2] for m in v_norm] # orientation weight
                count += self.weights[2]
            elif(d < self.radii[2]): # if it's in radius of attraction
                # vector pointing from self to other
                v = list(map(operator.sub, other_pos, self.pos)) # frin sekf to other
                vs = sum([i**2 for i in v])**0.5 # square of quadrature
                v_norm = [i/vs for i in v] # normalized
                v_norm = [m * self.weights[0] for m in v_norm] # attraction weight
                count += self.weights[0]
            else:
                #print('too far')
                v_norm = list(self.get_local_xyz()) # else self weight
                v_norm = [m * self.weights[3] for m in v_norm] # self weight
                count += self.weights[3]
                #continue
            xs.append(v_norm[0]) # append weighted x
            ys.append(v_norm[1]) # append weighted y
            zs.append(v_norm[2]) # append weighted z

        # average these measurements
        x_norm = (sum(xs)/count)*(1 + self.noise[1]*2*(random.random())) # average x with some noise
        y_norm = (sum(ys)/count)*(1 + self.noise[1]*2*(random.random())) # average y with some noise
        z_norm = (sum(zs)/count)*(1 + self.noise[1]*(random.random())) # average z with some noise

        # normalize them again
        k = math.sqrt(x_norm**2 + y_norm**2 + z_norm**2)
        #print(k)
        vx = x_norm/k
        vy = y_norm/k
        vz = z_norm/k

        return x_norm, y_norm, z_norm

    '''
    # some functions to get variables
    '''
    def get_pos(self):
        return self.pos
    def get_ang(self):
        return self.ang
    def get_local_xyz(self):
        vx = self.ang[0]
        vy = self.ang[1]
        vz = self.ang[2]
        return vx, vy, vz
'''
# This is going to be the driver class. I'm naming it will basically drive the program
'''
class driver():

    # Declaring variables. If it's set to something, it's a somewhat arbitrary
    # default choice right now.
    school = []
    N = 2 # numer of fish
    positions = [] # each index cointains a 2d array of [[x values], [y values], [z values]]
    angles = [] # angles of each fish over time
    timesteps = 10000 # number of timesteps

    '''
    # The init function. This will create N fish and store them in school
    '''
    def __init__(self, school):

        # stoure user inputed number of fish. default 500
        self.school = school
        #self.positions.append([fish.get_pos() for fish in self.school]) # store positions
        #self.angles.append([fish.get_ang() for fish in self.school]) # store the angles
        self.N = len(school)

        return

    '''
    # Drives the simulation of the motion of the fish and stores the data
    '''
    def simulate(self, timesteps=10000):

        self.timesteps = timesteps

        # at each time step itearte over each fish and make them move
        for dt in range(timesteps): # iterate over each timestep
            for fish in self.school: # iterate over each fish now
                fish.move(self.school, ve=False) # move the fishy

            xs = [fish.get_pos()[0] for fish in self.school] # get the x values of each fish
            ys = [fish.get_pos()[1] for fish in self.school] # get the y values of each fish
            zs = [fish.get_pos()[2] for fish in self.school] # get the z values of each fish

            local_xs = [fish.get_local_xyz()[0] for fish in self.school]
            local_ys = [fish.get_local_xyz()[1] for fish in self.school]
            local_zs = [fish.get_local_xyz()[2] for fish in self.school]

            self.positions.append([xs, ys, zs, local_xs, local_ys, local_zs]) # store positions
            #self.angles.append([local_xs, local_ys, local_zs]) # store the angles

        return

    '''
    # Plots the simulation data
    '''
    def plot(self, filename):

        # here I initialize the figure and axes
        fig = plt.figure()
        ax = p3.Axes3D(fig)

        # this function here gets quiver data for a given timeframe index i
        def get_quiv(i, f):
            x = self.positions[i][0]
            y = self.positions[i][1]
            z = self.positions[i][2]
            u = self.positions[i][3]
            v = self.positions[i][4]
            w = self.positions[i][5]

            # scale angle
            u = [ a*f for a in u]
            v = [ b*f for b in v]
            w = [ c*f for c in w]
            return x,y,z,u,v,w

        # this does the animating of the quiver
        def update_quiver(i):
            self.quiv.remove() # clera old plot
            q = get_quiv(i, 5)

            self.quiv = ax.quiver(*q) # replot
            self.quiv.set_linewidth(2)
            self.plt_title.set_text("School at timestep {0}".format(i))
            self.ax.set_xlim(auto=True)
            self.ax.set_ylim(auto=True)
            self.ax.set_zlim(auto=True)
            return

        # this sets the limits of our quiver plot
        ax.set_xlim([-80,80])
        ax.set_ylim([-80,80])
        ax.set_zlim([-80,80])

        # this creates the initial quiver
        data = get_quiv(0, 5)
        quiv = ax.quiver(*data)

        # this creates the initial progress line
        title = plt.title("School at timestep {0}".format(0))

        # this creates the instance variables
        self.quiv = quiv # sets an instance variable that update_quiver can access
        self.plt_title = title # set title to be changed later
        self.ax = ax # set ax to be animated

        # Animation to do
        ani = FuncAnimation(fig, update_quiver, frames = self.timesteps, interval = 10, blit=False, repeat=True)
        dir = os.getcwd()
        ani.save(dir+'\\AlgorithmsInMolecularBio\\finalProject\\outputs\\'+filename, writer='imagemagick')

        # more formatting
        plt.xlabel("x")
        plt.ylabel("y")

        # show the plot
        plt.show()

        return
    '''
    # A couple functions to print values for debugging.
    '''
    def get_positions(self):
        return self.positions
    def get_angles(self):
        return self.angles

'''
# main method
'''
def main():
    radii = [35,40,160] # repulsion, orientation, attraction
    velocity = 12 # fish velocity
    noise = [0.1, 0.05] # velocity and angle noise
    weights = [2.0, 0.7, 3.5, 0.5] # attraction, repulsion, orientation, self
    N = 300 # number of fish
    frames = 1500 # frames to animate
    outfile = 'output.gif'

    # make the school
    print("Making {0} fish...".format(N))
    school = []
    for i in range(N):
        school.append(fish(r=radii, v = velocity))

    print("Simulating...")
    drive = driver(school)
    #drive.plot()
    drive.simulate(frames)

    print("Plotting...")
    drive.plot(outfile)

    return
# call main method
if __name__ == '__main__':main()
