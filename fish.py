'''
Vail Dorchester
Final Project
'''

import numpy as np
import random
import math
import operator

'''
# This here is my fishy
'''
class fish():

    # various variables
    radii = [] # radius of repulsion, orientation, and attraction
    noise = [0.2, 0.35] # velocity and angle noise
    weights = [1.5, 0.7] # Attraction & repulsion weights

    # physical properties position and velocity
    pos = [] # componentwise position
    r = 0.0 # spherical r
    vel = [] # componentwise velocity
    ang = [] # angles theta, phi using spherical (phi is azimuthal)
    dt = 0.25 # the timestep

    # init boundary conditions
    max_init_pos = 8 # can start randomly in a 200x200 box (unitless)

    '''
    This is the init function. In this, we just set some basic variables, init
    the position and velocity, solve for the spherical unit r, and initialize
    the angle.
    '''
    def __init__(self, r=[12,55,150], v=5):
        # set constraints on movement
        self.radii = r # init the radii
        self.noise = [0.2, 0.55] # setting the random noises
        self.weights = [1.5, 0.7] # setting the weights

        # init pos and velocity
        self.pos = [round(random.random()*self.max_init_pos) for i in range(3) ]
        self.vel = v
        self.r = self.calc_r(self.pos) # calculate r

        # now init the angle. times i cause theta is from 0 to pi and phi
        # is from 0 to 2*pi where i just conveniently works for the 2
        self.ang = [round(random.random()*(i+1)*np.pi, 2) for i in range(2)]

        # return
        return

    '''
    This function will just wrap all of the other functions that make the fish move.
    The function will simply take the school as a variable. There's also a verbose
    (ve) option. The function will basically calculate new angles and then move.
    '''
    def move(self, school, ve=False):
        t, p = self.calculate_angle(school, ve) # calculate the new angles to move in
        self.update_pos(t, p, ve) # move to new position
        return

    '''
    Update the position. This will take the new angles as arguments, and maybe other stuff, not sure yet,
    but it'll actually move the fish. I didn't want to call the calculate angle in here so that it's more
    modular and general.
    '''
    def update_pos(self, theta, phi, ve=False):
        # get vector components vx, vy, and vz. Here, I assume normalized radius 1,
        # and use spherical coordinate conversions to get the cartesian vector
        # that corresponds to the previously calculated theta and phi. This vector
        # is helpful because it has components x, y, z.
        vx = np.sin(theta)*np.cos(phi)
        vy = np.sin(theta)*np.sin(phi)
        vz = np.cos(theta)

        if(ve):print(vx,vy,vz) # verbose

        # calculate the new position in each of the principal directions.
        # the equation is initial position plus the quantity of the
        # timestep multiplied by the directional component by the velocity
        # constant by the noise.
        x = self.pos[0] + (self.dt * vx * self.vel * ((random.random()*2*self.noise[0]) - self.noise[0])) # x
        y = self.pos[1] + (self.dt * vy * self.vel * ((random.random()*2*self.noise[0]) - self.noise[0])) # y
        z = self.pos[2] + (self.dt * vz * self.vel * ((random.random()*2*self.noise[0]) - self.noise[0])) # z

        # now update the positions
        self.pos[0] = x
        self.pos[1] = y
        self.pos[2] = z

        return # end

    '''
    This function will update the velocity.
    '''
    def update_vel(self):

        return # return

    '''
    Calculating the angle. Turns out this is the bulk of the work because
    this is what determines where each fish goes. School is a list containing
    all the fishies.
    '''
    def calculate_angle(self, school, ve=False):
        # here i'll store the thetas and phis I calculate for each other fish
        # and then I'll average them. So it's deciding for each fish what to do
        # and then averaging all of those decisions.
        thetas = []
        phis = []
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
                continue # ignore it
            elif(d < self.radii[0]): # if it's in the radius of repulsion
                # vector pointing from other to self
                v = list(map(operator.sub, self.pos, other_pos))
                vr = self.calc_r(v) # get spherical r of v
                weight = self.weights[1] # assign repulsion weight

                # convert that vector into some angles (physics convention)
                theta = np.arctan(v[2]/vr) # arctan(z/x)
                if(v[0]==0): # if x = 0
                    if(v[1] > 0): # point is on positive y axis
                        phi = np.pi/2
                    else: # point is on negative y axis
                        phi = (3*np.pi)/2
                else: # if x != 0, we can do the normal definition
                    phi = np.arctan(v[1]/v[0]) # arctan(y/x)

                # average the angles
                theta_avg = (theta + self.ang[0])/2
                phi_avg = (phi + self.ang[1])/2

                if(ve):print(theta, self.ang[0], theta_avg)
            elif(d < self.radii[1]): # if it's in the radius of orientation
                # average self and other angles
                theta_avg = (other_ang[0] + self.ang[0]) / 2 # avg other theta & this one
                phi_avg = (other_ang[1] + self.ang[1]) / 2 # same but with phi
                weight = 1 # this doesn't technically have a weight
            elif(d < self.radii[2]): # if it's in radius of attraction
                # vector pointing from self to other
                v = list(map(operator.sub, other_pos, self.pos))
                vr = self.calc_r(v) # get spherical r of v
                weight = self.weights[0] # set attraction weight

                # convert that vector into some angles (physics convention)
                theta = np.arctan(v[2]/vr) # arctan(z/x)
                if(v[0]==0): # if x = 0
                    if(v[1] > 0): # point is on positive y axis
                        phi = np.pi/2
                    else: # point is on negative y axis
                        phi = (3*np.pi)/2
                else: # if x != 0, we can do the normal definition
                    phi = np.arctan(v[1]/v[0]) # arctan(y/x)

                # average the angles
                theta_avg = (theta + self.ang[0])/2
                phi_avg = (phi + self.ang[1])/2

            # End of the else statements
            # Now apply the angular noise variation
            theta_avg = theta_avg*((random.random()*2*self.noise[1]) - self.noise[1])
            phi_avg = phi_avg*((random.random()*2*self.noise[1]) - self.noise[1])

            # add them to the lists
            thetas.append(theta_avg)
            phis.append(phi_avg)

        # now average all of the decisions
        t = sum(thetas)/len(thetas)
        p = sum(phis)/len(phis)

        # and return the new angles theta and phi
        return t, p

    # calculates spherical r
    def calc_r(self, v):
        return math.sqrt(sum(map(lambda x:x*x, v)))

    # some functions to get variables
    def get_pos(self):
        return self.pos
    def get_ang(self):
        return self.ang
'''
# this function will drive the code so that I can easily just pass different
# parameters and it'll simulate and plot etc
'''
def driver():
    return
# main method
def main():
    radii = [150, 55, 12] # of attract, orientation, and repulsion
    fishes = [] # hold the fishies

    # init two fish to test with
    for i in range(2):
        fishes.append(fish(radii))
    # test stuff
    fishes[0].calculate_angle(fishes)
    return

# call main method
if __name__ == '__main__':
    main()
