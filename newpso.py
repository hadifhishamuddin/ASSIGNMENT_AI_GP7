# from IPython.display import Image
# from IPython.core.display import HTML 
# Image(url= "https://www.researchgate.net/profile/Malek_Sarhani/post/What_is_Velocity_in_Particle_Swarm_Optimization/attachment/5abfe2ccb53d2f63c3c3245d/AS%3A610191980630016%401522492513881/image/PSOEquation.png")

import random as rd
import numpy as np
import matplotlib.pyplot as plt
rd.seed(12)

W = 0.5 #0.5 0.6
c1 = 1.6 #0.8 1.6
c2 = 2.4 #0.9 2.4

n_iterations = 1000
n_particles = 30
target_error = 1e-6

class Particle():
    def __init__(self):
        x = (-1) ** bool(rd.getrandbits(1)) * rd.random() * 1000
        y = (-1) ** bool(rd.getrandbits(1)) * rd.random() * 1000
        self.position = np.array([x, y])
        self.pBest_position = self.position
        self.pBest_value = float('inf')
        self.velocity = np.array([0,0])

    def update(self):
        self.position = self.position + self.velocity

class Space():
    def __init__(self, target, target_error, n_particles):
        self.target = target
        self.target_error = target_error
        self.n_particles = n_particles
        self.particles = []
        self.gBest_value = float('inf')
        self.gBest_position = np.array([rd.random() * 50, rd.random() * 50])
            
    def fitness(self, particle):
        x = particle.position[0]
        y = particle.position[1]
        f =  x**2 + y**2 + 1
        return f
    
    def set_pBest(self):
        for particle in self.particles:
            fitness_candidate = self.fitness(particle)
            if(particle.pBest_value > fitness_candidate):
                particle.pBest_value = fitness_candidate
                particle.pBest_position = particle.position
                
    def set_gBest(self):
        for particle in self.particles:
            best_fitness_candidate = self.fitness(particle)
            if(self.gBest_value > best_fitness_candidate):
                self.gBest_value = best_fitness_candidate
                self.gBest_position = particle.position
                
    def update_particles(self):
        for particle in self.particles:
            global W
            inertial = W * particle.velocity
            self_confidence = c1 * rd.random() * (particle.pBest_position - particle.position)
            swarm_confidence = c2 * rd.random() * (self.gBest_position - particle.position)
            new_velocity = inertial + self_confidence + swarm_confidence
            particle.velocity = new_velocity
            particle.update()

    def show_particles(self, iteration):        
        print(iteration, 'iterations')
        print('BestPosition in this time:', self.gBest_position)
        print('BestValue in this time:', self.gBest_value)
        
        plt.ion()
        for particle in self.particles:
            plt.plot(particle.position[0], particle.position[1], 'ro')
            plt.draw()
        plt.plot(self.gBest_position[0], self.gBest_position[1], 'bo')
        plt.title(f'PSO iteration: {iteration}')
        plt.show()
        plt.draw()
        plt.pause(.01)
        plt.clf()
        #plt.ioff()
    
    def end(self):
        for particle in self.particles:
            plt.plot(particle.position[0], particle.position[1], 'ro')
        plt.plot(self.gBest_position[0], self.gBest_position[1], 'bo')
        plt.title(f'Best solution PSO')
        plt.show(block=True)   

search_space = Space(1, target_error, n_particles)
particle_vector = [Particle() for _ in range(search_space.n_particles)]
search_space.particles = particle_vector

iteration = 0
while(iteration < n_iterations):
    # set particle best & global best
    search_space.set_pBest()
    search_space.set_gBest()

    # visualization
    search_space.show_particles(iteration)
    
    # check conditional
    if(abs(search_space.gBest_value - search_space.target) <= search_space.target_error):
        break
        
    search_space.update_particles()
    iteration += 1

print("The best solution is: ", search_space.gBest_position, " in ", iteration, " iterations")
search_space.end()