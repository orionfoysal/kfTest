'''
Bayesian like implementation of Kalman Filter. 
'''

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from collections import namedtuple 
from sys import argv 

fileName = argv[1]

gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: '(={:.3f}, š={:.3f})'.format(s[0], s[1])


df = pd.read_excel(fileName)

fuel = np.array(df['Fuel'])

fuel = np.flip(fuel, axis=0)

plt.plot(fuel)

def gaussian_multiply(g1,g2):
    mean = (g1.var * g2.mean + g2.var * g1.mean)/(g1.var+g2.var)
    variance = (g1.var*g2.var)/(g1.var+g2.var)
    return gaussian(mean, variance)

def update(prior, likelihood):
    posterior = gaussian_multiply(likelihood, prior)
    return posterior

def predict(pos, movement):
    return gaussian(pos.mean + movement.mean, pos.var + movement.var)



process_var = 0.1
sensor_var = 50.**2 	#SD 50 - 68% of values will be within X +/- 50 
velocity = 0
dt = 1.

process_model = gaussian(velocity*dt, process_var)

x = gaussian(fuel[0], 50**2)
updated = []
for z in fuel:
    prior = predict(x, process_model)
    likelihood = gaussian(z, sensor_var)
    x = update(prior, likelihood)
    m,s = x 
    updated.append(m)

# plt.hold()
plt.plot(updated)


plt.show()