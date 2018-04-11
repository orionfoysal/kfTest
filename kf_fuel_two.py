
'''
Kalman method of KF implementation using 'K' parameter. 

'''
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from collections import namedtuple 
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-f','--filename',required=True,
        help = 'name of the excel file')

args = vars(ap.parse_args())


df = pd.read_excel(args['filename'])
fuel = np.array(df['Fuel'])
fuel = np.flip(fuel, axis=0)
plt.plot(fuel)


gaussian = namedtuple('Gaussian',['mean', 'var'])
gaussian.__repr__ = lambda s: '(={:.3f}, š={:.3f})'.format(s[0], s[1])

def update(prior, measurement):
    
    x, P = prior        # mean and variance of prior 
    z, R = measurement  # mean and variance of measurement 

    y = z - x           # Residual
    K = P / (P + R)     # Posterior variance 

    x = x + K*y         # Posterior
    P = (1 - K) * P     # Posterior variance 

    return gaussian(x, P)


def predict(posterior, movement):
    x, P = posterior    # mean and variance of posterior 
    dx, Q = movement    # mean and variance of movement 

    x = x + dx 
    P = P + Q 

    return gaussian(x, P)


process_var = 0.1
sensor_var = 50.**2 	#SD 50 - 68% of values will be within X +/- 50 
velocity = 0
dt = 1.

process_model = gaussian(velocity*dt, process_var)
pos = gaussian(fuel[0], 50**2)
updated = []

for z in fuel:
    prior = predict(pos, process_model)
    pos = update(prior, gaussian(z, sensor_var))
    updated.append(pos.mean)

# plt.hold()
plt.plot(updated)


plt.show()