import numpy as np
from math import exp
import operator
from functools import reduce

yin = np.array([1,1,1,0,1,0,1,0,0,0])
xin = np.array([0,1,0,1,1,1,0,1,1,0])
theta_init = 0.*np.array(list(range(10)))

gamma=0.01
precision = 0.0001
previous_step_size = 1
max_iters = 10000
iters = 0
cur_theta = theta_init
gx = lambda theta: exp(cur_theta.dot(xin)) / 1 + exp(cur_theta.dot(xin))

def dj_theta(theta,y=yin):
    g = np.array([(gx(a) - b) * a for a, b in zip(theta,y)])
    return g
g_init = dj_theta(cur_theta)
djt = g_init

while (previous_step_size > precision) and (iters < max_iters):
    prev_theta = cur_theta
    cur_theta += gamma*reduce(operator.mul, djt)
    djt = dj_theta(cur_theta)
    previous_step_size = reduce(operator.add, abs(cur_theta-prev_theta))
    iters +=1












