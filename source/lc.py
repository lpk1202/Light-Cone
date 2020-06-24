import numpy as np

import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

filename = ''
font = {#'family' : 'normal',
                #'weight' : 'bold',
                'size'  : 18}

plt.rc('font', **font)

u = np.linspace(0, np.pi/2, 10000)
v = np.linspace(0, 2 * np.pi/4, 10000)

x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones_like(v))


fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_wireframe(x, y, z)
plt.show()