import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax = plt.axes(projection='3d')

N=2000
# Data for a three-dimensional line
zline1 = np.linspace(0, 1, N)
xline1 = np.sqrt(1-zline1*zline1)
yline1 = np.ones(N)
zline2 = np.linspace(0, 1, N)
xline2 = np.zeros(N)
yline2 = -np.sqrt(1-zline2*zline2)+1
zline3 = np.zeros(N)
xline3 = np.linspace(0, 1, N)
yline3 = 1-np.sqrt(1-xline3*xline3)

xd1 = np.zeros(N)
yd1 = np.ones(N)
zd1 = np.linspace(0, 1, N)
xd2 = np.zeros(N)
yd2 = np.linspace(0, 1, N)
zd2 = np.zeros(N)
xd3 = np.linspace(0, 1, N)
yd3 = np.ones(N)
zd3 = np.zeros(N)

s = 1/np.sqrt(3)
x11 = np.zeros(N)
y11 = np.linspace(1-s,1,N)
z11 = np.zeros(N)+s
x12 = np.zeros(N)
y12 = np.zeros(N)+1-s
z12 = np.linspace(0,s,N)

x21 = np.linspace(0,s,N)
y21 = np.ones(N)
z21 = np.zeros(N)+s
x22 = np.zeros(N)+s
y22 = np.ones(N)
z22 = np.linspace(0,s,N)

x31 = np.linspace(0,s,N)
y31 = np.ones(N)-s
z31 = np.zeros(N)
x32 = np.zeros(N)+s
y32 = np.linspace(1-s,1,N)
z32 = np.zeros(N)

x41 = np.linspace(0,s,N)
y41 = np.ones(N)-s
z41 = np.zeros(N)+s
x42 = np.zeros(N)+s
y42 = np.linspace(1-s,1,N)
z42 = np.zeros(N)+s

xx = np.zeros(N)+s
yy = np.zeros(N)+1-s
zz = np.linspace(0,s,N)

ax.plot3D(x11,y11,z11,'black')
ax.plot3D(x12,y12,z12,'black')
ax.plot3D(x21,y21,z21,'black')
ax.plot3D(x22,y22,z22,'black')
ax.plot3D(x31,y31,z31,'black')
ax.plot3D(x32,y32,z32,'black')
ax.plot3D(x41,y41,z41,'black')
ax.plot3D(x42,y42,z42,'black')
ax.plot3D(xx,yy,zz,'black')
ax.plot3D(xd1,yd1,zd1,'black',ls='--')
ax.plot3D(xd2,yd2,zd2,'black',ls='--')
ax.plot3D(xd3,yd3,zd3,'black',ls='--')
ax.plot3D(xline1, yline1, zline1,'black')
ax.plot3D(xline2, yline2, zline2,'black')
ax.plot3D(xline3, yline3, zline3,'black')

# Hide grid lines
ax.grid(False)

# Hide axes ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_axis_off()
ax.text(0,-0.33, 0, r"$z=1.4$", color='blue',fontsize=23)
ax.text(1,1.04, 0, r"$z=1.4$", color='blue',fontsize=23)
ax.text(0-0.1,1, 1.06, r"$z=1.4$", color='blue',fontsize=23)
ax.text(s-0.1,1-s-0.06, s+0.02, r"$z=1.4$", color='blue',fontsize=23)
ax.text(0,1.06, 0, r"$O$", color='blue',fontsize=23)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
plt.savefig('cube.pdf')
plt.show()