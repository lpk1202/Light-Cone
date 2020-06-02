import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
from matplotlib import rc
import matplotlib as mpl
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)

filename = ''
font = {#'family' : 'normal',
				#'weight' : 'bold',
				'size'  : 18}

plt.rc('font', **font)

def reverse_colourmap(cmap, name = 'my_cmap_r'):
		"""
		In: 
		cmap, name 
		Out:
		my_cmap_r

		Explanation:
		t[0] goes from 0 to 1
		row i:   x  y0  y1 -> t[0] t[1] t[2]
									 /
									/
		row i+1: x  y0  y1 -> t[n] t[1] t[2]

		so the inverse should do the same:
		row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
									 /
									/
		row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
		"""        
		reverse = []
		k = []   

		for key in cmap._segmentdata:    
				k.append(key)
				channel = cmap._segmentdata[key]
				data = []

				for t in channel:                    
						data.append((1-t[0],t[2],t[1]))            
				reverse.append(sorted(data))    

		LinearL = dict(zip(k,reverse))
		my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL) 
		return my_cmap_r

cmap = mpl.cm.RdBu
cmap_r = reverse_colourmap(cmap)
o2 = np.loadtxt('../../nbodykit/fastpm-python/Example_Full/data/d3.txt')

o3 = np.loadtxt('../../nbodykit/fastpm-python/Example_Full/data/dd3.txt')
oo3 = np.zeros((7,7,7))
oo2 = np.zeros((7,7,7))
for i in np.arange(7):
	for j in np.arange(7):
		for k in np.arange(7):
			oo3[i][j][k] = o3[i*7*7+j*7+k] 
			oo2[i][j][k] = o2[i*7*7+j*7+k]

oo3 = ((oo3*0.36-0.18)+0.18)/0.36

r1 = [0,1]
r2 = [0,1]
X, Y = np.meshgrid(r1, r2)
one = np.ones(4).reshape(2, 2)

for i in np.arange(7):
	for j in np.arange(7):
		for k in np.arange(7):
			if abs(oo3[i][j][k]-0.6958116077228357)<=1e-7:
				print(i,' ',j,' ',k)


points = np.array([[1,2,3],
											[1, -1, -1 ],
											[1, 1, -1],
											[-1, 1, -1],
											[-1, -1, 1],
											[1, -1, 1 ],
											[1, 1, 1],
											[-1, 1, 1]])

fig = plt.figure(figsize=(20,8))
ax = fig.add_subplot(121, projection='3d')

ooo = np.ones(4).reshape(2, 2)*oo2[0][6][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-7,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-7,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[0][6][5]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+5,Y+7*np.sqrt(3)-7,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,Y+7*np.sqrt(3)-7,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one-one+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+5,X+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+5,X+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[0][6][4]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+4,Y+7*np.sqrt(3)-7,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,Y+7*np.sqrt(3)-7,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one-one+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+4,X+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+4,X+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[0][6][3]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+3,Y+7*np.sqrt(3)-7,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,Y+7*np.sqrt(3)-7,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one-one+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+3,X+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+3,X+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[0][6][2]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+2,Y+7*np.sqrt(3)-7,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,Y+7*np.sqrt(3)-7,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one-one+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+2,X+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+2,X+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[0][6][1]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+1,Y+7*np.sqrt(3)-7,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,Y+7*np.sqrt(3)-7,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one-one+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+1,X+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+1,X+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[0][6][0]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X,Y+7*np.sqrt(3)-7,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,Y+7*np.sqrt(3)-7,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one-one+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one,X+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one,X+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

###################################################################################################
ooo = np.ones(4).reshape(2, 2)*oo2[1][6][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-6,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-6,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[1][6][5]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+5,Y+7*np.sqrt(3)-6,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,Y+7*np.sqrt(3)-6,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one-one+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+5,X+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+5,X+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[1][6][4]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+4,Y+7*np.sqrt(3)-6,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,Y+7*np.sqrt(3)-6,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one-one+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+4,X+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+4,X+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[1][6][3]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+3,Y+7*np.sqrt(3)-6,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,Y+7*np.sqrt(3)-6,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one-one+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+3,X+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+3,X+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[1][6][2]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+2,Y+7*np.sqrt(3)-6,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,Y+7*np.sqrt(3)-6,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one-one+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+2,X+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+2,X+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[1][6][1]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+1,Y+7*np.sqrt(3)-6,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,Y+7*np.sqrt(3)-6,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one-one+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+1,X+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+1,X+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[1][6][0]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X,Y+7*np.sqrt(3)-6,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,Y+7*np.sqrt(3)-6,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one-one+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one,X+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one,X+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

############################################

ooo = np.ones(4).reshape(2, 2)*oo2[2][6][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-5,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-5,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[2][6][5]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+5,Y+7*np.sqrt(3)-5,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,Y+7*np.sqrt(3)-5,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one-one+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+5,X+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+5,X+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[2][6][4]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+4,Y+7*np.sqrt(3)-5,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,Y+7*np.sqrt(3)-5,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one-one+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+4,X+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+4,X+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[2][6][3]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+3,Y+7*np.sqrt(3)-5,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,Y+7*np.sqrt(3)-5,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one-one+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+3,X+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+3,X+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[2][6][2]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+2,Y+7*np.sqrt(3)-5,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,Y+7*np.sqrt(3)-5,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one-one+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+2,X+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+2,X+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[2][6][1]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+1,Y+7*np.sqrt(3)-5,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,Y+7*np.sqrt(3)-5,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one-one+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+1,X+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+1,X+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[2][6][0]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X,Y+7*np.sqrt(3)-5,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,Y+7*np.sqrt(3)-5,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one-one+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one,X+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one,X+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

############################################

# for i in np.arange(7):
# 	print(oo2[3][i][6])
ooo = np.ones(4).reshape(2, 2)*oo2[3][6][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-4,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-4,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[3][6][5]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+5,Y+7*np.sqrt(3)-4,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,Y+7*np.sqrt(3)-4,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one-one+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+5,X+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+5,X+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[3][6][4]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+4,Y+7*np.sqrt(3)-4,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,Y+7*np.sqrt(3)-4,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one-one+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+4,X+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+4,X+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[3][6][3]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+3,Y+7*np.sqrt(3)-4,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,Y+7*np.sqrt(3)-4,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one-one+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+3,X+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+3,X+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[3][6][2]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+2,Y+7*np.sqrt(3)-4,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,Y+7*np.sqrt(3)-4,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one-one+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+2,X+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+2,X+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[3][6][1]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+1,Y+7*np.sqrt(3)-4,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,Y+7*np.sqrt(3)-4,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one-one+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+1,X+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+1,X+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[3][6][0]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X,Y+7*np.sqrt(3)-4,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,Y+7*np.sqrt(3)-4,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one-one+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one,X+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one,X+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

############################################

ooo = np.ones(4).reshape(2, 2)*oo2[2][6][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-3,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-3,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[2][6][5]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+5,Y+7*np.sqrt(3)-3,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,Y+7*np.sqrt(3)-3,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one-one+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+5,X+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+5,X+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[2][6][4]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+4,Y+7*np.sqrt(3)-3,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,Y+7*np.sqrt(3)-3,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one-one+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+4,X+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+4,X+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[2][6][3]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+3,Y+7*np.sqrt(3)-3,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,Y+7*np.sqrt(3)-3,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one-one+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+3,X+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+3,X+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[2][6][2]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+2,Y+7*np.sqrt(3)-3,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,Y+7*np.sqrt(3)-3,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one-one+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+2,X+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+2,X+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[2][6][1]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+1,Y+7*np.sqrt(3)-3,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,Y+7*np.sqrt(3)-3,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one-one+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+1,X+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+1,X+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[2][6][0]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X,Y+7*np.sqrt(3)-3,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,Y+7*np.sqrt(3)-3,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one-one+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one,X+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one,X+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

############################################

ooo = np.ones(4).reshape(2, 2)*oo2[1][6][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-2,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-2,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[1][6][5]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+5,Y+7*np.sqrt(3)-2,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,Y+7*np.sqrt(3)-2,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one-one+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+5,X+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+5,X+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[1][6][4]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+4,Y+7*np.sqrt(3)-2,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,Y+7*np.sqrt(3)-2,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one-one+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+4,X+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+4,X+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[1][6][3]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+3,Y+7*np.sqrt(3)-2,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,Y+7*np.sqrt(3)-2,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one-one+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+3,X+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+3,X+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[1][6][2]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+2,Y+7*np.sqrt(3)-2,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,Y+7*np.sqrt(3)-2,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one-one+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+2,X+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+2,X+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[1][6][1]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+1,Y+7*np.sqrt(3)-2,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,Y+7*np.sqrt(3)-2,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one-one+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+1,X+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+1,X+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[1][6][0]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X,Y+7*np.sqrt(3)-2,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,Y+7*np.sqrt(3)-2,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one-one+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one,X+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one,X+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

############################################

ooo = np.ones(4).reshape(2, 2)*oo2[0][6][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-1,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-1,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[0][6][5]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+5,Y+7*np.sqrt(3)-1,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,Y+7*np.sqrt(3)-1,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one-one+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+5,X+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+5,X+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[0][6][4]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+4,Y+7*np.sqrt(3)-1,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,Y+7*np.sqrt(3)-1,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one-one+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+4,X+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+4,X+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[0][6][3]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+3,Y+7*np.sqrt(3)-1,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,Y+7*np.sqrt(3)-1,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one-one+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+3,X+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+3,X+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[0][6][2]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+2,Y+7*np.sqrt(3)-1,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,Y+7*np.sqrt(3)-1,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one-one+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+2,X+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+2,X+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[0][6][1]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+1,Y+7*np.sqrt(3)-1,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,Y+7*np.sqrt(3)-1,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one-one+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+1,X+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+1,X+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[0][6][0]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X,Y+7*np.sqrt(3)-1,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,Y+7*np.sqrt(3)-1,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one-one+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one,X+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one,X+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

############################################

ooo = np.ones(4).reshape(2, 2)*oo2[6][5][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-7,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-7,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][5][5]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+5,Y+7*np.sqrt(3)-7,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,Y+7*np.sqrt(3)-7,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one-one+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+5,X+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+5,X+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][5][4]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+4,Y+7*np.sqrt(3)-7,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,Y+7*np.sqrt(3)-7,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one-one+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+4,X+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+4,X+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][5][3]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+3,Y+7*np.sqrt(3)-7,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,Y+7*np.sqrt(3)-7,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one-one+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+3,X+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+3,X+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][5][2]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+2,Y+7*np.sqrt(3)-7,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,Y+7*np.sqrt(3)-7,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one-one+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+2,X+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+2,X+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][5][1]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+1,Y+7*np.sqrt(3)-7,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,Y+7*np.sqrt(3)-7,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one-one+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+1,X+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+1,X+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][5][0]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X,Y+7*np.sqrt(3)-7,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,Y+7*np.sqrt(3)-7,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one-one+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one,X+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one,X+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

###################################################################################################

ooo = np.ones(4).reshape(2, 2)*oo2[6][4][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-7,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-7,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][4][5]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+5,Y+7*np.sqrt(3)-7,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,Y+7*np.sqrt(3)-7,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one-one+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+5,X+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+5,X+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][5][4]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+4,Y+7*np.sqrt(3)-7,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,Y+7*np.sqrt(3)-7,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one-one+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+4,X+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+4,X+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][5][4]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+3,Y+7*np.sqrt(3)-7,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,Y+7*np.sqrt(3)-7,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one-one+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+3,X+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+3,X+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][4][2]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+2,Y+7*np.sqrt(3)-7,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,Y+7*np.sqrt(3)-7,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one-one+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+2,X+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+2,X+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][4][1]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+1,Y+7*np.sqrt(3)-7,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,Y+7*np.sqrt(3)-7,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one-one+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+1,X+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+1,X+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][4][0]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X,Y+7*np.sqrt(3)-7,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,Y+7*np.sqrt(3)-7,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one-one+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one,X+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one,X+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

###################################################################################################

ooo = np.ones(4).reshape(2, 2)*oo2[6][3][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-7,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-7,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][3][5]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+5,Y+7*np.sqrt(3)-7,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,Y+7*np.sqrt(3)-7,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one-one+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+5,X+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+5,X+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][3][4]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+4,Y+7*np.sqrt(3)-7,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,Y+7*np.sqrt(3)-7,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one-one+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+4,X+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+4,X+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][3][3]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+3,Y+7*np.sqrt(3)-7,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,Y+7*np.sqrt(3)-7,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one-one+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+3,X+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+3,X+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][3][2]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+2,Y+7*np.sqrt(3)-7,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,Y+7*np.sqrt(3)-7,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one-one+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+2,X+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+2,X+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][3][1]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+1,Y+7*np.sqrt(3)-7,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,Y+7*np.sqrt(3)-7,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one-one+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+1,X+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+1,X+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][3][0]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X,Y+7*np.sqrt(3)-7,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,Y+7*np.sqrt(3)-7,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one-one+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one,X+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one,X+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

###################################################################################################s

ooo = np.ones(4).reshape(2, 2)*oo2[6][2][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-7,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-7,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][2][5]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+5,Y+7*np.sqrt(3)-7,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,Y+7*np.sqrt(3)-7,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one-one+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+5,X+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+5,X+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][2][4]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+4,Y+7*np.sqrt(3)-7,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,Y+7*np.sqrt(3)-7,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one-one+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+4,X+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+4,X+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][2][3]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+3,Y+7*np.sqrt(3)-7,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,Y+7*np.sqrt(3)-7,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one-one+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+3,X+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+3,X+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][2][2]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+2,Y+7*np.sqrt(3)-7,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,Y+7*np.sqrt(3)-7,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one-one+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+2,X+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+2,X+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][2][1]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+1,Y+7*np.sqrt(3)-7,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,Y+7*np.sqrt(3)-7,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one-one+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+1,X+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+1,X+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][2][0]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X,Y+7*np.sqrt(3)-7,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,Y+7*np.sqrt(3)-7,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one-one+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one,X+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one,X+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

###################################################################################################

ooo = np.ones(4).reshape(2, 2)*oo2[6][1][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-7,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-7,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][1][5]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+5,Y+7*np.sqrt(3)-7,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,Y+7*np.sqrt(3)-7,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one-one+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+5,X+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+5,X+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][1][4]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+4,Y+7*np.sqrt(3)-7,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,Y+7*np.sqrt(3)-7,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one-one+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+4,X+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+4,X+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][1][3]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+3,Y+7*np.sqrt(3)-7,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,Y+7*np.sqrt(3)-7,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one-one+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+3,X+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+3,X+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][1][2]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+2,Y+7*np.sqrt(3)-7,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,Y+7*np.sqrt(3)-7,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one-one+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+2,X+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+2,X+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][1][1]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+1,Y+7*np.sqrt(3)-7,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,Y+7*np.sqrt(3)-7,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one-one+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+1,X+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+1,X+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][1][0]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X,Y+7*np.sqrt(3)-7,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,Y+7*np.sqrt(3)-7,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one-one+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one,X+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one,X+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

###################################################################################################

ooo = np.ones(4).reshape(2, 2)*oo2[6][0][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-7,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-7,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][0][5]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+5,Y+7*np.sqrt(3)-7,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,Y+7*np.sqrt(3)-7,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one-one+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+5,one+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+5,X+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+5,X+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][0][4]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+4,Y+7*np.sqrt(3)-7,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,Y+7*np.sqrt(3)-7,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one-one+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+4,one+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+4,X+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+4,X+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][0][3]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+3,Y+7*np.sqrt(3)-7,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,Y+7*np.sqrt(3)-7,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one-one+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+3,one+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+3,X+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+3,X+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][0][2]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+2,Y+7*np.sqrt(3)-7,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,Y+7*np.sqrt(3)-7,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one-one+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+2,one+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+2,X+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+2,X+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][0][1]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+1,Y+7*np.sqrt(3)-7,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,Y+7*np.sqrt(3)-7,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one-one+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+1,one+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+1,X+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+1,X+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[6][0][0]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X,Y+7*np.sqrt(3)-7,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,Y+7*np.sqrt(3)-7,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one-one+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X,one+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one,X+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one,X+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

###################################################################################################

ooo = np.ones(4).reshape(2, 2)*oo2[5][6][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-6,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-6,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[5][5][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-6,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-6,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-6,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-6,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-6,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-6,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[5][4][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-6,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-6,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-6,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-6,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-6,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-6,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[5][3][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-6,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-6,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-6,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-6,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-6,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-6,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[5][2][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-6,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-6,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-6,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-6,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-6,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-6,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[5][1][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-6,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-6,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-6,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-6,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-6,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-6,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[5][0][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-6,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-6,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-6,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-6,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-6,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-6,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

# ###################################################################################################

ooo = np.ones(4).reshape(2, 2)*oo2[4][6][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-5,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-5,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[4][5][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-5,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-5,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-5,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-5,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-5,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-5,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[4][4][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-5,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-5,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-5,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-5,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-5,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-5,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[4][3][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-5,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-5,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-5,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-5,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-5,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-5,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[4][2][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-5,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-5,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-5,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-5,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-5,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-5,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[4][1][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-5,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-5,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-5,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-5,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-5,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-5,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[4][0][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-5,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-5,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-5,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-5,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-5,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-5,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

# ###################################################################################################

ooo = np.ones(4).reshape(2, 2)*oo2[3][6][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-4,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-4,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[3][5][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-4,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-4,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-4,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-4,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-4,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-4,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[3][4][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-4,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-4,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-4,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-4,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-4,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-4,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[3][3][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-4,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-4,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-4,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-4,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-4,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-4,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[3][2][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-4,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-4,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-4,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-4,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-4,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-4,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[3][1][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-4,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-4,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-4,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-4,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-4,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-4,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[3][0][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-4,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-4,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-4,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-4,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-4,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-4,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

# ###################################################################################################

ooo = np.ones(4).reshape(2, 2)*oo2[2][6][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-3,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-3,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[2][5][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-3,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-3,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-3,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-3,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-3,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-3,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[2][4][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-3,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-3,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-3,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-3,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-3,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-3,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[2][3][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-3,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-3,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-3,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-3,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-3,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-3,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[2][2][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-3,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-3,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-3,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-3,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-3,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-3,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[2][1][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-3,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-3,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-3,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-3,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-3,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-3,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[2][0][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-3,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-3,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-3,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-3,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-3,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-3,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

# ###################################################################################################

ooo = np.ones(4).reshape(2, 2)*oo2[1][6][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-2,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-2,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[1][5][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-2,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-2,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-2,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-2,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-2,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-2,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[1][4][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-2,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-2,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-2,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-2,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-2,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-2,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[1][3][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-2,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-2,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-2,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-2,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-2,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-2,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[1][2][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-2,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-2,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-2,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-2,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-2,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-2,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[1][1][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-2,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-2,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-2,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-2,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-2,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-2,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[1][0][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-2,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-2,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-2,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-2,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-2,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-2,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

# ###################################################################################################

ooo = np.ones(4).reshape(2, 2)*oo2[0][6][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-1,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-1,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[0][5][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-1,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-1,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-1,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-1,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-1,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-1,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[0][4][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-1,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-1,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-1,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-1,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-1,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-1,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[0][3][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-1,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-1,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-1,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-1,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-1,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-1,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[0][2][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-1,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-1,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-1,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-1,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-1,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-1,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[0][1][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-1,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-1,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-1,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-1,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-1,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-1,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo2[0][0][6]
my_col = cmap_r(ooo)
surf = ax.plot_surface(X+6,Y+7*np.sqrt(3)-1,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,Y+7*np.sqrt(3)-1,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one-one+7*np.sqrt(3)-1,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(X+6,one+7*np.sqrt(3)-1,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one+6,X+7*np.sqrt(3)-1,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax.plot_surface(one-one+6,X+7*np.sqrt(3)-1,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

# ###################################################################################################

N=2000
# Data for a three-dimensional line
zline1 = (np.linspace(0, 1, N))
xline1 = (np.sqrt(1-zline1*zline1))*7*np.sqrt(3)
yline1 = (np.ones(N))*7*np.sqrt(3)
zline2 = (np.linspace(0, 1, N))
xline2 = (np.zeros(N))*7*np.sqrt(3)
yline2 = (-np.sqrt(1-zline2*zline2)+1)*7*np.sqrt(3)
zline3 = (np.zeros(N))*7*np.sqrt(3)
xline3 = (np.linspace(0, 1, N))
yline3 = (1-np.sqrt(1-xline3*xline3))*7*np.sqrt(3)
zline1 = (np.linspace(0, 1, N))*7*np.sqrt(3)
xline3 = (np.linspace(0, 1, N))*7*np.sqrt(3)
zline2 = (np.linspace(0, 1, N))*7*np.sqrt(3)

xd1 = (np.zeros(N))*7*np.sqrt(3)
yd1 = (np.ones(N))*7*np.sqrt(3)
zd1 = (np.linspace(0, 1, N))*7*np.sqrt(3)
xd2 = (np.zeros(N))*7*np.sqrt(3)
yd2 = (np.linspace(0, 1, N))*7*np.sqrt(3)
zd2 = (np.zeros(N))*7*np.sqrt(3)
xd3 = (np.linspace(0, 1, N))*7*np.sqrt(3)
yd3 = (np.ones(N))*7*np.sqrt(3)
zd3 = (np.zeros(N))*7*np.sqrt(3)

s = 1/np.sqrt(3)
x11 = (np.zeros(N))*7*np.sqrt(3)
y11 = (np.linspace(1-s,1,N))*7*np.sqrt(3)
z11 = (np.zeros(N)+s)*7*np.sqrt(3)
x12 = (np.zeros(N))*7*np.sqrt(3)
y12 = (np.zeros(N)+1-s)*7*np.sqrt(3)
z12 = (np.linspace(0,s,N))*7*np.sqrt(3)

x21 = (np.linspace(0,s,N))*7*np.sqrt(3)
y21 = (np.ones(N))*7*np.sqrt(3)
z21 = (np.zeros(N)+s)*7*np.sqrt(3)
x22 = (np.zeros(N)+s)*7*np.sqrt(3)
y22 = (np.ones(N))*7*np.sqrt(3)
z22 = (np.linspace(0,s,N))*7*np.sqrt(3)

x31 = (np.linspace(0,s,N))*7*np.sqrt(3)
y31 = (np.ones(N)-s)*7*np.sqrt(3)
z31 = (np.zeros(N))*7*np.sqrt(3)
x32 = (np.zeros(N)+s)*7*np.sqrt(3)
y32 = (np.linspace(1-s,1,N))*7*np.sqrt(3)
z32 = (np.zeros(N))*7*np.sqrt(3)

x41 = (np.linspace(0,s,N))*7*np.sqrt(3)
y41 = (np.ones(N)-s)*7*np.sqrt(3)
z41 = (np.zeros(N)+s)*7*np.sqrt(3)
x42 = (np.zeros(N)+s)*7*np.sqrt(3)
y42 = (np.linspace(1-s,1,N))*7*np.sqrt(3)
z42 = (np.zeros(N)+s)*7*np.sqrt(3)

xx = (np.zeros(N)+s)*7*np.sqrt(3)
yy = (np.zeros(N)+1-s)*7*np.sqrt(3)
zz = (np.linspace(0,s,N))*7*np.sqrt(3)

def get_cube():   
    phi = np.arange(1,10,2)*np.pi/4
    Phi, Theta = np.meshgrid(phi, phi)

    x = np.cos(Phi)*np.sin(Theta)
    y = np.sin(Phi)*np.sin(Theta)
    z = np.cos(Theta)/np.sqrt(2)
    return x,y,z

a = 1
b = 1
c = 1
x,y,z = get_cube()

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

# fig.colorbar(surf, label=r'x')
ax.grid(False)

# Hide axes ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_axis_off()
# ax.text(0,-0.33*7*np.sqrt(3), 0, r"$z=1.4$", color='blue',fontsize=23)
# ax.text(1*7*np.sqrt(3),1.04*7*np.sqrt(3), 0, r"$z=1.4$", color='blue',fontsize=23)
# ax.text(0-0.7,7, 1.06, r"$z=1.4$", color='blue',fontsize=23)
ax.text(5,-0.33*7*np.sqrt(3)-1, 0, r"$\hat{d}_{\rm m}^{rm LC}(\vec{r})$",fontsize=23)
# ax.text((s-0.1)*7*np.sqrt(3),(1-s-0.06)*7*np.sqrt(3), (s+0.02)*14, r"$z=1.4$", color='blue',fontsize=23)
# ax.text(0,1.06*7*np.sqrt(3), 0, r"$O$", color='blue',fontsize=23)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')


ax1 = fig.add_subplot(122, projection='3d')
ooo = np.ones(4).reshape(2, 2)*oo3[0][6][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-7,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-7,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[0][6][5]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+5,Y+7*np.sqrt(3)-7,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,Y+7*np.sqrt(3)-7,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one-one+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+5,X+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+5,X+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[0][6][4]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+4,Y+7*np.sqrt(3)-7,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,Y+7*np.sqrt(3)-7,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one-one+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+4,X+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+4,X+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[0][6][3]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+3,Y+7*np.sqrt(3)-7,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,Y+7*np.sqrt(3)-7,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one-one+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+3,X+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+3,X+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[0][6][2]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+2,Y+7*np.sqrt(3)-7,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,Y+7*np.sqrt(3)-7,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one-one+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+2,X+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+2,X+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[0][6][1]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+1,Y+7*np.sqrt(3)-7,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,Y+7*np.sqrt(3)-7,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one-one+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+1,X+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+1,X+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[0][6][0]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X,Y+7*np.sqrt(3)-7,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,Y+7*np.sqrt(3)-7,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one-one+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one,X+7*np.sqrt(3)-7,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one,X+7*np.sqrt(3)-7,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

###################################################################################################
ooo = np.ones(4).reshape(2, 2)*oo3[1][6][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-6,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-6,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[1][6][5]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+5,Y+7*np.sqrt(3)-6,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,Y+7*np.sqrt(3)-6,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one-one+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+5,X+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+5,X+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[1][6][4]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+4,Y+7*np.sqrt(3)-6,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,Y+7*np.sqrt(3)-6,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one-one+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+4,X+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+4,X+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[1][6][3]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+3,Y+7*np.sqrt(3)-6,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,Y+7*np.sqrt(3)-6,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one-one+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+3,X+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+3,X+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[1][6][2]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+2,Y+7*np.sqrt(3)-6,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,Y+7*np.sqrt(3)-6,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one-one+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+2,X+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+2,X+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[1][6][1]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+1,Y+7*np.sqrt(3)-6,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,Y+7*np.sqrt(3)-6,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one-one+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+1,X+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+1,X+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[1][6][0]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X,Y+7*np.sqrt(3)-6,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,Y+7*np.sqrt(3)-6,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one-one+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one,X+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one,X+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

############################################

ooo = np.ones(4).reshape(2, 2)*oo3[2][6][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-5,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-5,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[2][6][5]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+5,Y+7*np.sqrt(3)-5,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,Y+7*np.sqrt(3)-5,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one-one+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+5,X+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+5,X+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[2][6][4]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+4,Y+7*np.sqrt(3)-5,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,Y+7*np.sqrt(3)-5,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one-one+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+4,X+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+4,X+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[2][6][3]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+3,Y+7*np.sqrt(3)-5,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,Y+7*np.sqrt(3)-5,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one-one+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+3,X+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+3,X+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[2][6][2]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+2,Y+7*np.sqrt(3)-5,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,Y+7*np.sqrt(3)-5,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one-one+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+2,X+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+2,X+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[2][6][1]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+1,Y+7*np.sqrt(3)-5,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,Y+7*np.sqrt(3)-5,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one-one+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+1,X+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+1,X+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[2][6][0]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X,Y+7*np.sqrt(3)-5,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,Y+7*np.sqrt(3)-5,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one-one+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one,X+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one,X+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

############################################

# for i in np.arange(7):
# 	print(oo3[3][i][6])
ooo = np.ones(4).reshape(2, 2)*oo3[3][6][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-4,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-4,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[3][6][5]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+5,Y+7*np.sqrt(3)-4,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,Y+7*np.sqrt(3)-4,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one-one+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+5,X+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+5,X+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[3][6][4]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+4,Y+7*np.sqrt(3)-4,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,Y+7*np.sqrt(3)-4,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one-one+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+4,X+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+4,X+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[3][6][3]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+3,Y+7*np.sqrt(3)-4,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,Y+7*np.sqrt(3)-4,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one-one+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+3,X+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+3,X+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[3][6][2]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+2,Y+7*np.sqrt(3)-4,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,Y+7*np.sqrt(3)-4,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one-one+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+2,X+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+2,X+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[3][6][1]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+1,Y+7*np.sqrt(3)-4,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,Y+7*np.sqrt(3)-4,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one-one+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+1,X+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+1,X+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[3][6][0]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X,Y+7*np.sqrt(3)-4,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,Y+7*np.sqrt(3)-4,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one-one+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one,X+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one,X+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

############################################

ooo = np.ones(4).reshape(2, 2)*oo3[2][6][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-3,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-3,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[2][6][5]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+5,Y+7*np.sqrt(3)-3,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,Y+7*np.sqrt(3)-3,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one-one+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+5,X+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+5,X+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[2][6][4]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+4,Y+7*np.sqrt(3)-3,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,Y+7*np.sqrt(3)-3,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one-one+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+4,X+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+4,X+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[2][6][3]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+3,Y+7*np.sqrt(3)-3,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,Y+7*np.sqrt(3)-3,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one-one+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+3,X+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+3,X+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[2][6][2]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+2,Y+7*np.sqrt(3)-3,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,Y+7*np.sqrt(3)-3,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one-one+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+2,X+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+2,X+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[2][6][1]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+1,Y+7*np.sqrt(3)-3,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,Y+7*np.sqrt(3)-3,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one-one+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+1,X+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+1,X+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[2][6][0]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X,Y+7*np.sqrt(3)-3,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,Y+7*np.sqrt(3)-3,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one-one+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one,X+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one,X+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

############################################

ooo = np.ones(4).reshape(2, 2)*oo3[1][6][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-2,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-2,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[1][6][5]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+5,Y+7*np.sqrt(3)-2,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,Y+7*np.sqrt(3)-2,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one-one+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+5,X+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+5,X+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[1][6][4]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+4,Y+7*np.sqrt(3)-2,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,Y+7*np.sqrt(3)-2,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one-one+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+4,X+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+4,X+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[1][6][3]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+3,Y+7*np.sqrt(3)-2,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,Y+7*np.sqrt(3)-2,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one-one+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+3,X+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+3,X+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[1][6][2]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+2,Y+7*np.sqrt(3)-2,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,Y+7*np.sqrt(3)-2,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one-one+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+2,X+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+2,X+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[1][6][1]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+1,Y+7*np.sqrt(3)-2,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,Y+7*np.sqrt(3)-2,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one-one+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+1,X+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+1,X+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[1][6][0]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X,Y+7*np.sqrt(3)-2,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,Y+7*np.sqrt(3)-2,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one-one+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one,X+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one,X+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

############################################

ooo = np.ones(4).reshape(2, 2)*oo3[0][6][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-1,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-1,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[0][6][5]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+5,Y+7*np.sqrt(3)-1,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,Y+7*np.sqrt(3)-1,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one-one+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+5,X+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+5,X+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[0][6][4]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+4,Y+7*np.sqrt(3)-1,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,Y+7*np.sqrt(3)-1,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one-one+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+4,X+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+4,X+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[0][6][3]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+3,Y+7*np.sqrt(3)-1,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,Y+7*np.sqrt(3)-1,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one-one+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+3,X+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+3,X+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[0][6][2]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+2,Y+7*np.sqrt(3)-1,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,Y+7*np.sqrt(3)-1,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one-one+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+2,X+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+2,X+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[0][6][1]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+1,Y+7*np.sqrt(3)-1,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,Y+7*np.sqrt(3)-1,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one-one+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+1,X+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+1,X+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[0][6][0]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X,Y+7*np.sqrt(3)-1,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,Y+7*np.sqrt(3)-1,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one-one+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one,X+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one,X+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

############################################

ooo = np.ones(4).reshape(2, 2)*oo3[6][5][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-7,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-7,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][5][5]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+5,Y+7*np.sqrt(3)-7,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,Y+7*np.sqrt(3)-7,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one-one+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+5,X+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+5,X+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][5][4]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+4,Y+7*np.sqrt(3)-7,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,Y+7*np.sqrt(3)-7,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one-one+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+4,X+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+4,X+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][5][3]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+3,Y+7*np.sqrt(3)-7,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,Y+7*np.sqrt(3)-7,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one-one+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+3,X+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+3,X+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][5][2]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+2,Y+7*np.sqrt(3)-7,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,Y+7*np.sqrt(3)-7,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one-one+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+2,X+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+2,X+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][5][1]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+1,Y+7*np.sqrt(3)-7,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,Y+7*np.sqrt(3)-7,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one-one+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+1,X+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+1,X+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][5][0]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X,Y+7*np.sqrt(3)-7,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,Y+7*np.sqrt(3)-7,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one-one+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one,X+7*np.sqrt(3)-7,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one,X+7*np.sqrt(3)-7,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

###################################################################################################

ooo = np.ones(4).reshape(2, 2)*oo3[6][4][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-7,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-7,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][4][5]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+5,Y+7*np.sqrt(3)-7,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,Y+7*np.sqrt(3)-7,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one-one+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+5,X+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+5,X+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][5][4]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+4,Y+7*np.sqrt(3)-7,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,Y+7*np.sqrt(3)-7,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one-one+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+4,X+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+4,X+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][5][4]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+3,Y+7*np.sqrt(3)-7,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,Y+7*np.sqrt(3)-7,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one-one+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+3,X+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+3,X+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][4][2]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+2,Y+7*np.sqrt(3)-7,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,Y+7*np.sqrt(3)-7,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one-one+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+2,X+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+2,X+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][4][1]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+1,Y+7*np.sqrt(3)-7,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,Y+7*np.sqrt(3)-7,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one-one+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+1,X+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+1,X+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][4][0]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X,Y+7*np.sqrt(3)-7,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,Y+7*np.sqrt(3)-7,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one-one+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one,X+7*np.sqrt(3)-7,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one,X+7*np.sqrt(3)-7,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

###################################################################################################

ooo = np.ones(4).reshape(2, 2)*oo3[6][3][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-7,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-7,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][3][5]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+5,Y+7*np.sqrt(3)-7,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,Y+7*np.sqrt(3)-7,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one-one+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+5,X+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+5,X+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][3][4]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+4,Y+7*np.sqrt(3)-7,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,Y+7*np.sqrt(3)-7,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one-one+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+4,X+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+4,X+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][3][3]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+3,Y+7*np.sqrt(3)-7,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,Y+7*np.sqrt(3)-7,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one-one+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+3,X+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+3,X+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][3][2]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+2,Y+7*np.sqrt(3)-7,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,Y+7*np.sqrt(3)-7,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one-one+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+2,X+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+2,X+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][3][1]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+1,Y+7*np.sqrt(3)-7,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,Y+7*np.sqrt(3)-7,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one-one+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+1,X+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+1,X+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][3][0]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X,Y+7*np.sqrt(3)-7,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,Y+7*np.sqrt(3)-7,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one-one+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one,X+7*np.sqrt(3)-7,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one,X+7*np.sqrt(3)-7,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

###################################################################################################s

ooo = np.ones(4).reshape(2, 2)*oo3[6][2][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-7,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-7,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][2][5]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+5,Y+7*np.sqrt(3)-7,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,Y+7*np.sqrt(3)-7,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one-one+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+5,X+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+5,X+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][2][4]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+4,Y+7*np.sqrt(3)-7,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,Y+7*np.sqrt(3)-7,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one-one+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+4,X+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+4,X+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][2][3]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+3,Y+7*np.sqrt(3)-7,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,Y+7*np.sqrt(3)-7,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one-one+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+3,X+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+3,X+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][2][2]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+2,Y+7*np.sqrt(3)-7,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,Y+7*np.sqrt(3)-7,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one-one+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+2,X+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+2,X+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][2][1]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+1,Y+7*np.sqrt(3)-7,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,Y+7*np.sqrt(3)-7,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one-one+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+1,X+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+1,X+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][2][0]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X,Y+7*np.sqrt(3)-7,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,Y+7*np.sqrt(3)-7,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one-one+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one,X+7*np.sqrt(3)-7,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one,X+7*np.sqrt(3)-7,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

###################################################################################################

ooo = np.ones(4).reshape(2, 2)*oo3[6][1][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-7,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-7,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][1][5]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+5,Y+7*np.sqrt(3)-7,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,Y+7*np.sqrt(3)-7,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one-one+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+5,X+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+5,X+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][1][4]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+4,Y+7*np.sqrt(3)-7,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,Y+7*np.sqrt(3)-7,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one-one+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+4,X+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+4,X+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][1][3]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+3,Y+7*np.sqrt(3)-7,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,Y+7*np.sqrt(3)-7,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one-one+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+3,X+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+3,X+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][1][2]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+2,Y+7*np.sqrt(3)-7,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,Y+7*np.sqrt(3)-7,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one-one+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+2,X+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+2,X+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][1][1]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+1,Y+7*np.sqrt(3)-7,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,Y+7*np.sqrt(3)-7,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one-one+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+1,X+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+1,X+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][1][0]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X,Y+7*np.sqrt(3)-7,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,Y+7*np.sqrt(3)-7,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one-one+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one,X+7*np.sqrt(3)-7,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one,X+7*np.sqrt(3)-7,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

###################################################################################################

ooo = np.ones(4).reshape(2, 2)*oo3[6][0][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-7,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-7,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][0][5]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+5,Y+7*np.sqrt(3)-7,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,Y+7*np.sqrt(3)-7,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one-one+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+5,one+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+5,X+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+5,X+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][0][4]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+4,Y+7*np.sqrt(3)-7,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,Y+7*np.sqrt(3)-7,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one-one+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+4,one+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+4,X+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+4,X+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][0][3]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+3,Y+7*np.sqrt(3)-7,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,Y+7*np.sqrt(3)-7,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one-one+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+3,one+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+3,X+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+3,X+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][0][2]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+2,Y+7*np.sqrt(3)-7,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,Y+7*np.sqrt(3)-7,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one-one+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+2,one+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+2,X+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+2,X+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][0][1]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+1,Y+7*np.sqrt(3)-7,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,Y+7*np.sqrt(3)-7,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one-one+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+1,one+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+1,X+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+1,X+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[6][0][0]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X,Y+7*np.sqrt(3)-7,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,Y+7*np.sqrt(3)-7,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one-one+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X,one+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one,X+7*np.sqrt(3)-7,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one,X+7*np.sqrt(3)-7,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

###################################################################################################

ooo = np.ones(4).reshape(2, 2)*oo3[5][6][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-6,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-6,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-6,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-6,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[5][5][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-6,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-6,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-6,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-6,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-6,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-6,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[5][4][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-6,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-6,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-6,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-6,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-6,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-6,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[5][3][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-6,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-6,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-6,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-6,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-6,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-6,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[5][2][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-6,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-6,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-6,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-6,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-6,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-6,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[5][1][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-6,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-6,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-6,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-6,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-6,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-6,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[5][0][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-6,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-6,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-6,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-6,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-6,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-6,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

# ###################################################################################################

ooo = np.ones(4).reshape(2, 2)*oo3[4][6][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-5,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-5,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-5,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-5,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[4][5][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-5,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-5,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-5,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-5,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-5,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-5,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[4][4][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-5,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-5,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-5,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-5,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-5,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-5,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[4][3][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-5,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-5,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-5,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-5,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-5,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-5,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[4][2][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-5,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-5,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-5,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-5,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-5,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-5,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[4][1][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-5,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-5,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-5,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-5,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-5,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-5,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[4][0][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-5,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-5,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-5,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-5,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-5,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-5,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

# ###################################################################################################

ooo = np.ones(4).reshape(2, 2)*oo3[3][6][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-4,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-4,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-4,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-4,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[3][5][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-4,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-4,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-4,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-4,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-4,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-4,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[3][4][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-4,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-4,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-4,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-4,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-4,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-4,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[3][3][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-4,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-4,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-4,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-4,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-4,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-4,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[3][2][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-4,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-4,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-4,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-4,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-4,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-4,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[3][1][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-4,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-4,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-4,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-4,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-4,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-4,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[3][0][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-4,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-4,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-4,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-4,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-4,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-4,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

# ###################################################################################################

ooo = np.ones(4).reshape(2, 2)*oo3[2][6][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-3,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-3,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-3,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-3,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[2][5][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-3,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-3,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-3,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-3,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-3,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-3,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[2][4][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-3,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-3,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-3,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-3,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-3,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-3,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[2][3][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-3,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-3,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-3,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-3,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-3,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-3,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[2][2][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-3,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-3,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-3,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-3,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-3,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-3,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[2][1][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-3,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-3,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-3,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-3,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-3,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-3,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[2][0][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-3,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-3,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-3,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-3,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-3,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-3,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

# ###################################################################################################

ooo = np.ones(4).reshape(2, 2)*oo3[1][6][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-2,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-2,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-2,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-2,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[1][5][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-2,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-2,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-2,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-2,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-2,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-2,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[1][4][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-2,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-2,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-2,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-2,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-2,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-2,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[1][3][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-2,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-2,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-2,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-2,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-2,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-2,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[1][2][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-2,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-2,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-2,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-2,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-2,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-2,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[1][1][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-2,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-2,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-2,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-2,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-2,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-2,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[1][0][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-2,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-2,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-2,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-2,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-2,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-2,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

# ###################################################################################################

ooo = np.ones(4).reshape(2, 2)*oo3[0][6][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-1,one+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-1,one-one+6,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-1,Y+6, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-1,Y+6,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[0][5][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-1,one+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-1,one-one+5,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-1,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-1,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-1,Y+5, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-1,Y+5,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[0][4][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-1,one+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-1,one-one+4,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-1,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-1,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-1,Y+4, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-1,Y+4,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[0][3][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-1,one+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-1,one-one+3,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-1,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-1,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-1,Y+3, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-1,Y+3,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[0][2][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-1,one+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-1,one-one+2,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-1,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-1,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-1,Y+2, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-1,Y+2,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[0][1][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-1,one+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-1,one-one+1,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-1,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-1,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-1,Y+1, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-1,Y+1,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

ooo = np.ones(4).reshape(2, 2)*oo3[0][0][6]
my_col = cmap_r(ooo)
surf = ax1.plot_surface(X+6,Y+7*np.sqrt(3)-1,one, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,Y+7*np.sqrt(3)-1,one-one,facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one-one+7*np.sqrt(3)-1,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(X+6,one+7*np.sqrt(3)-1,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one+6,X+7*np.sqrt(3)-1,Y, facecolors = my_col, cmap=cmap_r,vmin=-0.18,vmax=0.18)
ax1.plot_surface(one-one+6,X+7*np.sqrt(3)-1,Y,facecolors = my_col,  cmap=cmap_r,vmin=-0.18,vmax=0.18)

# ###################################################################################################

N=2000
# Data for a three-dimensional line
zline1 = (np.linspace(0, 1, N))
xline1 = (np.sqrt(1-zline1*zline1))*7*np.sqrt(3)
yline1 = (np.ones(N))*7*np.sqrt(3)
zline2 = (np.linspace(0, 1, N))
xline2 = (np.zeros(N))*7*np.sqrt(3)
yline2 = (-np.sqrt(1-zline2*zline2)+1)*7*np.sqrt(3)
zline3 = (np.zeros(N))*7*np.sqrt(3)
xline3 = (np.linspace(0, 1, N))
yline3 = (1-np.sqrt(1-xline3*xline3))*7*np.sqrt(3)
zline1 = (np.linspace(0, 1, N))*7*np.sqrt(3)
xline3 = (np.linspace(0, 1, N))*7*np.sqrt(3)
zline2 = (np.linspace(0, 1, N))*7*np.sqrt(3)

xd1 = (np.zeros(N))*7*np.sqrt(3)
yd1 = (np.ones(N))*7*np.sqrt(3)
zd1 = (np.linspace(0, 1, N))*7*np.sqrt(3)
xd2 = (np.zeros(N))*7*np.sqrt(3)
yd2 = (np.linspace(0, 1, N))*7*np.sqrt(3)
zd2 = (np.zeros(N))*7*np.sqrt(3)
xd3 = (np.linspace(0, 1, N))*7*np.sqrt(3)
yd3 = (np.ones(N))*7*np.sqrt(3)
zd3 = (np.zeros(N))*7*np.sqrt(3)

s = 1/np.sqrt(3)
x11 = (np.zeros(N))*7*np.sqrt(3)
y11 = (np.linspace(1-s,1,N))*7*np.sqrt(3)
z11 = (np.zeros(N)+s)*7*np.sqrt(3)
x12 = (np.zeros(N))*7*np.sqrt(3)
y12 = (np.zeros(N)+1-s)*7*np.sqrt(3)
z12 = (np.linspace(0,s,N))*7*np.sqrt(3)

x21 = (np.linspace(0,s,N))*7*np.sqrt(3)
y21 = (np.ones(N))*7*np.sqrt(3)
z21 = (np.zeros(N)+s)*7*np.sqrt(3)
x22 = (np.zeros(N)+s)*7*np.sqrt(3)
y22 = (np.ones(N))*7*np.sqrt(3)
z22 = (np.linspace(0,s,N))*7*np.sqrt(3)

x31 = (np.linspace(0,s,N))*7*np.sqrt(3)
y31 = (np.ones(N)-s)*7*np.sqrt(3)
z31 = (np.zeros(N))*7*np.sqrt(3)
x32 = (np.zeros(N)+s)*7*np.sqrt(3)
y32 = (np.linspace(1-s,1,N))*7*np.sqrt(3)
z32 = (np.zeros(N))*7*np.sqrt(3)

x41 = (np.linspace(0,s,N))*7*np.sqrt(3)
y41 = (np.ones(N)-s)*7*np.sqrt(3)
z41 = (np.zeros(N)+s)*7*np.sqrt(3)
x42 = (np.zeros(N)+s)*7*np.sqrt(3)
y42 = (np.linspace(1-s,1,N))*7*np.sqrt(3)
z42 = (np.zeros(N)+s)*7*np.sqrt(3)

xx = (np.zeros(N)+s)*7*np.sqrt(3)
yy = (np.zeros(N)+1-s)*7*np.sqrt(3)
zz = (np.linspace(0,s,N))*7*np.sqrt(3)

def get_cube():   
    phi = np.arange(1,10,2)*np.pi/4
    Phi, Theta = np.meshgrid(phi, phi)

    x = np.cos(Phi)*np.sin(Theta)
    y = np.sin(Phi)*np.sin(Theta)
    z = np.cos(Theta)/np.sqrt(2)
    return x,y,z

a = 1
b = 1
c = 1
x,y,z = get_cube()

ax1.plot3D(x11,y11,z11,'black')
ax1.plot3D(x12,y12,z12,'black')
ax1.plot3D(x21,y21,z21,'black')
ax1.plot3D(x22,y22,z22,'black')
ax1.plot3D(x31,y31,z31,'black')
ax1.plot3D(x32,y32,z32,'black')
ax1.plot3D(x41,y41,z41,'black')
ax1.plot3D(x42,y42,z42,'black')
ax1.plot3D(xx,yy,zz,'black')
ax1.plot3D(xd1,yd1,zd1,'black',ls='--')
ax1.plot3D(xd2,yd2,zd2,'black',ls='--')
ax1.plot3D(xd3,yd3,zd3,'black',ls='--')
ax1.plot3D(xline1, yline1, zline1,'black')
ax1.plot3D(xline2, yline2, zline2,'black')
ax1.plot3D(xline3, yline3, zline3,'black')

# fig.colorbar(surf, label=r'x')
ax1.grid(False)

# Hide ax1es ticks
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])
ax1.set_axis_off()
# ax1.text(0,-0.33*7*np.sqrt(3), 0, r"$z=1.4$", color='blue',fontsize=23)
# ax1.text(1*7*np.sqrt(3),1.04*7*np.sqrt(3), 0, r"$z=1.4$", color='blue',fontsize=23)
# ax1.text(0-0.7,7, 1.06, r"$z=1.4$", color='blue',fontsize=23)
ax1.text(5,-0.33*7*np.sqrt(3)-1, 0, r"$d_{\rm m}^{\rm LC}(\vec{r})$",fontsize=23)
# ax1.text((s-0.1)*7*np.sqrt(3),(1-s-0.06)*7*np.sqrt(3), (s+0.02)*14, r"$z=1.4$", color='blue',fontsize=23)
# ax1.text(0,1.06*7*np.sqrt(3), 0, r"$O$", color='blue',fontsize=23)
ax1.xaxis.pane.fill = False
ax1.yaxis.pane.fill = False
ax1.zaxis.pane.fill = False
ax1.xaxis.pane.set_edgecolor('w')
ax1.yaxis.pane.set_edgecolor('w')
ax1.zaxis.pane.set_edgecolor('w')

axlist = [ax,ax1]
cbar = fig.colorbar(surf, ax=axlist,cmap=cmap_r,ticks=[0,0.25, 0.5, 0.75,1],aspect=40,pad=0.08,shrink=0.904)
cbar.ax.set_yticklabels([r'-0.18','-0.09','0','0.09', r'0.18'],fontsize=18) 

# ax.set_xlim(0,14)
# ax.set_ylim(0,14)
# ax.set_zlim(0,14)
# plt.savefig('cube_l.pdf')
plt.show()