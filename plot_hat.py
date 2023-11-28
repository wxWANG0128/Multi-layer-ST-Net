import numpy as np
import matplotlib.pyplot as plt
import os
import transbigdata as tbd
import matplotlib.colors as mcolors

path = 'plot/metr/'

y12 = np.load(path+'y12.npy')
y12 = np.expand_dims(y12,1)
yhat12 = np.load(path+'yhat12.npy')
yhat12 = np.expand_dims(yhat12,1)
y3 = np.load(path+'y3.npy')
y3 = np.expand_dims(y3,1)
yhat3 = np.load(path+'yhat3.npy')
yhat3 = np.expand_dims(yhat3,1)
if path == 'plot/metr/':
    sensor = np.loadtxt('data/sensor_graph/graph_sensor_locations.csv', delimiter=",")[:, -2:]
else:
    sensor = np.loadtxt('data/sensor_graph/graph_sensor_locations_bay.csv', delimiter=",")[:, -2:]


plot_y12 = np.concatenate((sensor, y12),axis=1)
plot_yhat12 = np.concatenate((sensor, yhat12),axis=1)
plot_y3 = np.concatenate((sensor, y3),axis=1)
plot_yhat3 = np.concatenate((sensor, yhat3),axis=1)

b1 = min(sensor[:,1])-0.04
b2 = min(sensor[:,0])-0.04
b3 = max(sensor[:,1])+0.04
b4 = max(sensor[:,0])+0.04

bounds = [b1, b2, b3, b4]
# Plot Frame
fig =plt.figure(1,(10,10),dpi=300)
ax =plt.subplot(111)
plt.sca(ax)
# Add map basemap
tbd.plot_map(plt,bounds,zoom = 11,style = 4)
# Add scale bar and north arrow
tbd.plotscale(ax,bounds = bounds,textsize = 10,compasssize = 1,accuracy = 500,rect = [0.06,0.03],zorder = 10)
plt.xlim(bounds[0],bounds[2])
plt.ylim(bounds[1],bounds[3])
colors = plot_yhat12[:,2]
cmap = mcolors.ListedColormap(['red', 'orange','yellow','green', 'blue'])
plt.scatter(plot_yhat12[:,1], plot_y12[:,0], s=20, c=colors ,cmap=cmap, vmin=20, vmax=70)
plt.colorbar(shrink=0.65, aspect=20)
plt.savefig(path+'60_hat.png', transparent=None, dpi=300)