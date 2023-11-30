import numpy as np
import matplotlib.pyplot as plt
import os
import scipy

## Set node and time window
start = 1000
end = start + 500
node = 190
step_ahead = 6
## Set path
path = 'plot/bay_old/'

data_real = np.load(path+'y.npy')
data_hat = np.load(path+'yhat.npy')

yreal = []
yhat = []

for i in range(start,end+1):
    a = data_real[i,:,:]
    a = a[:, step_ahead - 1]
    a = np.expand_dims(a, axis=1)
    yreal.append(a)
yreal = np.concatenate(yreal,axis=-1)

for i in range(start,end+1):
    a = data_hat[i,:,:]
    a = a[:, step_ahead - 1]
    a = np.expand_dims(a, axis=1)
    yhat.append(a)
yhat = np.concatenate(yhat,axis=-1)

yreal = yreal[node,:]
yhat = yhat[node,:]
x = range(1,yhat.size+1)
#plot
fig, ax = plt.subplots()
ax.plot(x, yreal, label='Ground truth', linewidth=0.4)
ax.plot(x, yhat, label='Prediction', linewidth=0.4)
ax.set_xlabel('time(hour)')
ax.set_ylabel('Travel speed (mph)')
ax.legend()
plt.ylim(20, 80)

#plt.show(dpi=300)
plt.savefig(path+"node"+str(node)+':'+str(start)+'_'+str(end)+".png", dpi=300)



