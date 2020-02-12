import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymetalog as pm


fish_data = np.loadtxt('fishout.csv', delimiter=',', skiprows=1, dtype='str')[:,1].astype(np.float)

# metalog creation
fish_metalog = pm.metalog(x=fish_data, bounds=[0,40], boundedness='b', term_limit=15, term_lower_bound=2, step_len=.001, penalty=None, save_data=True)

# summary function
pm.summary(fish_metalog)

# # plot function - right now this saves plots to local
pm.plot(fish_metalog)
plt.show()

# # metalog random sampling
r_gens = pm.rmetalog(fish_metalog, n = 1000, term = 9, generator='hdr')
plt.hist(r_gens,14)
plt.show()

# quantiles from a percentile
qs = pm.qmetalog(fish_metalog, y = [0.25, 0.5, 0.75], term = 9)
print("qmetalog demo: "+str(qs))

# probabilities from a quantile
ps = pm.pmetalog(fish_metalog, q = [3, 10, 25], term = 9)
print("pmetalog demo: "+str(ps))

# density from a quantile
ds = pm.dmetalog(fish_metalog, q = [3, 10, 25], term = 9)
print("dmetalog demo: "+str(ds))

# Bayesian metalog updating
# Will split dataset and fit to one part and update using other part

np.random.seed(37)
train_percent = 0.1

training_idx = np.random.randint(fish_data.shape[0], size=int(fish_data.shape[0] * train_percent))
test_idx = np.random.randint(fish_data.shape[0], size=int(fish_data.shape[0] * (1-train_percent)))
training, test = fish_data[training_idx], fish_data[test_idx]

fish_metalog = pm.metalog(x=training, bounds=[0,40], boundedness='b', term_limit=13, term_lower_bound=2, step_len=.001, penalty='l2', alpha=0.0001, save_data=True)

pm.summary(fish_metalog)
pm.plot(fish_metalog)
plt.show()

fish_metalog2 = pm.update(fish_metalog, test, penalty='l2', alpha=0.0001)

pm.summary(fish_metalog2)
pm.plot(fish_metalog2)
plt.show()

