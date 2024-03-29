import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymetalog as pm


fish_data = pm.example_data

# metalog creation
fish_metalog = pm.metalog(
    x=fish_data,
    bounds=[0, 40],
    boundedness="b",
    term_limit=15,
    term_lower_bound=2,
    step_len=0.001,
    penalty=None,
)

# summary function
pm.summary(fish_metalog)

# # plot function - right now this saves plots to local
pm.plot(fish_metalog)
plt.show()

# # metalog random sampling
r_gens = pm.rmetalog(fish_metalog, n=1000, term=9, generator="hdr")
plt.hist(r_gens, 14)
plt.show()

# quantiles from a percentile
qs = pm.qmetalog(fish_metalog, y=[0.25, 0.5, 0.75], term=9)
print("qmetalog demo: " + str(qs))

# probabilities from a quantile
ps = pm.pmetalog(fish_metalog, q=[3, 10, 25], term=9)
print("pmetalog demo: " + str(ps))

# density from a quantile
ds = pm.dmetalog(fish_metalog, q=[3, 10, 25], term=9)
print("dmetalog demo: " + str(ds))
