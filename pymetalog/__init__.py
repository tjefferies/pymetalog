
import os
import numpy as np
from .metalog import metalog
from .class_method import rmetalog, plot, qmetalog, pmetalog, dmetalog, summary, update

name = "pymetalog"

this_pth = os.path.dirname(__file__)
data_path = os.path.join(this_pth, 'examples', 'fishout.csv')
example_data = np.loadtxt(data_path, delimiter=',', skiprows=1, dtype='str')[:,1].astype(np.float)

