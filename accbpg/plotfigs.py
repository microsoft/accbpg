# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import numpy as np
#import matplotlib.pyplot as plt
from matplotlib.pyplot import *


def plot_comparisons(axis, y_vals, labels, x_vals=[], plotdiff=False, 
                 yscale="linear", xscale="linear", 
                 xlim=[], ylim=[], xlabel="", ylabel="", legendloc=0,
                 linestyles=['k:', 'g-', 'b-.', 'k-', 'r--', 'k-', 'm-'],
                 linedash=[[1,2], [], [4,2,1,2], [], [4,2], [], [], []]):
    """
    Plot comparison figures using matplotlib.pyplot.
    """
    
    y_shift = 0
    if plotdiff:
        y_shift = y_vals[0].min()
        for i in range(len(y_vals)):
            y_shift = min(y_shift, y_vals[i].min())     

    for i in range(len(y_vals)):
        if len(x_vals) > 0:
            xi = x_vals[i]
        else:
            xi = np.arange(len(y_vals[i])) + 1
            
        axis.plot(xi, y_vals[i]-y_shift, linestyles[i], label=labels[i], 
                  dashes=linedash[i])
    
    axis.set_xscale(xscale)
    axis.set_yscale(yscale)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if legendloc == "no":
        pass
    elif legendloc == "outside":
        axis.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    else: 
        axis.legend(loc=legendloc)
 
    if len(xlim) > 0:
        axis.set_xlim(xlim)
    if len(ylim) > 0:
        axis.set_ylim(ylim)
