# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import numpy as np
import matplotlib.pyplot as plt


def plot_comparisons(axis, y_vals, labels, x_vals=[], plotdiff=False, axlog="", 
                 xlim=[], ylim=[], xlabel="", ylabel="", legendloc= "",
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

    if axlog == "y":
        plotcommand = axis.semilogy
    elif axlog == "xy":
        plotcommand = axis.loglog      
    else:
        plotcommand = axis.plot
            
    for i in range(len(y_vals)):
        if len(x_vals) > 0:
            xi = x_vals[i]
        else:
            xi = np.arange(len(y_vals[i])) + 1
            
        plotcommand(xi, y_vals[i]-y_shift, linestyles[i], label=labels[i]) 
                    #dashes=linedash[i], label=labels[i])
    
    if len(legendloc) > 0:
        axis.legend(loc=legendloc)
    else:
        axis.legend()
        
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
 
    if len(xlim) > 0:
        axis.set_xlim(xlim)
    if len(ylim) > 0:
        axis.set_ylim(ylim)
