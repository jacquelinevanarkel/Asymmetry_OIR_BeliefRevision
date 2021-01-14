import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import matplotlib as mpl

# Read in data
#path = Path('/Users/Jacqueline/Documents/Master_Thesis/Simulation')
# results = pd.read_pickle("results.p")
# print(results[:10])

if __name__ == '__main__':
    results = pd.read_pickle("results.p")
    pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.max_colwidth', -1)
    #print(results.loc[results["n_repair"] > 0])
    #print(results.loc[results['n_interactions'] > 1])
    #print(results["ended max sim"][:10])
    #print(results.loc[results["ended max sim"]==False])
    #print(results[:20])
    #print(results[-1:])
    print(results[:30])
