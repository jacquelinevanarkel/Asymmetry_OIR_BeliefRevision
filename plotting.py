import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import matplotlib as mpl

# Read in data
#path = Path('/Users/Jacqueline/Documents/Master_Thesis/Simulation')
results = pd.read_pickle("results.p")

# Main manipulations: degree of overlap & degree of asymmetry --> effect on repair -->
# repair effect on intention communicated? / degree asymmetry

# Plot: degree of overlap & degree of asymmetry --> n_repair
#i = results.index.get_loc('0')
#df.iloc[i-1]

# Plot: degree of overlap & degree of asymmetry --> intention communicated

# Plot: asymmetry intention over time

if __name__ == '__main__':
    results = pd.read_pickle("results.p")
    pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.max_colwidth', -1)
    #print(results.loc[results["n_repair"] > 0])
    print(results.loc[results['n_interactions'] > 1])
    #print(results["ended max sim"][:10])
    #print(results.loc[results["ended max sim"]==False])
    #print(results[:20])
    #print(results[-1:])
    #print(results[:30])
