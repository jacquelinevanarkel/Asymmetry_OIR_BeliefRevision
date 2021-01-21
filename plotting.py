import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read in data
# results_8 = pd.read_pickle("/Users/Jacqueline/Documents/Master_Thesis/Simulation/run1/results_8.p")
# results_10 = pd.read_pickle("/Users/Jacqueline/Documents/Master_Thesis/Simulation/run1/results_10.p")
results_12 = pd.read_pickle("results_12.p")
# print(results_8.head())
# print(results_10.columns)
# results = pd.concat([results_8, results_10])

results = pd.read_pickle("results.p")

# If you want to view the data, here are some options to view the entire dataframe
pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.max_colwidth', -1)

# Main manipulations: degree of overlap & degree of asymmetry --> effect on repair -->
# repair effect on intention communicated? / degree asymmetry

# ----------------------------------------------------------------------------------------------------------------------
# Subplots: degree of asymmetry and overlap on mean number of times repair initiated and mean number of times intention
# communicated
# ----------------------------------------------------------------------------------------------------------------------

# Plot: degree of overlap & degree of asymmetry --> n_repair

# First select the last row of information of the conversation
df = results.drop_duplicates(subset=["simulation_number"], keep='last')

fig, ax = plt.subplots(1,2)

# Create the plots and set labels and titles
sns.barplot(x="overlap", y="n_repair", hue="asymmetry", data=df, ax=ax[0])
ax[0].set_ylabel("Mean number of repair initiations")
ax[0].set_xlabel("Degree of overlap")
ax[0].set_ylim(0,1)

# Plot: degree of overlap & degree of asymmetry --> intention communicated

# Create the plots and set labels and titles
sns.barplot(x="overlap", y="intention_communicated", hue="asymmetry", data=df, ax=ax[1])
ax[1].set_ylabel("Mean number of times intention communicated")
ax[1].set_xlabel("Degree of overlap")
ax[1].set_ylim(0,1)

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# ---------------- The asymmetry over number of turns for different number of times repair is initiated ----------------
# ----------------------------------------------------------------------------------------------------------------------

# Plot: asymmetry intention over time

# Calculate the mean asymmetry for every turn when repair is initiated
results['asymmetry_count'] = results['asymmetry_count'].astype(int)

asymmetry_repair = results.groupby(["n_turns", "n_repair"], as_index=False)['asymmetry_count'].mean()

sns.lineplot(x="n_turns", y="asymmetry_count", hue="n_repair", data=asymmetry_repair)

plt.title("Mean asymmetry over turns with and without repair initiated")
plt.ylabel("Asymmetry")
plt.xlabel("Turn number")

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# -------------- The mean intention communicated for different amounts of nodes and edges in the network ---------------
# ----------------------------------------------------------------------------------------------------------------------

# Plot: number of nodes & number of edges of intention communicated (/asymmetry in intention?)
df_intention = df.groupby(['n_nodes', 'amount_edges'])['intention_communicated'].value_counts(normalize=True)\
    .reset_index(name='Counts')
df_intention_true = df_intention[df_intention["intention_communicated"] == True]

sns.barplot(x="n_nodes", y="Counts", hue="amount_edges", data=df_intention_true)
plt.title("Mean intention communicated for different amounts of nodes and edges in the network")
plt.ylabel("Mean intention communicated")
plt.xlabel("Number of nodes")
plt.ylim(0, 1)

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------- Mean lengths of utterances spoken by speaker and listener ------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# Calculate the mean utterance length of the speaker
utterances = results["utterance speaker"].dropna()
utterances["length"] = utterances.str.len()
print(utterances["length"].mean())

# Calculate the mean length of the repair request
repair_df = results
repair_df["length"] = repair_df["repair request"].str.len()
print(repair_df["length"].mean())

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------- Similarity speaker matches asymmetry in intention? ----------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# Plot the different asymmetry levels of the intention when the conversation was ended because of max sim for degree of
# overlap and asymmetry

df2 = df.groupby(["asymmetry_intention", "overlap", "asymmetry"])["conversation ended max sim"].value_counts()\
    .reset_index(name='Counts')

g = sns.FacetGrid(df2, col="asymmetry_intention", col_wrap=3)
g.map(sns.barplot, "asymmetry", "Counts", "overlap", order=[0, 50, 100], hue_order=[0, 50, 100])
g.add_legend(title="Degree of overlap")
plt.show()

