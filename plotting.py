import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools

# Read in data
# Simulation run 1
results_8_10_12 = pd.read_pickle("results.p")

# Simulation run 2
results_8 = pd.read_pickle("results_8.p")
results_10 = pd.read_pickle("results_10.p")
results_12 = pd.read_pickle("results_12.p")

# Simulation run 3
results_8_3 = pd.read_pickle("/Users/Jacqueline/Documents/Master_Thesis/Simulation/run3/results_8.p")
results_10_3 = pd.read_pickle("/Users/Jacqueline/Documents/Master_Thesis/Simulation/run3/results_10.p")
results_12_3 = pd.read_pickle("/Users/Jacqueline/Documents/Master_Thesis/Simulation/run3/results_12.p")

results = pd.concat([results_8_10_12, results_8, results_10, results_12, results_8_3, results_10_3, results_12_3])

# As a default, set the asymmetry level to 0 when the overlap is 0 (as the asymmetry doesn't play a role)
results.loc[results.overlap == 0, "asymmetry"] = 0

# First select the last row of information of the conversation for every result
df8 = results_8.drop_duplicates(subset=["simulation_number"], keep='last')
df10 = results_10.drop_duplicates(subset=["simulation_number"], keep='last')
df12 = results_12.drop_duplicates(subset=["simulation_number"], keep='last')
df8_10_12 = results_8_10_12.drop_duplicates(subset=["simulation_number"], keep='last')
df8_3 = results_8_3.drop_duplicates(subset=["simulation_number"], keep='last')
df10_3 = results_10_3.drop_duplicates(subset=["simulation_number"], keep='last')
df12_3 = results_12_3.drop_duplicates(subset=["simulation_number"], keep='last')

# Then add the last rows together to form a dataframe of the last 'state' of every conversation
df = pd.concat([df8, df10, df12, df8_10_12, df8_3, df10_3, df12_3])

# As a default, set the asymmetry level to 0 when the overlap is 0 (as the asymmetry doesn't play a role)
df.loc[df.overlap == 0, "asymmetry"] = 0

# Print the number of results for the different numbers of nodes
print("n results 8 nodes: ", len(df[df['n_nodes'] == 8]))
print("n results 10 nodes: ", len(df[df['n_nodes'] == 10]))
print("n results 12 nodes: ", len(df[df['n_nodes'] == 12]))

# If you want to view the data, here are some options to view the entire dataframe
pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.max_colwidth', -1)

# ----------------------------------------------------------------------------------------------------------------------
# Subplots: degree of asymmetry and overlap on mean number of times repair initiated and mean number of times intention
# communicated
# ----------------------------------------------------------------------------------------------------------------------

# Plot: degree of overlap & degree of asymmetry --> n_repair

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

# Version 2: how often agents initiate repair (x-axis) for different success rates (asymmetry in intention)
intention_count = df.groupby(['n_repair'])['asymmetry_intention'].value_counts(normalize=True)\
    .reset_index(name='Counts')
sns.barplot(x="asymmetry_intention", y="Counts", hue="n_repair", data=intention_count)
plt.title("Number of times the conversation is ended with a certain asymmetry level of the intention for different "
          "number of times repair is initiated")
plt.ylabel("Mean count")
plt.xlabel("Asymmetry of intention")
plt.legend(loc='upper right', title="Number of times \n repair is initiated")

df_asymmetry_intention = df.groupby(["n_repair", "overlap", "asymmetry"])["asymmetry_intention"].\
    value_counts(normalize=True).reset_index(name='Mean count')

g = sns.FacetGrid(df_asymmetry_intention, col="asymmetry", row="overlap")
g.map(sns.barplot, "asymmetry_intention", "Mean count", "n_repair", order=[0, 1, 2, 3, 4, 5, 6, 7, 8],
      hue_order=[0, 1, 2])
g.add_legend(title="Number of times repair initiated")

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# ---------------- The asymmetry over number of turns for different number of times repair is initiated ----------------
# ----------------------------------------------------------------------------------------------------------------------

# Plot: asymmetry intention over time

# Calculate the mean asymmetry (of the whole network) for every turn when repair is initiated
# results['index1'] = results.index
# results = results.fillna(value=np.nan)
#
# conditions = [(results['conversation state'] == 0) & (results['n_turns'] == 1),
#               (results['conversation state'] == 0) & (results['n_turns'] == 2),
#               (results['conversation state'] == 0) & (results['n_turns'] == 1) & (results['repair request'] is False),
#               (results['conversation state'] == 0) & (results['n_turns'] == 3) & (results['utterance speaker'] != np.nan),
#               (results['conversation state'] == 0) & (results['n_turns'] == 3) & (results['utterance speaker'].isna()),
#
#               (results['conversation state'] == 1) & (),
#               (results['conversation state'] == 1) & (results['utterance speaker'].isna()) & (results['repair request'].isna()),
#               (results['conversation state'] == 1) & (results['n_turns'] == 1) & (results['repair request'] is False),
#               (results['conversation state'] == 1) & (results['n_turns'] == 3) & (results['utterance speaker'] is not
#                                                                                   None),
#               (results['conversation state'] == 1) & (results['n_turns'] == 3) & (results['utterance speaker'].isna()),
#
#               (results['conversation state'] == 2) & (results['n_turns'] == 1),
#               (results['conversation state'] == 2) & (results['n_turns'] == 2),
#               (results['conversation state'] == 2) & (results['n_turns'] == 1) & (results['repair request'] is False),
#               (results['conversation state'] == 2) & (results['n_turns'] == 3) & (results['utterance speaker'] is not
#                                                                                   None),
#               (results['conversation state'] == 2) & (results['n_turns'] == 3) & (results['utterance speaker'].isna())
# ]
#
# values = ['initialisation', 'listener update utterance', 'listener update utterance no repair', 'speaker update repair',
#           'listener update solution', 'initialisation 2', 'listener update utterance 2',
#           'listener update utterance no repair 2', 'speaker update repair 2', 'listener update solution 2',
#           'initialisation 3', 'listener update utterance 3', 'listener update utterance no repair 3',
#           'speaker update repair 3', 'listener update solution 3']
# results['state'] = np.select(conditions, values)
#
# results['asymmetry_count'] = results['asymmetry_count'].astype(int)
#
# asymmetry_repair = results.groupby(["n_turns", "n_repair", "state", "conversation state"], as_index=False)['asymmetry_count'].mean()
# print(asymmetry_repair)
#
# sns.lineplot(x="state", y="asymmetry_count", hue="n_repair", data=asymmetry_repair)
#
# plt.title("Mean asymmetry over turns with and without repair initiated")
# plt.ylabel("Asymmetry")
# plt.xlabel("Turn number")
#
# plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# -------------- The mean intention communicated for different amounts of nodes and edges in the network ---------------
# ----------------------------------------------------------------------------------------------------------------------

# Plot: number of nodes & number of edges of intention communicated (/asymmetry in intention?)
# df_intention = df.groupby(['n_nodes', 'amount_edges'])['intention_communicated'].value_counts(normalize=True)\
#     .reset_index(name='Counts')
# df_intention_true = df_intention[df_intention["intention_communicated"] == True]
# print(df_intention_true)

# Compute chance levels for every node type
nodes_8 = 0.5**(0.75 * 8)
nodes_10 = 0.5**(0.75 * 10)
nodes_12 = 0.5**(0.75 * 12)

sns.barplot(x="n_nodes", y="intention_communicated", hue="amount_edges", data=df)
plt.title("Mean intention communicated for different amounts of nodes and edges in the network")
plt.ylabel("Mean intention communicated")
plt.xlabel("Number of nodes")
plt.ylim(0, 1)
plt.hlines(y=(nodes_8, nodes_10, nodes_12), xmin=(-0.4, 0.6, 1.6), xmax=(0.4, 1.4, 2.4), colors="black")

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------- Mean lengths of utterances spoken by speaker and listener ------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# Calculate the mean utterance length of the speaker
utterances = results["utterance speaker"].dropna()
utterances["length"] = utterances.str.len()
print("Mean utterance length :", utterances["length"].mean())

# Calculate the mean length of the repair request
repair_df = results
repair_df["length"] = repair_df["repair request"].str.len()
print("Mean repair request length: ", repair_df["length"].mean())

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------- Similarity speaker matches asymmetry in intention? ----------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# Plot the different asymmetry levels of the intention when the conversation was ended because of max sim for degree of
# overlap and asymmetry
df['intention length'] = df['intention'].str.len()
df['normalised asymmetry'] = df['asymmetry_intention']/df['intention length']
df['normalised asymmetry'] = df['normalised asymmetry'].astype(float)

# sns.violinplot(x="asymmetry", y="normalised asymmetry", hue="overlap", data=df)
# sns.stripplot(x="asymmetry", y="normalised asymmetry", hue="overlap", data=df, dodge=True)
g = sns.catplot(x="asymmetry", y="normalised asymmetry", hue="overlap", col="n_nodes", kind="violin", data=df, legend_out=True)
# plt.title("Degree of \n overlap")
# plt.ylabel("Normalised asymmetry intention")
# plt.xlabel("Degree of asymmetry")
g.set_xlabels('Degree of asymmetry')
g.set_ylabels('Normalised asymmetry intention')
g._legend.set_title("Degree of \n overlap")
new_labels = ['0%', '50%', '100%']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)

plt.show()

