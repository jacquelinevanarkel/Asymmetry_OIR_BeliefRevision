import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Read in data

# Simulation run 1
# results_8 = pd.read_pickle("results_8.p")
# results_10 = pd.read_pickle("results_10.p")
# results_12 = pd.read_pickle("results_12_1run.p")

# # Simulation run 2
# results_8_2 = pd.read_pickle("/Users/Jacqueline/Documents/Master_Thesis/Simulation/run3/results_8.p")
# results_10_2 = pd.read_pickle("/Users/Jacqueline/Documents/Master_Thesis/Simulation/run3/results_10.p")
# results_12_2 = pd.read_pickle("/Users/Jacqueline/Documents/Master_Thesis/Simulation/run3/results_12_4run.p")

# Simulation final runs

# # 1260 runs (70 of combination of manipulations)
# results_8_1 = pd.read_pickle("/Users/Jacqueline/Documents/Master_Thesis/Simulation/Finalrun/3/results_8.p")
# results_10_1 = pd.read_pickle("/Users/Jacqueline/Documents/Master_Thesis/Simulation/Finalrun/3/results_10.p")
# results_12_1 = pd.read_pickle("/Users/Jacqueline/Documents/Master_Thesis/Simulation/Finalrun/3/results_12.p")
#
# # 4140 runs (230 of combination of manipulations)
# results_8_2 = pd.read_pickle("/Users/Jacqueline/Documents/Master_Thesis/Simulation/Finalrun/4/results_8.p")
# results_8_2["simulation_number"] = results_8_2["simulation_number"] + 1260
# results_10_2 = pd.read_pickle("/Users/Jacqueline/Documents/Master_Thesis/Simulation/Finalrun/4/results_10.p")
# results_10_2["simulation_number"] = results_10_2["simulation_number"] + 1260
# results_12_2 = pd.read_pickle("/Users/Jacqueline/Documents/Master_Thesis/Simulation/Finalrun/4/results_12.p")
# results_12_2["simulation_number"] = results_12_2["simulation_number"] + 1260
#
# results_8 = pd.concat([results_8_1, results_8_2])
# results_10 = pd.concat([results_10_1, results_10_2])
# results_12 = pd.concat([results_12_1, results_12_2])

results_8 = pd.read_pickle("/Users/Jacqueline/Documents/Master_Thesis/Simulation/Finalrun/5/results_8.p")
results_10 = pd.read_pickle("/Users/Jacqueline/Documents/Master_Thesis/Simulation/Finalrun/5/results_10.p")
results_12 = pd.read_pickle("/Users/Jacqueline/Documents/Master_Thesis/Simulation/Finalrun/5/results_12.p")
#results_14 = pd.read_pickle("/Users/Jacqueline/Documents/Master_Thesis/Simulation/Finalrun/4/results_14.p")

results = pd.concat([results_8, results_10, results_12])

# As a default, set the asymmetry level to 0 when the overlap is 0 (as the asymmetry doesn't play a role)
results.loc[results.overlap == 0, "asymmetry"] = 0

# First select the last row of information of the conversation for every result
df8 = results_8.drop_duplicates(subset=["simulation_number"], keep='last')
df10 = results_10.drop_duplicates(subset=["simulation_number"], keep='last')
df12 = results_12.drop_duplicates(subset=["simulation_number"], keep='last')
#df14 = results_14.drop_duplicates(subset=["simulation_number"], keep='last')
# df8_2 = results_8_2.drop_duplicates(subset=["simulation_number"], keep='last')
# df10_2 = results_10_2.drop_duplicates(subset=["simulation_number"], keep='last')
# df12_2 = results_12_2.drop_duplicates(subset=["simulation_number"], keep='last')

# Then add the last rows together to form a dataframe of the last 'state' of every conversation
df = pd.concat([df8, df10, df12])

# As a default, set the asymmetry level to 0 when the overlap is 0 (as the asymmetry doesn't play a role)
df.loc[df.overlap == 0, "asymmetry"] = 0

# Print the number of results for the different numbers of nodes
print("n results 8 nodes: ", len(df[df['n_nodes'] == 8]))
print("n results 10 nodes: ", len(df[df['n_nodes'] == 10]))
print("n results 12 nodes: ", len(df[df['n_nodes'] == 12]))
#print("n results 14 nodes: ", len(df[df['n_nodes'] == 14]))

# If you want to view the data, here are some options to view the entire dataframe
pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.max_colwidth', -1)

# Seaborn settings used for plots
sns.set_style("whitegrid")
colors = ['#bdd7e7', '#6baed6', '#3182bd', '#08519c']
sns.set_palette(sns.color_palette(colors))
# ----------------------------------------------------------------------------------------------------------------------
# Subplots: degree of asymmetry and overlap on mean number of times repair initiated and mean number of times intention
# communicated
# ----------------------------------------------------------------------------------------------------------------------

# Plot: degree of overlap & degree of asymmetry --> n_repair

# fig, ax = plt.subplots(1,2)
#
# # Create the plots and set labels and titles
# sns.barplot(x="overlap", y="n_repair", hue="asymmetry", data=df, ax=ax[0])
# ax[0].set_ylabel("Mean number of repair initiations")
# ax[0].set_xlabel("Degree of overlap")
# ax[0].set_ylim(0,1)
#
# # Plot: degree of overlap & degree of asymmetry --> intention communicated
#
# # Create the plots and set labels and titles
# sns.barplot(x="overlap", y="intention_communicated", hue="asymmetry", data=df, ax=ax[1])
# ax[1].set_ylabel("Mean number of times intention communicated")
# ax[1].set_xlabel("Degree of overlap")
# ax[1].set_ylim(0,1)
#
# plt.show()

# Version 2: how often agents initiate repair (x-axis) for different success rates (asymmetry in intention)
intention_count = df.groupby(['n_repair'])['asymmetry_intention'].value_counts(normalize=True)\
    .reset_index(name='Counts')
sns.barplot(x="asymmetry_intention", y="Counts", hue="n_repair", data=intention_count)
#plt.title("Number of times the conversation is ended with a certain asymmetry level of the intention for different "
#          "number of times repair is initiated")
plt.legend(loc='upper right', title="nRepair")
plt.xlabel("Asymmetry of intention")
plt.ylabel("Mean count")

# Normalised asymmetry of intention
df_asymmetry_intention = df.groupby(["n_repair", "overlap", "asymmetry"])["asymmetry_intention"].\
    value_counts(normalize=True).reset_index(name='Mean count')

g = sns.FacetGrid(df_asymmetry_intention, col="asymmetry", row="overlap")
g.map(sns.barplot, "asymmetry_intention", "Mean count", "n_repair", order=[0, 1, 2, 3, 4, 5, 6, 7, 8],
      hue_order=[0, 1, 2, 3], palette=colors)
g.add_legend(title="nRepair")
g.set_xlabels("Asymmetry of intention")

plt.show()

# Version 3: normalised asymmetry
df['intention length'] = df['intention'].str.len()
df['normalised asymmetry'] = df['asymmetry_intention']/df['intention length']
df['normalised asymmetry'] = df['normalised asymmetry'].astype(float)

intention_count_normalised = df.groupby(['n_repair'])['normalised asymmetry'].value_counts(normalize=True)\
    .reset_index(name='Counts')

intention_count_normalised['normalised asymmetry bins'] = pd.qcut(intention_count_normalised['normalised asymmetry'],
                                                                  q=10)

sns.barplot(x="normalised asymmetry bins", y="Counts", hue="n_repair", data=intention_count_normalised)
# plt.title("Number of times the conversation is ended with a certain normalised asymmetry level of the intention for "
#           "different number of times repair is initiated")
plt.ylabel("Mean count")
plt.xlabel("Normalised asymmetry of intention")
plt.legend(loc='upper right', title="nRepair")

plt.show()

# Normalised asymmetry of intention for different conditions heatmap
repair0 = df[df["n_repair"] == 0]
repair1 = df[df["n_repair"] == 1]
repair2 = df[df["n_repair"] == 2]
repair3 = df[df["n_repair"] == 3]

data1 = pd.pivot_table(repair0, values='normalised asymmetry', index=['overlap'], columns='asymmetry')
data2 = pd.pivot_table(repair1, values='normalised asymmetry', index=['overlap'], columns='asymmetry')
data3 = pd.pivot_table(repair2, values='normalised asymmetry', index=['overlap'], columns='asymmetry')
data4 = pd.pivot_table(repair3, values='normalised asymmetry', index=['overlap'], columns='asymmetry')

fig, ax = plt.subplots(1, 4)
sns.heatmap(data1, ax=ax[0], vmin=0, vmax=1, cmap="Blues", cbar=False, annot=True)
sns.heatmap(data2, ax=ax[1], vmin=0, vmax=1, cmap="Blues", cbar=False, annot=True)
sns.heatmap(data3, ax=ax[2], vmin=0, vmax=1, cmap="Blues", cbar=False, annot=True)
sns.heatmap(data4, ax=ax[3], vmin=0, vmax=1, cmap="Blues", annot=True)

ax[0].set_title("nRepair = 0")
ax[1].set_title("nRepair = 1")
ax[2].set_title("nRepair = 2")
ax[3].set_title("nRepair = 3")
ax[0].set(ylabel="Degree of overlap", xlabel="")
ax[1].set(ylabel="", xlabel="Degree of asymmetry")
ax[2].set(ylabel="", xlabel="")
ax[3].set(ylabel="", xlabel="")
ax[0].yaxis.get_label().set_fontsize(14)
ax[1].xaxis.get_label().set_fontsize(14)

plt.show()

# Mean normalised asymmetry of intention
df["n_repair"] = df["n_repair"].astype(str)
sns.set_style("whitegrid")
sns.barplot(x="normalised asymmetry", y="n_repair", data=df, order=["0", "1", "2", "3"])
sns.stripplot(x="normalised asymmetry", y="n_repair", data=df, order=["0", "1", "2", "3"], color="gray")
plt.ylabel("Number of repair initiations", fontsize=18)
# plt.xlabel("Mean normalised asymmetry intention", fontsize=14)
plt.xlabel("Normalised asymmetry remaining in communicative intention beliefs", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
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

# Asymmetry of the entire network for different conversation states and different number of times that repair is
# initiated in a conversation
results['asymmetry_count'] = results['asymmetry_count'].astype(int)
#asymmetry_repair = results.groupby(["n_turns", "n_repair", "state", "conversation state", "n_nodes"], as_index=False)['asymmetry_count'].mean()
# sns.lineplot(x="state", y="asymmetry_count", hue="n_repair", data=asymmetry_repair)
#
# plt.title("Mean asymmetry for different conversation states with and without repair initiated")
# plt.ylabel("Asymmetry of network")
# plt.xlabel("Conversation state")

results_state = results[results["state"] != "listener initialisation"]
results_state = results_state[results_state["state"] != "end conversation"]
#g = sns.relplot(x="state", y="asymmetry_count", hue="n_repair", col="n_nodes", kind="line", data=results_state)
g = sns.FacetGrid(col="n_nodes", data=results_state, palette=sns.color_palette(colors))
g.map(sns.lineplot, "state", "asymmetry_count", "n_repair")

# iterate over axes of FacetGrid
for ax in g.axes.flat:
    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels, rotation=45, horizontalalignment='right')
    ax.legend(title="nRepair")
g.set_xlabels('Conversation state')
g.set_ylabels('Asymmetry of network')
axes = g.axes.flatten()
axes[0].set_title("8 nodes")
axes[1].set_title("10 nodes")
axes[2].set_title("12 nodes")
# g._legend.set_title("Number of times repair initiated")
# new_labels = ['0', '1', '2']
# for t, l in zip(g._legend.texts, new_labels): t.set_text(l)

plt.show()

# Version 2: split on interaction number

conditions = [(results_state['conversation state'] == 0) & (results_state['state'] == "initialisation"),
              (results_state['conversation state'] == 0) & (results_state['state'] == "listener update utterance"),
              (results_state['conversation state'] == 0) & (results_state['state'] == "speaker update network repair"),
              (results_state['conversation state'] == 0) & (results_state['state'] == "listener update solution"),
              (results_state['conversation state'] == 0) & (results_state['state'] == "speaker update network repair_1"),
              (results_state['conversation state'] == 0) & (results_state['state'] == "listener update solution_1"),
              (results_state['conversation state'] == 0) & (results_state['state'] == "speaker update network repair_2"),
              (results_state['conversation state'] == 0) & (results_state['state'] == "listener update solution_2"),

              (results_state['conversation state'] == 1) & (results_state['state'] == "initialisation"),
              (results_state['conversation state'] == 1) & (results_state['state'] == "listener update utterance"),
              (results_state['conversation state'] == 1) & (results_state['state'] == "speaker update network repair"),
              (results_state['conversation state'] == 1) & (results_state['state'] == "listener update solution"),
              (results_state['conversation state'] == 1) & (results_state['state'] == "speaker update network repair_1"),
              (results_state['conversation state'] == 1) & (results_state['state'] == "listener update solution_1"),
              (results_state['conversation state'] == 1) & (results_state['state'] == "speaker update network repair_2"),
              (results_state['conversation state'] == 1) & (results_state['state'] == "listener update solution_2"),

              (results_state['conversation state'] == 2) & (results_state['state'] == "initialisation"),
              (results_state['conversation state'] == 2) & (results_state['state'] == "listener update utterance"),
              (results_state['conversation state'] == 2) & (results_state['state'] == "speaker update network repair"),
              (results_state['conversation state'] == 2) & (results_state['state'] == "listener update solution"),
              (results_state['conversation state'] == 2) & (results_state['state'] == "speaker update network repair_1"),
              (results_state['conversation state'] == 2) & (results_state['state'] == "listener update solution_1"),
              (results_state['conversation state'] == 2) & (results_state['state'] == "speaker update network repair_2"),
              (results_state['conversation state'] == 2) & (results_state['state'] == "listener update solution_2")]

# values = ['initialisation', 'revision utterance', 'producer update repair', 'revision repair solution', 'producer update repair cl', 'revision clarification', 'producer update repair cl 2', 'revision clarification 2',
#           'initialisation 2', 'revision utterance 2', 'producer update repair 2', 'revision repair solution 2', 'producer update repair 2cl', 'revision clarification 2.1', 'producer update repair 2cl2', 'revision clarification 2.2',
#           'initialisation 3', 'revision utterance 3', 'producer update repair 3', 'revision repair solution 3', 'producer update repair 3cl', 'revision clarification 3.1', 'producer update repair 3cl2', 'revision clarification 3.2']
values = ['initialisation', 'revision utterance', 'producer update repair', 'revision repair solution',
          'producer update repair', 'revision repair solution', 'producer update repair', 'revision repair solution', 'initialisation 2', 'revision utterance 2', 'producer update repair 2', 'revision repair solution 2',
          'producer update repair 2', 'revision repair solution 2', 'producer update repair 2', 'revision repair solution 2', 'initialisation 3', 'revision utterance 3', 'producer update repair 3', 'revision repair solution 3',
          'producer update repair 3', 'revision repair solution 3', 'producer update repair 3', 'revision repair solution 3']
results_state['state conversation'] = np.select(conditions, values)

results_state = results_state.drop(results_state[results_state['state conversation'] == 'producer update repair'].index)
results_state = results_state.drop(results_state[results_state['state conversation'] == 'producer update repair 2'].index)
results_state = results_state.drop(results_state[results_state['state conversation'] == 'producer update repair 3'].index)

# Add all clarification repair together to one repair sequence

colors = ['#bdd7e7', '#6baed6', '#3182bd', '#bdd7e7', '#6baed6', '#3182bd', '#08519c', '#bdd7e7', '#6baed6', '#3182bd', '#08519c']

g = sns.FacetGrid(col="n_nodes", data=results_state, palette=colors)
results_state["normalised_asymmetry_network"] = results_state["asymmetry_count"]/results_state["n_nodes"]
results_state["normalised_asymmetry_network"] = results_state["normalised_asymmetry_network"].astype(float)

g.map(sns.lineplot, "state conversation", "normalised_asymmetry_network", "n_repair")

# iterate over axes of FacetGrid
for ax in g.axes.flat:
    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels, rotation=45, horizontalalignment='right', size=14)
    ax.legend(title="nRepair", fontsize=20)
    y_labels = ax.get_yticklabels()
    ax.set_yticklabels(y_labels, size=16)
g.set_xlabels('Conversation state', size=18)
g.set_ylabels('Mean normalised asymmetry of network', size=18)
axes = g.axes.flatten()
axes[0].set_title("8 nodes", size=18)
axes[1].set_title("10 nodes", size=18)
axes[2].set_title("12 nodes", size=18)

plt.show()

# Plot for all the nodes together
colors = ['#bdd7e7', '#6baed6', '#3182bd', '#08519c']
sns.set(font_scale=1.5)
g = sns.lineplot(x="state conversation", y="normalised_asymmetry_network", hue="n_repair", palette=colors, data=results_state, estimator=None)
plt.xlabel("Conversation state")
plt.ylabel("Mean normalised asymmetry of network")
plt.xticks(
    rotation=15,
    horizontalalignment='right',
    fontweight='light'
)

plt.show()

sns.set(font_scale=1)
# ----------------------------------------------------------------------------------------------------------------------
# -------------- The mean intention communicated for different amounts of nodes and edges in the network ---------------
# ----------------------------------------------------------------------------------------------------------------------

# Plot: number of nodes & number of edges of intention communicated (/asymmetry in intention?)
# df_intention = df.groupby(['n_nodes', 'amount_edges'])['intention_communicated'].value_counts(normalize=True)\
#     .reset_index(name='Counts')
# df_intention_true = df_intention[df_intention["intention_communicated"] == True]
# print(df_intention_true)

# # Compute chance levels for every node type
# nodes_8 = 0.5**(0.75 * 8)
# nodes_10 = 0.5**(0.75 * 10)
# nodes_12 = 0.5**(0.75 * 12)
#
# nodes_8_max = 0.5**(0.25 * 8)
# nodes_10_max = 0.5**(0.25 * 10)
# nodes_12_max = 0.5**(0.25 * 12)
#
# sns.barplot(x="n_nodes", y="intention_communicated", hue="amount_edges", data=df, ci="sd")
# #plt.title("Mean intention communicated for different amounts of nodes and edges in the network")
# plt.ylabel("Mean intention communicated")
# plt.xlabel("Number of nodes")
# plt.ylim(0, 1)
# plt.hlines(y=(nodes_8, nodes_10, nodes_12), xmin=(-0.4, 0.6, 1.6), xmax=(0.4, 1.4, 2.4), colors="black")
# plt.hlines(y=(nodes_8_max, nodes_10_max, nodes_12_max), xmin=(-0.4, 0.6, 1.6), xmax=(0.4, 1.4, 2.4), colors="black")
#
# plt.show()

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
# ------------------------------- Distribution of utterance and repair request length ----------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# utterance_length = results
# utterance_length['utterance length'] = utterance_length['utterance speaker'].str.len()
# utterance_length['repair length'] = utterance_length['repair request'].str.len()
# # pd.melt(utterance_length, id_vars=['asymmetry', 'overlap'], value_vars=['utterance length', 'repair length'],
# #          ignore_index=False, var_name='length_type', value_name='utt length')
# # print(utterance_length[:20])
#
# # also repair solutions as utterance length in distribution: only select utterance length that are not a repair solution
# utterance_length.loc[utterance_length['repair request'].notnull(), 'utterance length'] = np.nan
# fig, ax = plt.subplots(1, 2)
# sns.violinplot(x='asymmetry', y='utterance length', hue='overlap', ax=ax[0], data=utterance_length)
# ax[0].set_ylabel("Utterance length")
# ax[0].set_xlabel("Degree of asymmetry")
# ax[0].legend(title="Degree of overlap")
# sns.violinplot(x='asymmetry', y='repair length', hue='overlap', ax=ax[1], data=utterance_length)
# ax[1].set_xlabel("Degree of asymmetry")
# ax[1].set_ylabel("Repair initiation length")
# ax[1].legend(title="Degree of overlap")
#
# plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------- Similarity speaker matches asymmetry in intention? ----------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# Plot the different asymmetry levels of the intention when the conversation was ended because of max sim for degree of
# overlap and asymmetry
df['intention length'] = df['intention'].str.len()
df['normalised asymmetry'] = df['asymmetry_intention']/df['intention length']
df['normalised asymmetry'] = df['normalised asymmetry'].astype(float)

g = sns.catplot(x="asymmetry", y="normalised asymmetry", hue="overlap", col="n_nodes", kind="violin", data=df,
                legend_out=True)
# plt.title("Degree of \n overlap")
# plt.ylabel("Normalised asymmetry intention")
# plt.xlabel("Degree of asymmetry")
g.set_xlabels('Degree of asymmetry')
g.set_ylabels('Normalised asymmetry intention')
axes = g.axes.flatten()
axes[0].set_title("8 nodes")
axes[1].set_title("10 nodes")
axes[2].set_title("12 nodes")
g._legend.set_title("Degree of \n overlap")
new_labels = ['0%', '50%', '100%']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)

plt.show()

# Version 2: nodes put together
ax = sns.violinplot(x="asymmetry", y="normalised asymmetry", hue="overlap", data=df)
sns.stripplot(x="asymmetry", y="normalised asymmetry", hue="overlap", data=df, dodge=True)
ax.legend_.remove()
plt.ylabel("Normalised asymmetry intention")
plt.xlabel("Degree of asymmetry")
handles, labels = ax.get_legend_handles_labels()

# When creating the legend, only use the first three elements to effectively remove the last two
plt.legend(handles[0:3], labels[0:3], title="Degree of \n overlap")

plt.show()

# Perceived understanding vs actual understanding

# Actual understanding normalised asymmetry = 0-0.25
df["understanding"] = np.where(df["normalised asymmetry"] <= 0.25, 1, 0)
understanding = df[df["understanding"] == "Understanding"]

# Count data
understanding_count = df.groupby(['n_repair', 'asymmetry', 'overlap'])['understanding'].mean().reset_index\
    (name='understanding')
understanding_count["total"] = 1

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 15))

colors = ['#b35806', '#f1a340', '#fee0b6', '#d8daeb', '#998ec3', '#542788']

sns.set_palette(sns.color_palette(colors[:3]))
sns.barplot(x="total", y="n_repair", hue="asymmetry", data=understanding_count, ci=None)

sns.set_palette(sns.color_palette(colors[3:]))
sns.barplot(x="understanding", y="n_repair", hue="asymmetry", data=understanding_count, ci=None)

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True, title="Perceived understanding vs. Factual understanding \n "
                                                         "Degree of asymmetry")
ax.set(ylabel="nRepair",
       xlabel="Mean count")
sns.despine(left=True, bottom=True)

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Do beliefs stay coherent? ----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
colors = ['#bdd7e7', '#6baed6', '#3182bd', '#08519c']

# n_repair, conversation state, degree of overlap/asymmetry?
results_state['coherence listener'] = results_state['coherence listener'].astype(int)

g = sns.FacetGrid(col="asymmetry", data=results_state, palette=colors)
g.map(sns.lineplot, "state conversation", "coherence listener", "n_repair")

# iterate over axes of FacetGrid
for ax in g.axes.flat:
    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels, rotation=45, horizontalalignment='right')
    ax.legend(title="nRepair")
g.set_xlabels('Conversation state')
g.set_ylabels('Coherence listener network')
axes = g.axes.flatten()
axes[0].set_title("0% asymmetry")
axes[1].set_title("50% asymmetry")
axes[2].set_title("100% asymmetry")

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- Do inferred beliefs change a lot? ------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# i = 0
# for nodes in results_state['nodes listener']:
#     results_state.loc[i, 'inf nodes'] = [(x, y['truth_value']) for x, y in nodes if y['type'] == 'inf']
#     i +=1
#
# print(results_state['inf nodes'][:5])
#
# g = sns.FacetGrid(col="asymmetry", data=results_state, palette=sns.color_palette(colors))
# g.map(sns.lineplot, "state conversation", "coherence listener", "n_repair")
#
# # iterate over axes of FacetGrid
# for ax in g.axes.flat:
#     labels = ax.get_xticklabels() # get x labels
#     ax.set_xticklabels(labels, rotation=45, horizontalalignment='right')
#     ax.legend(title="nRepair")
# g.set_xlabels('Conversation state')
# g.set_ylabels('Coherence listener network')
# axes = g.axes.flatten()
# axes[0].set_title("0% asymmetry")
# axes[1].set_title("50% asymmetry")
# axes[2].set_title("100% asymmetry")
#
# plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------ How do the asymmetry of intention and network compare? --------------------------------
# ----------------------------------------------------------------------------------------------------------------------
results_state_0 = results_state[results_state['state conversation'] == "initialisation"]

results_state_0_new = pd.melt(results_state_0, id_vars=['n_nodes'], value_vars=['asymmetry_count',
                                                                                'asymmetry_intention'],
                              ignore_index=False, var_name='asymmetry_compare', value_name='count_asymmetry')

# Normalised asymmetry of intention vs normalised asymmetry of network
sns.barplot(x='asymmetry_compare', y='count_asymmetry', hue="n_nodes", data=results_state_0_new)
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- How often repair initiated? --------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# For different overlap and asymmetry levels
repair_counts = df.groupby(['asymmetry', 'overlap'])['n_repair'].value_counts(normalize=True).reset_index(name='Counts')
print(repair_counts)

colors = ['#bdd7e7', '#6baed6', '#3182bd', '#08519c']
sns.set(font_scale=1.5, style='whitegrid')
sns.set_palette(sns.color_palette(colors))

g = sns.FacetGrid(repair_counts, col="asymmetry")
g.map(sns.barplot, "n_repair", "Counts", "overlap", hue_order=[0, 50, 100], palette=colors)
g.add_legend(title="Degree of overlap")
# g.set_xlabels("nRepair")
# g.set_xlabels("Number of repair initiations")
g.axes[0, 1].set_xlabel('Number of repair initiations')
g.axes[0, 0].set_xlabel(' ')
g.axes[0, 2].set_xlabel(' ')

# g.axes[0,0].set_yticklabels(g.get_yticks(), size=16)
# g.axes[0,1].set_yticklabels(g.get_yticks(), size=16)
# g.axes[0,2].set_yticklabels(g.get_yticks(), size=16)
#
# g.axes[0,0].set_xticklabels(g.get_xticks(), size=16)
# g.axes[0,1].set_xticklabels(g.get_xticks(), size=16)
# g.axes[0,2].set_xticklabels(g.get_xticks(), size=16)

g.set_ylabels("Mean counts")
g.set(ylim=(0, 1))

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Asymmetry network at start vs end? -------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# For different times repair is initiated
df8_start = results_8.drop_duplicates(subset=["simulation_number"], keep='first')
df10_start = results_10.drop_duplicates(subset=["simulation_number"], keep='first')
df12_start = results_12.drop_duplicates(subset=["simulation_number"], keep='first')
df_start = pd.concat([df8_start, df10_start, df12_start])
df_start_asym = df_start["asymmetry_count"]
df_compare = df.join(df_start_asym)

#sns.barplot(x="asymmetry_count", y="asymmetry_count", data=df_compare)