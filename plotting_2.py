import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
# colors = ['#bdd7e7', '#6baed6', '#3182bd', '#08519c']
# sns.set(font_scale=1.3, palette=sns.color_palette(colors), style="whitegrid")
# ----------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- Asymmetry network at start vs end? -------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# For different times repair is initiated
df8_start = results_8.drop_duplicates(subset=["simulation_number"], keep='first')
df10_start = results_10.drop_duplicates(subset=["simulation_number"], keep='first')
df12_start = results_12.drop_duplicates(subset=["simulation_number"], keep='first')
df_start = pd.concat([df8_start, df10_start, df12_start])
df_start["simulation_number_new"] = np.arange(df_start.shape[0])
df_start["normalised_asymmetry_start"] = df_start["asymmetry_count"]/df_start["n_nodes"]
decimals = 2
df_start["normalised_asymmetry_start"] = df_start["normalised_asymmetry_start"].apply(lambda x: round(x, decimals))
df_start["normalised_asymmetry_intention_start"] = df_start["asymmetry_intention"]/df_start["n_nodes"]
df_start["normalised_asymmetry_intention_start"] = df_start["normalised_asymmetry_intention_start"].apply(lambda x: round(x, decimals))
df["normalised_asymmetry"] = df["asymmetry_count"]/df["n_nodes"]
df["normalised_asymmetry_intention"] = df["asymmetry_intention"]/df["n_nodes"]
df["simulation_number_new"] = np.arange(df.shape[0])
df_start_asym = df_start[["normalised_asymmetry_start", "simulation_number_new", "normalised_asymmetry_intention_start"]]

#df_compare = df.join(df_start_asym.set_index('simulation_number_new'), on="simulation_number_new", rsuffix="_start")
df_compare = pd.merge(df, df_start_asym, on="simulation_number_new")

colors = ["#fee090", "#fdae61", "#f46d43", "#a50026", "#abd9e9", "#74add1", "#4575b4", "#313695"]
sns.set(font_scale=1.3, palette=sns.color_palette(colors), style="whitegrid")

# #sns.barplot(x="normalised_asymmetry_start", y="normalised_asymmetry", hue="n_repair", data=df_compare)
# ax = sns.factorplot(x="normalised_asymmetry_start", y="normalised_asymmetry", hue="n_repair", data=df_compare, colors=colors[:4])
# sns.stripplot(x="normalised_asymmetry_start", y="normalised_asymmetry", hue="n_repair", data=df_compare, jitter=True, dodge=True, palette=sns.color_palette(colors[4:]))
# #sns.relplot(x="normalised_asymmetry_start", y="normalised_asymmetry", hue="n_repair", data=df_compare, size="weight", palette=sns.color_palette(colors[4:]))
#
# ax.set(ylim=(-0.05,1))
#
# plt.ylabel("Remaining normalised asymmetry of networks")
# plt.xlabel("Normalised asymmetry of start networks")
#
# # # Get the handles and labels. For this example it'll be 2 tuples
# # # of length 4 each.
# # handles, labels = ax.get_legend_handles_labels()
# #
# # # When creating the legend, only use the first two elements
# # # to effectively remove the last two.
# # l = plt.legend(handles[0:4], labels[0:4], loc="upper left", title="n_repair")
#
# plt.legend(loc="upper left", title="n_repair", bbox_to_anchor=(1.05, 1))
# #plt.plot([0, 0.5, 1], [0, 0.5, 1], 'o:', color='blue')
#
# plt.show()
#
# # Same but for intention
# #sns.barplot(x="normalised_asymmetry_start", y="normalised_asymmetry", hue="n_repair", data=df_compare)
# ax = sns.factorplot(x="normalised_asymmetry_intention_start", y="normalised_asymmetry_intention", hue="n_repair", data=df_compare, colors=colors[:4])
# sns.stripplot(x="normalised_asymmetry_intention_start", y="normalised_asymmetry_intention", hue="n_repair", data=df_compare, jitter=True, dodge=True, palette=sns.color_palette(colors[4:]))
#
# ax.set(ylim=(-0.05,1))
#
# plt.ylabel("Remaining normalised asymmetry of intention")
# plt.xlabel("Normalised asymmetry of start intention")
# plt.legend(loc="upper left", title="n_repair", bbox_to_anchor=(1.05, 1))
# #plt.plot([0, 0.5, 1], [0, 0.5, 1], 'o:', color='blue')
#
# plt.show()
#
# # Is intention communicated when the asymmetry of the network is high/low?
# ax = sns.factorplot(x="normalised_asymmetry_start", y="normalised_asymmetry_intention", hue="n_repair", data=df_compare)
# sns.stripplot(x="normalised_asymmetry_start", y="normalised_asymmetry_intention", hue="n_repair", data=df_compare, jitter=True, dodge=True, palette=sns.color_palette(colors[4:]))
#
# ax.set(ylim=(-0.05,1))
#
# plt.ylabel("Remaining normalised asymmetry of intention")
# plt.xlabel("Normalised asymmetry of start network")
# plt.legend(loc="upper left", title="n_repair", bbox_to_anchor=(1.05, 1))
#
# plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------ perceived vs non perceived misunderstanding -------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# perceived cases where repair is initiated, non perceived cases where repair is not initiated and there is some
# asymmetry in the intention

#df_start_asym = df_start[["normalised_asymmetry_intention", "simulation_number_new"]]
#df_compare2 = df.join(df_start_asym.set_index('simulation_number_new'), on="simulation_number_new", rsuffix="_start")
df_compare["non_perceived"] = np.where((df_compare["normalised_asymmetry_intention_start"] != 0) & (df_compare["n_repair"] == 0), True, False)
counts = df_compare.groupby(['normalised_asymmetry_intention_start'])['non_perceived'].value_counts().reset_index(name='Counts')
grouped_df = counts.groupby(['normalised_asymmetry_intention_start', 'non_perceived']).agg({'Counts': 'sum'})
percents_df = grouped_df.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
percents_df.reset_index(inplace=True)
percents_df["total"] = 100
percents_df_perceived = percents_df[percents_df["non_perceived"] == False]

sns.barplot(x="normalised_asymmetry_intention_start", y="total", data=percents_df, ci=None, color='#bdd7e7')

g = sns.barplot(x="normalised_asymmetry_intention_start", y="Counts", data=percents_df_perceived, ci=None, color='#6baed6')

plt.legend(["not perceived", "perceived"])
g.axhline(y=50, color='black', linestyle='dashed')
plt.ylabel("Percentage")
plt.xlabel("Normalised asymmetry of intention at start")

# g = sns.barplot(x="normalised_asymmetry_intention_start", y="Counts", hue="non_perceived", data=percents_df)
#
# h, l = g.get_legend_handles_labels()
# g.legend(h, ["perceived", "not perceived"])
# #plt.legend(labels=["perceived", "not perceived"])
# plt.ylabel("Percentage")
# plt.xlabel("Normalised asymmetry of intention at start")

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------- Repair length for different asym levels ----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
results_10["simulation_number"] = results_10["simulation_number"] + 3600
results_12["simulation_number"] = results_12["simulation_number"] + 7200

results_new = pd.concat([results_8, results_10, results_12])
results_new["repair_length"] = results_new["repair request"].str.len()
results_new["start_asym"] = np.where(results_new["conversation state"] == "Start", results_new["asymmetry_intention"], np.nan)

repair_length = results_new.groupby(['simulation_number', 'n_repair'])['repair_length'].mean().reset_index(name='Mean')
df_start['simulation_number'] = df_start['simulation_number_new']
new = pd.merge(df_start, repair_length, on="simulation_number")
new2 = new[new['n_repair_y'] > 0]


ax = sns.factorplot(x="normalised_asymmetry_intention_start", y="Mean", hue="n_repair_y", data=new2)
sns.stripplot(x="normalised_asymmetry_intention_start", y="Mean", hue="n_repair_y", data=new2, jitter=True, dodge=True, palette=sns.color_palette(colors[4:]))

plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- Frequency for different asym levels ----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
#sns.countplot(x="normalised_asymmetry_intention_start", hue="n_repair", data=df_compare)

x,y = 'normalised_asymmetry_intention_start', 'n_repair'

(df_compare
.groupby(x)[y]
.value_counts(normalize=True)
.mul(100)
.rename('percent')
.reset_index()
.pipe((sns.catplot,'data'), x=x,y='percent',hue=y,kind='bar'))


plt.show()