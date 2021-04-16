import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import pandas as pd
import numpy as np
import matplotlib.ticker as tkr
import itertools

results_8 = pd.read_pickle("/Users/Jacqueline/Documents/Master_Thesis/Simulation/Finalrun/5/results_8.p")
results_10 = pd.read_pickle("/Users/Jacqueline/Documents/Master_Thesis/Simulation/Finalrun/5/results_10.p")
results_12 = pd.read_pickle("/Users/Jacqueline/Documents/Master_Thesis/Simulation/Finalrun/5/results_12.p")
#results_14 = pd.read_pickle("/Users/Jacqueline/Documents/Master_Thesis/Simulation/Finalrun/4/results_14.p")

results_10["simulation_number"] = results_10["simulation_number"] + 3600
results_12["simulation_number"] = results_12["simulation_number"] + 7200

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
pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.max_colwidth', None)

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
df_start["normalised_asymmetry_intention_start"] = df_start["asymmetry_intention"]/(df_start['intention'].str.len())
df_start["normalised_asymmetry_intention_start"] = df_start["normalised_asymmetry_intention_start"].apply(lambda x: round(x, decimals))
df_start['normalised asymmetry intention bins'] = pd.cut(df_start['normalised_asymmetry_intention_start'], bins=10)
df_start["normalised_asymmetry_bins"] = pd.cut(df_start['normalised_asymmetry_start'], bins=10)
df_start["network_asymmetry_start"] = df_start["asymmetry_count"] - df_start["asymmetry_intention"]
df_start["network_asymmetry_start_norm"] = df_start["network_asymmetry_start"]/(df_start["n_nodes"] - (df_start['intention'].str.len()))
df["network_asymmetry"] = df["asymmetry_count"] - df["asymmetry_intention"]
df["network_asymmetry_norm"] = df["network_asymmetry"]/(df["n_nodes"] - (df['intention'].str.len()))
df["normalised_asymmetry"] = df["asymmetry_count"]/df["n_nodes"]
df["normalised_asymmetry_intention"] = df["asymmetry_intention"]/(df['intention'].str.len())
df["simulation_number_new"] = np.arange(df.shape[0])
df_start_asym = df_start[["normalised_asymmetry_start", "simulation_number_new", "normalised_asymmetry_intention_start",
                          "normalised asymmetry intention bins", "normalised_asymmetry_bins",
                          "network_asymmetry_start_norm"]]

#df_compare = df.join(df_start_asym.set_index('simulation_number_new'), on="simulation_number_new", rsuffix="_start")
df_compare = pd.merge(df, df_start_asym, on="simulation_number_new")

#df_size = df_compare.groupby(["normalised_asymmetry_start", "normalised_asymmetry", "n_repair"]).size().reset_index(name='Counts')

#colors = ["#fee090", "#fdae61", "#f46d43", "#a50026", "#abd9e9", "#74add1", "#4575b4", "#313695"]
#colors = ["#abd9e9", "#74add1", "#4575b4", "#313695"]
#sns.set(font_scale=1.6, palette=sns.color_palette(colors), style="whitegrid")
palette_new = sns.color_palette("colorblind", 5)
palette = [(0.00392156862745098, 0.45098039215686275, 0.6980392156862745), (0.8705882352941177, 0.5607843137254902, 0.0196078431372549), (0.00784313725490196, 0.6196078431372549, 0.45098039215686275), (0.8, 0.47058823529411764, 0.7372549019607844)]
sns.set(font_scale=1.6, style="whitegrid")

#sns.barplot(x="normalised_asymmetry_start", y="normalised_asymmetry", hue="n_repair", data=df_compare)
# ax = sns.factorplot(x="normalised_asymmetry_bins", y="normalised_asymmetry", hue="n_repair", data=df_compare, colors=colors[:4])
# sns.stripplot(x="normalised_asymmetry_bins", y="normalised_asymmetry", hue="n_repair", data=df_compare, jitter=True, dodge=True, palette=sns.color_palette(colors[4:]))
# ax = sns.factorplot(x="normalised_asymmetry_start", y="normalised_asymmetry", hue="n_repair", data=df_compare, colors=colors[:4])
# sns.stripplot(x="normalised_asymmetry_start", y="normalised_asymmetry", hue="n_repair", data=df_compare, jitter=True, dodge=True, palette=sns.color_palette(colors[4:]))
#sns.scatterplot(x="normalised_asymmetry_start", y="normalised_asymmetry", hue="n_repair", data=df_size, size="Counts", palette=sns.color_palette(colors[4:]))
#sns.relplot(x="normalised_asymmetry_start", y="normalised_asymmetry", hue="n_repair", data=df_size, size="Counts", palette=sns.color_palette(colors[4:]), kind="strip")

df_compare['network_asymmetry_norm'] = df_compare['network_asymmetry_norm'].astype(float)
df_compare['network_asymmetry_start_norm'] = df_compare['network_asymmetry_start_norm'].astype(float)
#ax = sns.lmplot(x='normalised_asymmetry_start', y='normalised_asymmetry', hue='n_repair', data=df_compare)
#ax = sns.lmplot(x='network_asymmetry_start_norm', y='network_asymmetry_norm', hue='n_repair', data=df_compare)

g = sns.jointplot(x='network_asymmetry_start_norm', y='network_asymmetry_norm', hue='n_repair', palette=palette, data=df_compare)

ax = g.ax_joint
ax.set(ylim=(-0.05,1.05), xticks=np.arange(0.0, 1.1, 0.1).tolist(), xlim=(-0.01, 1.01))

g.set_axis_labels("Normalised asymmetry of networks minus the intention at start", "Remaining normalised asymmetry of networks minus the intention")
# plt.ylabel("Remaining normalised asymmetry of networks minus the intention")
# plt.xlabel("Normalised asymmetry of networks minus the intention at start")

# # Get the handles and labels. For this example it'll be 2 tuples
# # of length 4 each.
# handles, labels = ax.get_legend_handles_labels()
#
# # When creating the legend, only use the first two elements
# # to effectively remove the last two.
# l = plt.legend(handles[0:4], labels[0:4], loc="upper left", title="n_repair")

ax.legend(title="nRepair")
#plt.legend(loc="upper left", title="nRepair", bbox_to_anchor=(1.05, 1))
#plt.plot([0, 0.5, 1], [0, 0.5, 1], 'o:', color='blue')

#plt.tight_layout()
plt.show()

# Same but for intention
#sns.barplot(x="normalised_asymmetry_start", y="normalised_asymmetry", hue="n_repair", data=df_compare)
# ax = sns.factorplot(x="normalised asymmetry intention bins", y="normalised_asymmetry_intention", hue="n_repair", data=df_compare, colors=colors[:4])
# sns.stripplot(x="normalised asymmetry intention bins", y="normalised_asymmetry_intention", hue="n_repair", data=df_compare, jitter=True, dodge=True, palette=sns.color_palette(colors[4:]))

df_compare["normalised_asymmetry_intention"] = df_compare["normalised_asymmetry_intention"].astype(float)
g = sns.jointplot(x='normalised_asymmetry_intention_start', y='normalised_asymmetry_intention', hue='n_repair', palette=palette, data=df_compare)
ax = g.ax_joint
g.set_axis_labels("Normalised asymmetry of intention at start", "Remaining normalised asymmetry of intention")
ax.set(ylim=(-0.05,1.05), xticks=np.arange(0.0, 1.1, 0.1).tolist(), xlim=(-0.01, 1.01))

#ax = sns.lmplot(x='normalised_asymmetry_intention_start', y='normalised_asymmetry_intention', hue='n_repair', markers=["o", "x", "^", "*"], x_ci="sd", data=df_compare)

#ax.set(ylim=(-0.05,1.05), xticks=np.arange(0.0, 1.1, 0.1).tolist(), xlim=(-0.01, 1.01))

# plt.ylabel("Remaining normalised asymmetry of intention")
# plt.xlabel("Normalised asymmetry of intention at start")
# plt.legend(loc="upper left", title="nRepair", bbox_to_anchor=(1.05, 1))
#plt.plot([0, 0.5, 1], [0, 0.5, 1], 'o:', color='blue')
ax.legend(title="nRepair")
plt.show()

# # Is intention communicated when the asymmetry of the network is high/low?
# ax = sns.factorplot(x="normalised_asymmetry_start", y="normalised_asymmetry_intention", hue="n_repair", data=df_compare)
# sns.stripplot(x="normalised_asymmetry_start", y="normalised_asymmetry_intention", hue="n_repair", data=df_compare, jitter=True, dodge=True, palette=sns.color_palette(colors[4:]))
#
# ax.set(ylim=(-0.05,1.05))
#
# plt.ylabel("Remaining normalised asymmetry of intention")
# plt.xlabel("Normalised asymmetry of network at start")
# plt.legend(loc="upper left", title="nRepair", bbox_to_anchor=(1.05, 1))
#
# plt.show()
# # ----------------------------------------------------------------------------------------------------------------------
# # ------------------------------------ perceived vs non perceived misunderstanding -------------------------------------
# # ----------------------------------------------------------------------------------------------------------------------
# # perceived cases where repair is initiated, non perceived cases where repair is not initiated and there is some
# # asymmetry in the intention
#
# #df_start_asym = df_start[["normalised_asymmetry_intention", "simulation_number_new"]]
# #df_compare2 = df.join(df_start_asym.set_index('simulation_number_new'), on="simulation_number_new", rsuffix="_start")
# #df_compare["non_perceived"] = np.where((df_compare["normalised_asymmetry_intention_start"] != 0) & (df_compare["n_repair"] == 0), True, False)
# df_compare["non_perceived"] = np.where(df_compare["n_repair"] == 0, True, False)
# counts = df_compare.groupby(['normalised asymmetry intention bins'])['non_perceived'].value_counts().reset_index(name='Counts')
# grouped_df = counts.groupby(['normalised asymmetry intention bins', 'non_perceived']).agg({'Counts': 'sum'})
# percents_df = grouped_df.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
# percents_df.reset_index(inplace=True)
# percents_df["total"] = 100
# percents_df_perceived = percents_df[percents_df["non_perceived"] == False]

# sns.barplot(x="normalised asymmetry intention bins", y="total", data=percents_df, ci=None, color='#bdd7e7')
#
# g = sns.barplot(x="normalised asymmetry intention bins", y="Counts", data=percents_df_perceived, ci=None, color='#6baed6')
#
# plt.legend(["not perceived", "perceived"])
# g.axhline(y=50, color='black', linestyle='dashed')
# plt.ylabel("Percentage")
# plt.xlabel("Normalised asymmetry of intention at start")

# g = sns.barplot(x="normalised_asymmetry_intention_start", y="Counts", hue="non_perceived", data=percents_df)
#
# h, l = g.get_legend_handles_labels()
# g.legend(h, ["perceived", "not perceived"])
# #plt.legend(labels=["perceived", "not perceived"])
# plt.ylabel("Percentage")
# plt.xlabel("Normalised asymmetry of intention at start")

# plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------- Repair length for different asym levels ----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
results_10["simulation_number"] = results_10["simulation_number"] + 3600
results_12["simulation_number"] = results_12["simulation_number"] + 7200

results_new = pd.concat([results_8, results_10, results_12])
results_new["repair_length"] = results_new["repair request"].str.len()/results_new["n_nodes"]
results_new["start_asym"] = np.where(results_new["conversation state"] == "Start", results_new["asymmetry_intention"], np.nan)

repair_length = results_new.groupby(['simulation_number', 'n_repair'])['repair_length'].mean().reset_index(name='Mean')
df_start['simulation_number'] = df_start['simulation_number_new']
new = pd.merge(df_start, repair_length, on="simulation_number")
new2 = new[new['n_repair_y'] > 0]


# ax = sns.factorplot(x="normalised asymmetry intention bins", y="Mean", hue="n_repair_y", data=new2, palette=["#fdae61", "#f46d43", "#a50026"])
# sns.stripplot(x="normalised asymmetry intention bins", y="Mean", hue="n_repair_y", data=new2, jitter=True, dodge=True, palette=sns.color_palette(colors[5:]))
# ax = sns.lmplot(x="normalised_asymmetry_intention_start", y="Mean", hue="n_repair_y", data=new2, palette=["#abd9e9", "#4575b4", "#313695"])
#
# plt.ylabel("Mean normalised length of the repair initiations")
# plt.xlabel("Normalised asymmetry of intention at start")
# ax.set(ylim=(0, 1))
# plt.legend(loc="upper left", title="nRepair", bbox_to_anchor=(1.05, 1))

g = sns.jointplot(x="normalised_asymmetry_intention_start", y="Mean", hue="n_repair_y", data=new2, palette=palette[1:])
ax = g.ax_joint
g.set_axis_labels("Normalised asymmetry of intention at start", "Normalised length of the repair initiations")
ax.set(ylim=(0, 1), xticks=np.arange(0.0, 1.1, 0.1).tolist(), xlim=(-0.01, 1.01))
ax.legend(title="nRepair")

plt.show()
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------- Frequency for different asym levels ----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
#sns.countplot(x="normalised_asymmetry_intention_start", hue="n_repair", data=df_compare)

# x,y = 'normalised_asymmetry_intention_start', 'n_repair'
#
# (df_compare
# .groupby(x)[y]
# .value_counts(normalize=True)
# .mul(100)
# .rename('percent')
# .reset_index()
# .pipe((sns.catplot,'data'), x=x,y='percent',hue=y,kind='bar'))
#
# plt.ylabel("Percentage")
# plt.xlabel("Normalised asymmetry of intention at start")
# plt.legend(loc="upper right")
#
# plt.show()
#
# #df_start_asym = df_start[["normalised_asymmetry_intention", "simulation_number_new"]]
# #df_compare2 = df.join(df_start_asym.set_index('simulation_number_new'), on="simulation_number_new", rsuffix="_start")
# counts = df_compare.groupby(['normalised asymmetry intention bins'])['n_repair'].value_counts().reset_index(name='Counts')
# grouped_df = counts.groupby(['normalised asymmetry intention bins', 'n_repair']).agg({'Counts': 'sum'})
# #grouped_df.reset_index(inplace=True)
# percents_df = grouped_df.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
# percents_df.reset_index(inplace=True)
# #percents_df["total"] = 100
#
# print(percents_df)
#
# n_0 = percents_df[percents_df["n_repair"] == 0]
# n_0 = n_0.drop(columns=['n_repair'])
# n_0 = n_0.set_index("normalised asymmetry intention bins")
# n_1 = percents_df[percents_df["n_repair"] == 1]
# n_1 = n_1.drop(columns=['n_repair'])
# n_1 = n_1.set_index("normalised asymmetry intention bins")
# n_2 = percents_df[percents_df["n_repair"] == 2]
# n_2 = n_2.drop(columns=['n_repair'])
# n_2 = n_2.set_index("normalised asymmetry intention bins")
# n_3 = percents_df[percents_df["n_repair"] == 3]
# n_3 = n_3.drop(columns=['n_repair'])
# n_3 = n_3.set_index("normalised asymmetry intention bins")
#
# # n_0 = grouped_df[grouped_df["n_repair"] == 0]
# # n_0 = n_0.drop(columns=['n_repair'])
# # n_0 = n_0.set_index("normalised asymmetry intention bins")
# # n_1 = grouped_df[grouped_df["n_repair"] == 1]
# # n_1 = n_1.drop(columns=['n_repair'])
# # n_1 = n_1.set_index("normalised asymmetry intention bins")
# # n_2 = grouped_df[grouped_df["n_repair"] == 2]
# # n_2 = n_2.drop(columns=['n_repair'])
# # n_2 = n_2.set_index("normalised asymmetry intention bins")
# # n_3 = grouped_df[grouped_df["n_repair"] == 3]
# # n_3 = n_3.drop(columns=['n_repair'])
# # n_3 = n_3.set_index("normalised asymmetry intention bins")
#
# n_1 = n_1.add(n_0, fill_value=0)
# n_2 = n_2.add(n_1, fill_value=0)
# n_3 = n_3.add(n_2, fill_value=0)
#
# a = sns.barplot(x=n_3.index, y="Counts", data=n_3, color=colors[7])
# b = sns.barplot(x=n_2.index, y="Counts", data=n_2, color=colors[6])
# c = sns.barplot(x=n_1.index, y="Counts", data=n_1, color=colors[5])
# d = sns.barplot(x=n_0.index, y="Counts", data=n_0, color=colors[4])
#
# plt.ylabel("Percentage")
# plt.xlabel("Normalised asymmetry of intention at start")
# e = plt.Rectangle((0,0),1,1,fc=colors[4], edgecolor = 'none')
# f = plt.Rectangle((0,0),1,1,fc=colors[5],  edgecolor = 'none')
# g = plt.Rectangle((0,0),1,1,fc=colors[6], edgecolor = 'none')
# h = plt.Rectangle((0,0),1,1,fc=colors[7],  edgecolor = 'none')
# plt.legend(title="nRepair", handles=[e, f, g, h], labels=["0", "1", "2", "3"], bbox_to_anchor=(1.01, 1))
# #plt.legend(title="nRepair", bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
#
# plt.show()

# # ----------------------------------------------------------------------------------------------------------------------
# # ------------------------------------------- Plots for robustness check -----------------------------------------------
# # ----------------------------------------------------------------------------------------------------------------------
# df["normalised_asymmetry_intention"] = df["normalised_asymmetry_intention"].astype(float)
# df["normalised_asymmetry"] = df["normalised_asymmetry"].astype(float)
# total_intention = pd.pivot_table(df, values="normalised_asymmetry_intention", index=['n_nodes'], columns='amount_edges')
# total_network = pd.pivot_table(df, values="normalised_asymmetry", index=['n_nodes'], columns='amount_edges')
#
# fig, ax = plt.subplots(1, 2)
# sns.heatmap(total_network, ax=ax[0], vmin=0, vmax=1, cmap="Blues", cbar=False, annot=True)
# sns.heatmap(total_intention, ax=ax[1], vmin=0, vmax=1, cmap="Blues", annot=True)
#
# ax[0].set_title("Network")
# ax[1].set_title("Intention")
# fig.text(0.5, 0.01, 'Amount of edges', ha='center', va='center')
# fig.text(0.08, 0.5, 'Number of nodes', ha='center', va='center', rotation='vertical')
# ax[0].set(ylabel="", xlabel="")
# ax[1].set(ylabel="", xlabel="")
#
# plt.show()

# # Normalised asymmetry of intention for different conditions heatmap
# df["normalised_asymmetry_intention"] = df["normalised_asymmetry_intention"].astype(float)
# df["normalised_asymmetry"] = df["normalised_asymmetry"].astype(float)
#
# repair0 = df[df["n_repair"] == 0]
# repair1 = df[df["n_repair"] == 1]
# repair2 = df[df["n_repair"] == 2]
# repair3 = df[df["n_repair"] == 3]
#
# data1 = pd.pivot_table(repair0, values="normalised_asymmetry_intention", index=['n_nodes'], columns='amount_edges')
# data2 = pd.pivot_table(repair1, values="normalised_asymmetry_intention", index=['n_nodes'], columns='amount_edges')
# data3 = pd.pivot_table(repair2, values="normalised_asymmetry_intention", index=['n_nodes'], columns='amount_edges')
# data4 = pd.pivot_table(repair3, values="normalised_asymmetry_intention", index=['n_nodes'], columns='amount_edges')
#
# fig, ax = plt.subplots(1, 4)
# sns.heatmap(data1, ax=ax[0], vmin=0, vmax=1, cmap="Blues", cbar=False, annot=True)
# sns.heatmap(data2, ax=ax[1], vmin=0, vmax=1, cmap="Blues", cbar=False, annot=True, yticklabels=False)
# sns.heatmap(data3, ax=ax[2], vmin=0, vmax=1, cmap="Blues", cbar=False, annot=True, yticklabels=False)
# sns.heatmap(data4, ax=ax[3], vmin=0, vmax=1, cmap="Blues", annot=True, yticklabels=False)
#
# ax[0].set_title("nRepair = 0")
# ax[1].set_title("nRepair = 1")
# ax[2].set_title("nRepair = 2")
# ax[3].set_title("nRepair = 3")
# fig.text(0.5, 0.01, 'Amount of edges', ha='center', va='center')
# fig.text(0.08, 0.5, 'Number of nodes', ha='center', va='center', rotation='vertical')
# ax[0].set(ylabel="", xlabel="")
# ax[1].set(ylabel="", xlabel="")
# ax[2].set(ylabel="", xlabel="")
# ax[3].set(ylabel="", xlabel="")
#
# plt.show()
#
# # Normalised asymmetry of network for different conditions heatmap
# data1 = pd.pivot_table(repair0, values="normalised_asymmetry", index=['n_nodes'], columns='amount_edges')
# data2 = pd.pivot_table(repair1, values="normalised_asymmetry", index=['n_nodes'], columns='amount_edges')
# data3 = pd.pivot_table(repair2, values="normalised_asymmetry", index=['n_nodes'], columns='amount_edges')
# data4 = pd.pivot_table(repair3, values="normalised_asymmetry", index=['n_nodes'], columns='amount_edges')
#
# fig, ax = plt.subplots(1, 4)
# sns.heatmap(data1, ax=ax[0], vmin=0, vmax=1, cmap="Blues", cbar=False, annot=True)
# sns.heatmap(data2, ax=ax[1], vmin=0, vmax=1, cmap="Blues", cbar=False, annot=True, yticklabels=False)
# sns.heatmap(data3, ax=ax[2], vmin=0, vmax=1, cmap="Blues", cbar=False, annot=True, yticklabels=False)
# sns.heatmap(data4, ax=ax[3], vmin=0, vmax=1, cmap="Blues", annot=True, yticklabels=False)
#
# ax[0].set_title("nRepair = 0")
# ax[1].set_title("nRepair = 1")
# ax[2].set_title("nRepair = 2")
# ax[3].set_title("nRepair = 3")
# fig.text(0.5, 0.01, 'Amount of edges', ha='center', va='center')
# fig.text(0.08, 0.5, 'Number of nodes', ha='center', va='center', rotation='vertical')
# ax[0].set(ylabel="", xlabel="")
# ax[1].set(ylabel="", xlabel="")
# ax[2].set(ylabel="", xlabel="")
# ax[3].set(ylabel="", xlabel="")
#
# plt.show()
#
# # ----------------------------------------------------------------------------------------------------------------------
# # --------------------------------------- Confirmation vs disconfirmation? ---------------------------------------------
# # ----------------------------------------------------------------------------------------------------------------------
# df_start_asym = df_start[["normalised_asymmetry_start", "simulation_number", "normalised_asymmetry_intention_start",
#                           "normalised asymmetry intention bins", "normalised_asymmetry_bins"]]
# results2 = pd.merge(results, df_start_asym, on="simulation_number")
#
# test = results2.groupby(["normalised asymmetry intention bins", "n_repair"])["confirmation?"].value_counts().reset_index(name='Counts')
# grouped_df = test.groupby(['normalised asymmetry intention bins', 'n_repair', 'confirmation?']).agg({'Counts': 'sum'})
# percents_df = grouped_df.groupby(level=[0,1]).apply(lambda x: 100 * x / float(x.sum()))
# percents_df.reset_index(inplace=True)
# percents_df["total"] = 100
# percents_df_confirmation = percents_df[percents_df["confirmation?"] == True]
#
# sns.barplot(x="normalised asymmetry intention bins", y="Counts", hue="n_repair", data=percents_df_confirmation, palette=sns.color_palette(colors[5:]))
# plt.ylabel("Percentage confirmation")
# plt.xlabel("Normalised asymmetry of intention at start")
#
# plt.show()

# # ----------------------------------------------------------------------------------------------------------------------
# # ---------------------- Intention communicated without producer communicating it directly? ----------------------------
# # ----------------------------------------------------------------------------------------------------------------------
# # Compare producer's utterances with intention
# results.dropna(subset=["utterance speaker"], inplace=True)
# results = results[results['utterance speaker'].map(lambda d: len(d)) > 0]
# results["utterance speaker"] = results["utterance speaker"].apply(lambda x: [a_tuple[0] for a_tuple in x])
# grouped_df = results.groupby("simulation_number")
# grouped_lists = grouped_df["utterance speaker"].apply(list)
# grouped_lists = grouped_lists.reset_index()
# grouped_lists["utterance speaker"] = grouped_lists["utterance speaker"].apply(lambda x: list(itertools.chain(*x)))
# final = pd.merge(grouped_lists, df, on=["simulation_number"])
# final["intention"] = final["intention"].apply(set)
# final["utterance speaker_x"] = final["utterance speaker_x"].apply(set)
# result = [x.issubset(y) for x, y in zip(final["intention"], final["utterance speaker_x"])]
# final["utterances_intention"] = result
# final_new = pd.merge(final, df_start, on=["simulation_number"])
#
# sns.countplot(x="n_repair_x", hue="utterances_intention", data=final_new)
# plt.show()
#
# final_new_2 = final_new.groupby(["n_repair_x"])["utterances_intention"].value_counts().reset_index(name='Counts')
# grouped_df = final_new_2.groupby(['n_repair_x', 'utterances_intention']).agg({'Counts': 'sum'})
# percents_df = grouped_df.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
# percents_df.reset_index(inplace=True)
# percents_df["total"] = 100
# percents_not_communicated = percents_df[percents_df["utterances_intention"] == False]
#
# print(percents_not_communicated)
#
# a = plt.Rectangle((0,0),1,1,fc='#6baed6', edgecolor = 'none')
# b = plt.Rectangle((0,0),1,1,fc='#bdd7e7',  edgecolor = 'none')
# sns.barplot(x="n_repair_x", y="total", data=percents_df, color='#6baed6')
# sns.barplot(x="n_repair_x", y="Counts", data=percents_not_communicated, ci=None, color='#bdd7e7')
# #
# # g = sns.barplot(x="normalised asymmetry intention bins", y="Counts", data=percents_df_perceived, ci=None, color='#6baed6')
#
# plt.xlabel("nRepair")
# plt.ylabel("Percentage")
# plt.legend(labels=["True", "False"], handles=[a,b], title="Complete intention uttered", loc="upper right")
#
# plt.show()
#
# colors = ["#abd9e9", "#74add1", "#4575b4", "#313695"]
# #"#fee090", "#fdae61", "#f46d43", "#a50026"
# results["normalised_asymmetry_intention"] = results["asymmetry_intention"]/(results['intention'].str.len())
# results["normalised_asymmetry_intention"] = results["normalised_asymmetry_intention"].astype(float)
# ax = sns.lineplot(x="n_turns", y="normalised_asymmetry_intention", hue="n_repair", palette=colors, data=results)
# plt.ylabel("Normalised asymmetry of the intention")
# plt.xlabel("Turns in a conversation")
# ax.axes.get_xaxis().set_visible(False)
#
# plt.show()
#
# colors = ["#abd9e9", "#74add1", "#4575b4", "#313695"]
# #"#fee090", "#fdae61", "#f46d43", "#a50026"
# results["normalised_asymmetry"] = results["asymmetry_count"]/(results['n_nodes'])
# results["normalised_asymmetry"] = results["normalised_asymmetry"].astype(float)
# ax = sns.lineplot(x="n_turns", y="normalised_asymmetry", hue="n_repair", palette=colors, data=results)
# plt.ylabel("Normalised asymmetry of the network")
# plt.xlabel("Turns in a conversation")
# ax.axes.get_xaxis().set_visible(False)
#
# plt.show()

# # ----------------------------------------------------------------------------------------------------------------------
# # -------------------------------------- Frequency of starting asymmetry levels ----------------------------------------
# # ----------------------------------------------------------------------------------------------------------------------
#
# colors = ["#74add1"]
# # rc={'xtick.labelsize': 12}
# # sns.set_context(rc=rc)
# # fig_dims = (8, 4)
# #fig, ax = plt.subplots(1,2)
# sns.set(font_scale=2, palette=sns.color_palette(colors), style="whitegrid")
#
# sns.countplot(x="normalised asymmetry intention bins", data=df_compare, palette=colors)
# plt.xlabel("Normalised asymmetry of intention at start")
# plt.ylabel("Frequency count")
# plt.ylim(0, 2500)
# #plt.legend(title="nRepair", bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
# plt.show()
#
# sns.countplot(x="normalised_asymmetry_bins", data=df_compare, palette=colors)
# plt.xlabel("Normalised asymmetry of networks at start")
# plt.ylabel("Frequency count")
# plt.ylim(0, 2500)
# #plt.legend(title="nRepair", loc="upper right")
# plt.show()

# # ----------------------------------------------------------------------------------------------------------------------
# # ------------------------------------------------ Utterance length ----------------------------------------------------
# # ----------------------------------------------------------------------------------------------------------------------
#
# utterance_results = results[results["repair request"].isna()]
# utterances = utterance_results["utterance speaker"].dropna()
# utterances["length"] = utterances.str.len()
#
# print(utterances[:100])
#
# sns.countplot(y="length", data=utterances)
# plt.show()
