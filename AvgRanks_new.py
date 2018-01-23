import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy.stats
import seaborn as sns


tableau20 =  [(166,206,227),(31,120,180),(178,223,138),(51,160,44),(251,154,153),(227,26,28),(202,178,214),(106,61,154),(253,191,111),(106,61,154),(255,255,153),(177,89,40)]


for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

#tableau = [tableau20[:6],tableau20[6:]]


grayscale = sns.dark_palette("white", n_colors=5)
grayscale2 = sns.dark_palette("white", n_colors=10)


base_data = pd.read_csv('acotsp_ranks.csv', delimiter=",")

base_subset = pd.read_csv('acotsp_stats.csv')

#print(base_data.head())
#print(base_data_info.head())

base_data_info = base_data.iloc[:,1:4]
base_data = base_data.iloc[:, 5:].as_matrix()


print(base_data)


race_parameters = ["DeleteWorst",
              "Hoeffding", "Bernstein", "Bayesian",
              "BlockingHoeffding", "BlockingBernstein", "BlockingBayesian",
              "FRaceT1", "FRaceT2",
              "AnovaRace", "BlockingAnovaRace"]




plotting_groups = [("DeleteWorst",
              "Hoeffding", "Bernstein", "Bayesian"),
                   ("BlockingHoeffding", "BlockingBernstein", "BlockingBayesian",
              "FRaceT1", "FRaceT2",
              "AnovaRace", "BlockingAnovaRace")]





race_labels =  {"DeleteWorst" : "DeleteWorst",
              "Hoeffding" : "Hoeffding",
               "Bernstein" : "Bernstein",
               "Bayesian" : "Bayesian",
              "BlockingHoeffding" : "BlockingHoeffding",
               "BlockingBernstein" : "BlockingBernstein",
               "BlockingBayesian" : "BlockingBayesian",
              "FRaceT1" : "FRaceT1",
               "FRaceT2" : "FRaceT2",
              "AnovaRace" : "AnovaRace",
               "BlockingAnovaRace" : "BlockingAnovaRace"}




no_cands = [16, 64, 256]

for cand in no_cands:

    #TRANSFORM DATA
    y_max = 50
    bin = 1
    df_long_full = pd.DataFrame()
    x = 0
    for race in race_parameters:
        x += 1
        to_plot_index = base_data_info[(base_data_info.race_type == race) & (base_data_info.no_cand == cand)].index
        y_max = int(np.max(base_subset[(base_subset.race_type == race) & (base_subset.no_cand == cand)]["no_task"]))
        print(race, y_max)
        # if x == 15:
        #     y_max = 85

        to_plot = base_data[to_plot_index, 0:y_max]
        #print(to_plot)
        plot_data = pd.DataFrame(to_plot).transpose()
        sequence = []
        order_seq = []
        race_type = []

        for i in range(int(np.shape(to_plot)[1] / bin)):
            race_type += [race]
            sequence += [str(i + 1)] * bin
            order_seq += [str(i + 1)]

        plot_data["instance"] = pd.Series(sequence, index=plot_data.index, dtype="unicode_")
        plot_data["race_type"] = pd.Series(race_type, index=plot_data.index, dtype="unicode_")
        df_long = pd.melt(plot_data, id_vars=['instance', 'race_type'], value_name='survivors')
        df_long_full = df_long_full.append(pd.melt(plot_data, id_vars=['instance', 'race_type'], value_name='survivors'))

    #print(df_long_full)


    #PLOT DATA
    g = 1
    for group in plotting_groups:
        sns.set_style("whitegrid")
        sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
        fig, ax = plt.subplots(figsize=(25, 10))
        to_plot = pd.DataFrame()
        for race in group:
            to_plot = to_plot.append(df_long_full[df_long_full.race_type == race])
        #print(to_plot)
        #print(to_plot.groupby("race_type"))
            #print(to_plot)
        #sns.pointplot(x='instance', y='survivors', order=order_seq, hue = 'race_type', data=to_plot, palette = grayscale, ci=95, ax=ax)
        #sns.pointplot(x='instance', y='survivors', order=order_seq, hue='race_type', data=to_plot, palette=sns.cubehelix_palette(n_colors=14, reverse=True ), ci=95, ax=ax,)
        sns.pointplot(x='instance',y='survivors', order=order_seq, hue='race_type', data=to_plot,
                      palette=tableau20, ax=ax, ci = 95, markers=".") #,linestyles= ["solid", "dotted"]*int(len(group)/2)

        # sns.pointplot(x='instance',y='survivors', order=order_seq, hue='race_type', data=to_plot,
        #               palette=tableau[g-1], ax=ax, ci = 95, markers=".") #,linestyles= ["solid", "dotted"]*int(len(group)/2)
        #


        plt.xlim(-1, 50)
        #plt.xlim(-1, 110)
        plt.xlim(-1, 59)

        max_avg_rank = cand/2 +1
        plt.ylim(0,max_avg_rank)

        x_ticks = np.arange(-1, 40, 10)
        x_ticks = np.arange(-1, 49, 10)##
        x_ticks_minor = np.arange(-1, 39, 1)
        x_ticks_minor = np.arange(-1, 59, 1)##

        y_ticks = np.arange(0, max_avg_rank, 1)
        #y_ticks = np.arange(0, 10, 1)##
        y_ticks_minor = np.arange(0, max_avg_rank, 0.5)
        #y_ticks_minor = np.arange(0, 9, 0.5)##

        ax.set_xticks(x_ticks)
        ax.set_xticks(x_ticks_minor, minor = True)
        ax.set_xticklabels(x_ticks + 1, fontsize = 34)

        ax.set_yticks(y_ticks)
        ax.set_yticks(y_ticks_minor, minor = True)
        ax.set_yticklabels(y_ticks, fontsize = 34)

        plt.ylabel("Mean Rank", fontsize = 34)
        plt.xlabel("Tasks", fontsize = 34)

        handles, labels = ax.get_legend_handles_labels()

        new_labs = []
        for lab in labels:
            new_labs.append(race_labels[lab])
        plt.legend(handles,new_labs,fontsize=24, markerscale = 3)


        file_name = "AvgRanks" + str(g) + "acotsp.pdf"

        plt.savefig(file_name, bbox_inches='tight')
        g += 1
        plt.show()
























#GET FINAL MEANS

#
# y_max = 100
# bin = 1
# df_long_full = pd.DataFrame()
# x = 0
# for race in race_parameters:
#     x += 1
#     to_plot_index = base_data_info[base_data_info.race_type == race].index
#     y_max = int(np.max(base_subset[(base_subset.race_type == race)]["no_ins"]))
#     print(race, y_max)
#
#     # if x == 15:
#     #     y_max = 85
#
#     to_plot = base_data[to_plot_index, 0:y_max]
#
#     print(to_plot[:,-1])
#     print(np.mean(to_plot[:,-1]))
#     print(np.std(to_plot[:, -1]))
#
#     plot_data = pd.DataFrame(to_plot).transpose()
#

