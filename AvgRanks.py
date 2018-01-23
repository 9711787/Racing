import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy.stats
import seaborn as sns

#
# tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
#              (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
#              (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
#              (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
#              (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

#
# tableau20 = [(27,158,119),(217,95,2),(117,112,179),(231,41,138), (102,166,30),(230,171,2),(166,118,29),(102,102,102)]
# tableau20 = [(228,26,28),(55,126,184),(77,175,74),(152,78,163),(255,127,0),(231,41,138),(166,86,40),(247,129,191)]
#
#tableau20 =  [(166,206,227),(31,120,180),(178,223,138),(51,160,44),(251,154,153),(227,26,28),(253,191,111),(255,127,0),(202,178,214),(106,61,154),(255,255,153),(177,89,40)]

tableau20 =  [(166,206,227),(31,120,180),(178,223,138),(51,160,44),(251,154,153),(227,26,28),(202,178,214),(106,61,154),(253,191,111),(106,61,154),(255,255,153),(177,89,40)]


for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

#tableau = [tableau20[:6],tableau20[6:]]


grayscale = sns.dark_palette("white", n_colors=5)
grayscale2 = sns.dark_palette("white", n_colors=10)




base_data_info = pd.read_csv('AvgRankBASE_info.csv') # original data
base_data = np.loadtxt('AvgRankBASE.csv', delimiter=",")
#
# # base_data_info = pd.read_csv('AvgRank0.25Diff_info.csv')
# # base_data = np.loadtxt('AvgRank0.25Diff.csv', delimiter=",")
#
# # base_data_info = pd.read_csv('AvgRank1Diff_info.csv')
# # base_data = np.loadtxt('AvgRank1Diff.csv', delimiter=",")
#
# # base_data_info = pd.read_csv('AvgRank1Diff_info.csv')
# # base_data = np.loadtxt('AvgRank1Diff.csv', delimiter=",")
# #
# # base_data_info = pd.read_csv('Diff0.125AvgRank_info.csv')
# # base_data = np.loadtxt('Diff0.125AvgRank.csv', delimiter=",")
# #
# base_data_info = pd.read_csv('AvgRank00625DIFF_info.csv')
# base_data = np.loadtxt('AvgRank00625DIFF.csv', delimiter=",")
# #
# base_data_info = pd.read_csv('AvgRank4DIFF_info.csv')
# base_data = np.loadtxt('AvgRank4DIFF.csv', delimiter=",")
#
# # base_data_info = pd.read_csv('AvgRank010DELTA_info.csv')
# # base_data = np.loadtxt('AvgRank010DELTA.csv', delimiter=",")
#
# # base_data_info = pd.read_csv('AvgRank1VARACCROSS_info.csv')
# # base_data = np.loadtxt('AvgRank1VARACCROSS.csv', delimiter=",")
#
# base_data_info = pd.read_csv('AvgRank01VARACCROSS_info.csv')
# base_data = np.loadtxt('AvgRank01VARACCROSS.csv', delimiter=",")
# # #
# base_data_info = pd.read_csv('AvgRank100VARACCROSS_info.csv')
# base_data = np.loadtxt('AvgRank100VARACCROSS.csv', delimiter=",")
#
# base_data_info = pd.read_csv('AvgRank01VARWITHIN_info.csv')
# base_data = np.loadtxt('AvgRank01VARWITHIN.csv', delimiter=",")
# #
# # # base_data_info = pd.read_csv('AvgRank10VARWITHIN_info.csv')
# # # base_data = np.loadtxt('AvgRank10VARWITHIN.csv', delimiter=",")
# #
# # base_data_info = pd.read_csv('AvgRank100VARWITHIN_info.csv')
# # base_data = np.loadtxt('AvgRank100VARWITHIN.csv', delimiter=",")
# #
# base_data_info = pd.read_csv('AvgRankExponentialZeroMean_info.csv')
# base_data = np.loadtxt('AvgRankExponentialZeroMean.csv', delimiter=",")
# #
# base_data_info = pd.read_csv('AvgRankExponentialZeroMedian_info.csv')
# base_data = np.loadtxt('AvgRankExponentialZeroMedian.csv', delimiter=",")
# #
# base_data_info = pd.read_csv('AvgRank90ALPHA_info.csv')
# base_data = np.loadtxt('AvgRank90ALPHA.csv', delimiter=",")
# #
# base_data_info = pd.read_csv('AvgRank99ALPHA_info.csv')
# base_data = np.loadtxt('AvgRank99ALPHA.csv', delimiter=",")
# #
# # base_data_info = pd.read_csv('AvgRankRealisticMC_info.csv')
# # base_data = np.loadtxt('AvgRankRealisticMC.csv', delimiter=",")
# #
# base_data_info = pd.read_csv('AvgRank2Start_info.csv')
# base_data = np.loadtxt('AvgRank2Start.csv', delimiter=",")
# #
# base_data_info = pd.read_csv('AvgRank10START_info.csv')
# base_data = np.loadtxt('AvgRank10START.csv', delimiter=",")

#base_subset = pd.read_csv('BASECase27072017.csv')# Original
# base_subset = pd.read_csv('BASECase30072017.csv')
# # #base_subset = pd.read_csv('BASECaseDIFF.csv')
# # #base_subset = pd.read_csv('BASECase0125DIFF.csv')
# base_subset = pd.read_csv('BASECase00625DIFF.csv')
# base_subset = pd.read_csv('BASECase4DIFF.csv')
# # #base_subset = pd.read_csv('BASECase010DELTA.csv')
# # #base_subset = pd.read_csv('BASECase1VARACCROSS.csv')
# base_subset = pd.read_csv('BASECase01VARACCROSS.csv')
# base_subset = pd.read_csv('BASECase100VARACCROSS.csv')
# base_subset = pd.read_csv('BASECase01VARWITHIN.csv')
# # #base_subset = pd.read_csv('BASECase10VARWITHIN.csv')
# #base_subset = pd.read_csv('BASECase100VARWITHIN.csv')
# base_subset = pd.read_csv('BASECaseExponentialZeroMean.csv')
# base_subset = pd.read_csv('BASECaseExponentialZeroMedian.csv')
# base_subset = pd.read_csv('BASECase90ALPHA.csv')
# base_subset = pd.read_csv('BASECase99ALPHA.csv')
# #base_subset = pd.read_csv('BASECaseRealisticMC.csv')
# base_subset = pd.read_csv('BASECase2Start.csv')
# base_subset = pd.read_csv('BASECase10START.csv')
base_subset = pd.read_csv('BASECaseXXX.csv')


race_parameters =  ["Hoeffding4Sigma","Bernstein4Sigma","DeleteWorstImprovedRho0.10","DeleteWorstImprovedRho0.20",
                                 "BlockingBayesFastGamma0",
                                 "BlockingBayesFastGamma1", "BayesImprovedGamma0",
                                 "BayesImprovedGamma1", "FriedmanT1Fast","FriedmanT2Fast",
                                 "BlockingHoeffdingFast4Sigma", "BlockingBernsteinFast4Sigma", "ANOVAFisherLSD",
                                 "pairedANOVAFisherLSD"]

race_parameters =  ["Hoeffding4Sigma","Bernstein4Sigma","DeleteWorstImprovedRho0.10","DeleteWorstImprovedRho0.20",
                                 "BlockingBayesFastGamma0",
                                 "BlockingBayesFastGamma1", "BayesImprovedGamma0",
                                 "BayesImprovedGamma1", "FriedmanT1Fast","FriedmanT2Fast",
                                 "BlockingHoeffdingFast4Sigma", "BlockingBernsteinFast4Sigma", "ANOVAFisherLSD",
                                 "pairedANOVAFisherLSD"]# , "Hoeffding4Sigma"]

# plotting_groups = [("DeleteWorstImprovedRho0.10", "DeleteWorstImprovedRho0.20"),
#                     ("Hoeffding4Sigma","BlockingHoeffdingFast4Sigma","Bernstein4Sigma", "BlockingBernsteinFast4Sigma"),
#                    ("BayesImprovedGamma0","BlockingBayesFastGamma0","BayesImprovedGamma1","BlockingBayesFastGamma1"),
#                     ("FriedmanT1Fast","FriedmanT2Fast","ANOVAFisherLSD","pairedANOVAFisherLSD")]
#
# plotting_groups = [["DeleteWorstImprovedRho0.10", "DeleteWorstImprovedRho0.20",
#                     "Hoeffding4Sigma","BlockingHoeffdingFast4Sigma","Bernstein4Sigma", "BlockingBernsteinFast4Sigma",
#                    "BayesImprovedGamma0","BlockingBayesFastGamma0","BayesImprovedGamma1","BlockingBayesFastGamma1",
#                     "FriedmanT1Fast","FriedmanT2Fast","ANOVAFisherLSD","pairedANOVAFisherLSD"]]
#
# plotting_groups = [("DeleteWorstImprovedRho0.10", "DeleteWorstImprovedRho0.20",
#                     "Hoeffding4Sigma","BlockingHoeffdingFast4Sigma","Bernstein4Sigma", "BlockingBernsteinFast4Sigma"),
#                    ("BayesImprovedGamma0","BlockingBayesFastGamma0","BayesImprovedGamma1","BlockingBayesFastGamma1",
#                     "FriedmanT1Fast","FriedmanT2Fast","ANOVAFisherLSD","pairedANOVAFisherLSD")]

plotting_groups = [("DeleteWorstImprovedRho0.10", "DeleteWorstImprovedRho0.20", "BayesImprovedGamma0","BlockingBayesFastGamma0","BayesImprovedGamma1","BlockingBayesFastGamma1"),
                   ("Hoeffding4Sigma","BlockingHoeffdingFast4Sigma","Bernstein4Sigma", "BlockingBernsteinFast4Sigma",
                    "FriedmanT1Fast","FriedmanT2Fast","ANOVAFisherLSD","pairedANOVAFisherLSD")]

plotting_groups = [("Hoeffding4Sigma","BlockingHoeffdingFast4Sigma","Bernstein4Sigma", "BlockingBernsteinFast4Sigma",
                    "BayesImprovedGamma0","BlockingBayesFastGamma0","BayesImprovedGamma1", "BlockingBayesFastGamma1"),
                   ("DeleteWorstImprovedRho0.10", "DeleteWorstImprovedRho0.20",
                    "FriedmanT1Fast","FriedmanT2Fast","ANOVAFisherLSD","pairedANOVAFisherLSD")]


race_labels =  {"Hoeffding4Sigma" : "Hoeffding" ,
                "Bernstein4Sigma" : "Bernstein" ,
                "DeleteWorstImprovedRho0.10" : "Del. Worst 10%" ,
                "DeleteWorstImprovedRho0.20" : "Del. Worst 20%",
                "BlockingBayesFastGamma0" : "Bl. Bayes γ = 0",
                "BlockingBayesFastGamma1" : "Bl. Bayes γ = 1",
                "BayesImprovedGamma0" : "Bayes γ = 0",
                "BayesImprovedGamma1" : "Bayes γ = 1",
                "FriedmanT1Fast" : "Friedman T1",
                "FriedmanT2Fast" : "Friedman T2",
                "BlockingHoeffdingFast4Sigma" : "Bl. Hoeffding",
                "BlockingBernsteinFast4Sigma" : "Bl. Bernstein",
                "ANOVAFisherLSD" : "ANOVA" ,
                "pairedANOVAFisherLSD" : "Bl. ANOVA"}


race_labels =  {"Hoeffding4Sigma" : "Hoeffding" ,
                "Bernstein4Sigma" : "Bernstein" ,
                "DeleteWorstImprovedRho0.10" : "Remove ρ = 0.1",
                "DeleteWorstImprovedRho0.20" : "Remove ρ = 0.2",
                "BlockingBayesFastGamma0" : "Bl. Bayes γ = 0",
                "BlockingBayesFastGamma1" : "Bl. Bayes γ = 1",
                "BayesImprovedGamma0" : "Bayes γ = 0",
                "BayesImprovedGamma1" : "Bayes γ = 1",
                "FriedmanT1Fast" : "Friedman T1",
                "FriedmanT2Fast" : "Friedman T2",
                "BlockingHoeffdingFast4Sigma" : "Bl. Hoeffding",
                "BlockingBernsteinFast4Sigma" : "Bl. Bernstein",
                "ANOVAFisherLSD" : "ANOVA" ,
                "pairedANOVAFisherLSD" : "Bl. ANOVA"}


#TRANSFORM DATA
y_max = 100
bin = 1
df_long_full = pd.DataFrame()
x = 0
for race in race_parameters:
    x += 1
    to_plot_index = base_data_info[base_data_info.race_type == race].index
    y_max = int(np.max(base_subset[(base_subset.race_type == race)]["no_task"]))
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


    plt.xlim(-1, 100)
    #plt.xlim(-1, 110)
    plt.xlim(-1, 109)

    plt.ylim(0, 9)

    x_ticks = np.arange(-1, 90, 10)
    x_ticks = np.arange(-1, 109, 10)##
    x_ticks_minor = np.arange(-1, 89, 1)
    x_ticks_minor = np.arange(-1, 109, 1)##

    y_ticks = np.arange(0, 9, 1)
    #y_ticks = np.arange(0, 10, 1)##
    y_ticks_minor = np.arange(0, 9, 0.5)
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

    #file_name = "BASEAvgRanks"+str(g)+".pdf"
    # file_name = "BASEAvgRanks" + str(g) + "4DIFF.pdf"
    # #file_name = "BASEAvgRanks" + str(g) + "010DELTA.pdf"
    # file_name = "BASEAvgRanks" + str(g) + "1VARACCROSS.pdf"
    #file_name = "BASEAvgRanks" + str(g) + "01VARACCROSS.pdf"
    # file_name = "BASEAvgRanks" + str(g) + "100VARWITHIN.pdf"
    # file_name = "BASEAvgRanks" + str(g) + "ExponentialZeroMean.pdf"
    #file_name = "BASEAvgRanks" + str(g) + "99ALPHA.pdf"
    #file_name = "BASEAvgRanks" + str(g) + "RealisticMC.pdf"
    #file_name = "BASEAvgRanks" + str(g) + "2Start.pdf"
    #file_name = "BASEAvgRanks" + str(g) + "10Start.pdf"

    file_name = "BASEAvgRanksGroup"+str(g)+"Improved.pdf"
    file_name = "BASEAvgRanksGroup" + str(g) + "Improved00625DIFF.pdf"
    file_name = "BASEAvgRanksGroup" + str(g) + "Improved4DIFF.pdf"
    # #file_name = "BASEAvgRanksGroup" + str(g) + "Improved010DELTA.pdf"
    # file_name = "BASEAvgRanksGroup" + str(g) + "Improved1VARACCROSS.pdf"
    file_name = "BASEAvgRanksGroup" + str(g) + "Improved01VARACCROSS.pdf"
    file_name = "BASEAvgRanksGroup" + str(g) + "Improved100VARWITHIN.pdf"
    # file_name = "BASEAvgRanksGroup" + str(g) + "ImprovedExponentialZeroMean.pdf"
    file_name = "BASEAvgRanksGroup" + str(g) + "Improved99ALPHA.pdf"
    file_name = "BASEAvgRanksGroup" + str(g) + "ImprovedRealisticMC.pdf"
    file_name = "BASEAvgRanksGroup" + str(g) + "Improved2Start.pdf"
    file_name = "BASEAvgRanksGroup" + str(g) + "Improved10Start.pdf"

    #plt.savefig(file_name, bbox_inches='tight')
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






