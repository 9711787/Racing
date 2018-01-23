import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy.stats
import seaborn as sns


def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))
#

# data = pd.read_csv('BASECase30072017.csv')
#
# # # data = pd.read_csv('BASECase025DIFF.csv')
# # #
# # # data = pd.read_csv('BASECase0125DIFF.csv')
# # #
# data = pd.read_csv('BASECase00625DIFF.csv')
# # #
# data = pd.read_csv('BASECase4DIFF.csv')
# # #
# # # data = pd.read_csv('BASECase1VARACCROSS.csv')
# # #
# # # data = pd.read_csv('BASECase01VARACCROSS.csv')
# # #
# data = pd.read_csv('BASECase100VARACCROSS.csv')
# # #
# # # data = pd.read_csv('BASECase01VARWITHIN.csv')
# # #
# # # data = pd.read_csv('BASECase10VARWITHIN.csv')
# # #
# #data = pd.read_csv('BASECase100VARWITHIN.csv')
# # #
# # # data = pd.read_csv('BASECaseExponentialZeroMean.csv')
# # #
# # # data = pd.read_csv('BASECaseExponentialZeroMedian.csv')
# # #
# # # #data = pd.read_csv('BASECase90ALPHA.csv')
# # #
# # # data = pd.read_csv('BASECase99ALPHA.csv')
# # #
# # data = pd.read_csv('BASECaseRealisticMC.csv')
# #
# # data = pd.read_csv('BASECase2Start.csv')
# #
# # data = pd.read_csv('BASECase10Start.csv')

data = pd.read_csv('BASECaseXXX.csv')


grayscale = sns.dark_palette("white", n_colors=5)
grayscale2 = sns.dark_palette("white", n_colors=5, reverse=True)

#MY METRICS
data['correct_avg'] = data['correct'] / data['no_surv']
data['no_surv_cand'] = data['no_surv'] / data['no_cand']
data['no_eval_cand'] = data['no_eval'] / data['no_cand']
data['budget_used'] = data['no_eval']/data['max_eval']
data['test_error'] = data['wrong_deletions']/data['tests_used_order']




race_parameters =  ["Hoeffding4Sigma","Bernstein4Sigma","DeleteWorstImprovedRho0.10","DeleteWorstImprovedRho0.20",
                                 "BlockingBayesFastGamma0",
                                 "BlockingBayesFastGamma1", "BayesImprovedGamma0",
                                 "BayesImprovedGamma1", "FriedmanT1Fast","FriedmanT2Fast",
                                 "BlockingHoeffdingFast4Sigma", "BlockingBernsteinFast4Sigma", "ANOVAFisherLSD",
                                 "pairedANOVAFisherLSD"]

plotting_groups = [("DeleteWorstImprovedRho0.10", "DeleteWorstImprovedRho0.20"),
                    ("Hoeffding4Sigma","BlockingHoeffdingFast4Sigma","Bernstein4Sigma", "BlockingBernsteinFast4Sigma"),
                   ("BayesImprovedGamma0","BlockingBayesFastGamma0","BayesImprovedGamma1","BlockingBayesFastGamma1"),
                    ("FriedmanT1Fast","FriedmanT2Fast","ANOVAFisherLSD","pairedANOVAFisherLSD")]

plotting_order = ["DeleteWorstImprovedRho0.10", "DeleteWorstImprovedRho0.20",
                    "Hoeffding4Sigma","BlockingHoeffdingFast4Sigma","Bernstein4Sigma", "BlockingBernsteinFast4Sigma",
                   "BayesImprovedGamma0","BlockingBayesFastGamma0","BayesImprovedGamma1","BlockingBayesFastGamma1",
                    "FriedmanT1Fast","FriedmanT2Fast","ANOVAFisherLSD","pairedANOVAFisherLSD"]


# race_labels =  {"Hoeffding4Sigma" : "Hoeffding" ,
#                 "Bernstein4Sigma" : "Bernstein" ,
#                 "DeleteWorstImprovedRho0.10" : "Del. Worst 10%" ,
#                 "DeleteWorstImprovedRho0.20" : "Del. Worst 20%",
#                 "BlockingBayesFastGamma0" : "Bl. Bayes γ = 0",
#                 "BlockingBayesFastGamma1" : "Bl. Bayes γ = 1",
#                 "BayesImprovedGamma0" : "Bayes γ = 0",
#                 "BayesImprovedGamma1" : "Bayes γ = 1",
#                 "FriedmanT1Fast" : "Friedman T1",
#                 "FriedmanT2Fast" : "Friedman T2",
#                 "BlockingHoeffdingFast4Sigma" : "Bl. Hoeffding",
#                 "BlockingBernsteinFast4Sigma" : "Bl. Bernstein",
#                 "ANOVAFisherLSD" : "ANOVA" ,
#                 "pairedANOVAFisherLSD" : "Bl. ANOVA"}

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



###FINAL PLOTS BASE CASE
base_subset = data
base_subset['race_type'].replace(race_labels, inplace=True)
plotting_order2 = []
for race in plotting_order:
    plotting_order2.append(race_labels[race])
results = ['budget_used', 'best_deleted', 'correct_avg', 'surv_rank_avg', 'false_positive', 'no_ins', 'no_eval_cand', 'no_surv_cand',
           'test_used_cand']

## CORRECT
sns.set_style("whitegrid")
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
fig, ax = plt.subplots(figsize=(25, 10))
sns.pointplot(y = 'race_type', x = 'correct', data = base_subset, order=plotting_order2, ax=ax, color="k", join=False, capsize=.4, ci=95 )
plt.xlim(-0.1, 1.1)
x_ticks = np.arange(0, 1.1, 0.10)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks, fontsize = 30)
ax.set_yticklabels(plotting_order2, fontsize = 30)
plt.xlabel("Proportion of Correct Races", fontsize=30)
plt.ylabel("")
#plt.title("Number of Candidates = 16, Candidate Difference = 0.5, Sigma Across = 10, Sigma Within = 1", fontsize=16)
# for i,box in enumerate(ax.artists):
#     box.set_edgecolor('black')
#     box.set_facecolor('white')
#
#     # iterate over whiskers and median lines
#     for j in range(6*i,6*(i+1)):
#          ax.lines[j].set_color('black')
plt.savefig("BASEcorrect10StartImproved.pdf", bbox_inches='tight')
plt.show()
print("avg correctness")
print(base_subset.groupby("race_type")["correct"].mean())
# # #


#WRONG TESTS
sns.set_style("whitegrid")
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
fig, ax = plt.subplots(figsize=(25, 10))
sns.boxplot(y = 'race_type', x = 'test_error', data = base_subset, order = plotting_order2, ax=ax)
plt.xlim(-0.1,1.1)
x_ticks = np.arange(0, 1.1, 0.10)
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks, fontsize = 30)
ax.set_yticklabels(plotting_order2, fontsize = 30)
plt.xlabel("Proportion Of Wrong Deletions", fontsize = 30)
plt.ylabel("")
#plt.title("Number of Candidates = 16, Candidate Difference = 0.5, Sigma Across = 10, Sigma Within = 1", fontsize=16)
for i,box in enumerate(ax.artists):
    box.set_edgecolor('black')
    box.set_facecolor('white')

    # iterate over whiskers and median lines
    for j in range(6*i,6*(i+1)):
         ax.lines[j].set_color('black')
plt.savefig("BASEwrong_test10StartImproved.pdf", bbox_inches='tight')
plt.show()


print(base_subset.groupby("race_type")['wrong_deletions'].mean())
print(base_subset.groupby("race_type")['tests_used_order'].mean())

print("avg correctness")
print(base_subset.groupby("race_type")["correct"].mean())
print(base_subset.groupby("race_type")["correct"].std())

print("wrong deletion")
print(base_subset.groupby("race_type")["test_error"].mean())
print(base_subset.groupby("race_type")["test_error"].std())


print("tasks")
print(base_subset.groupby("race_type")["no_task"].mean())
print(base_subset.groupby("race_type")["no_task"].std())

print("budget")
print(base_subset.groupby("race_type")["budget_used"].mean())
print(base_subset.groupby("race_type")["budget_used"].std())