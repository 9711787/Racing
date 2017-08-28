import numpy as np
import scipy.stats
import scipy.misc
import pandas as pd
import itertools
import time
import copy

### SIMULATION MODEL

def evaluations(no_cand = 16, max_task_cand = 15, cand_diff = 0.50, sigma_task = np.sqrt(10), sigma_eval = np.sqrt(1),
                task_error_type ="Normal", eval_error_type ="Normal"):

    max_task = no_cand * max_task_cand

    #GENERATE LIST OF CANDIDATE TRUE EXPECTED VALUES

    if cand_diff == 0:
        cand_means = [0]*no_cand
    else:
        cand_means = np.arange(0, no_cand*cand_diff, cand_diff)

    #GENERATE LIST OF TASK SPECIFIC ERRORS ERRORS

    if task_error_type == "Normal":
        task_error = np.random.normal(0, sigma_task, (max_task, 1))
    if task_error_type == "ExponentialZeroMean":
        task_error = (np.random.exponential(sigma_task, (max_task, 1)) - sigma_task)
    if task_error_type == "ExponentialZeroMedian":
        task_error = (np.random.exponential(sigma_task, (max_task, 1)) - sigma_task * np.log(2))

    #GENERATE WITHIN INSTANCE ERRORS AND EVALUATION

    if eval_error_type == "Normal":
        eval_error = np.random.normal(0, sigma_eval, (max_task, no_cand))
    if eval_error_type == "ExponentialZeroMean":
        eval_error = (np.random.exponential(sigma_eval, (max_task, no_cand)) - sigma_eval)
    if eval_error_type == "ExponentialZeroMedian":
        eval_error = (np.random.exponential(sigma_eval, (max_task, no_cand)) - sigma_eval * np.log(2))

    simulation_matrix = cand_means + task_error + eval_error

    return simulation_matrix


### RACING ALGORITHM

def race(data_matrix, race_type = None, start = 5, max_eval_cand = 15 , range_sd_multiplier = None, alpha = 0.05,
         multi_comp_type = "None", gamma = None, rho = None, out_type = "ranks"):

    delta = 1 - alpha
    no_cand = data_matrix.shape[1]
    max_eval = max_eval_cand * no_cand
    no_eval = 0
    no_task = 0
    max_inst_budget = start + (no_cand*(max_eval_cand - start))/2
    eval_matrix = np.empty((0,no_cand))
    survivors = np.ones(no_cand, dtype=bool)
    candidates = np.array(range(no_cand))
    candidate_ranks = np.arange(1, no_cand + 1)
    eliminations = []
    cand_elimination = []
    tests_used = 0
    positive_tests = 0
    tests_used_order = 0
    wrong_deletions = 0
    best_deleted = 0
    best_cand = 0

    if race_type in ['Hoeffding4Sigma', 'Bernstein4Sigma']:
        inf_list = [float("Inf")] * no_cand
        upper_bound = np.array(inf_list)
        lower_bound = -1 * np.array(inf_list)

    if multi_comp_type == "Budget":
        no_comp = max_eval - (no_cand - 1)

    if multi_comp_type == "None":
        no_comp = 1

    avg_surv_ranks = []
    best_cand_rank = []
    surv_in_race = []

    while (no_eval + sum(survivors)) <= max_eval:

        best_cand_rank.append(best_cand)
        surv_in_race.append(sum(survivors))

        #STOP THE RACE WHEN A WINNER IS KNOWN
        no_surv = sum(survivors)
        if no_surv == 1:
            break
        #COUNT NEXT INSTANCE
        no_task += 1

        #LOAD EVALUATIONS AND STORE THEM TEMPORARILY
        temp_eval = []
        for i in range(no_cand):
            if survivors[i] == True:
                temp_eval.append(data_matrix[no_task-1, i])
                no_eval += 1
            else:
                temp_eval.append(np.nan)

        eval_matrix = np.append(eval_matrix, np.array([temp_eval]), axis=0)

        if no_task >= start:
            if race_type == "Hoeffding4Sigma":
                range_sd_multiplier = 4
                means = np.mean(eval_matrix, axis=0)
                b = np.std(eval_matrix, axis=0) * range_sd_multiplier
                e = np.sqrt(((b**2)*(np.log(2*no_comp)-np.log(delta)))/(2*no_task))
                upper_bound = np.fmin(upper_bound, (means + e))
                lower_bound = np.fmax(lower_bound, (means - e))
                best_bound = np.nanmin(upper_bound)
                best_bound_index = np.argsort(upper_bound)[0]
                if best_cand == best_bound_index:
                    tests_used += no_surv - 1
                else:
                    tests_used += 1
                for j in range(no_cand - 1, -1, -1):
                        if lower_bound[j] > best_bound:
                            tests_used_order += 1
                            survivors[j] = False
                            if survivors[best_cand] == False:
                                best_deleted += 1
                                best_cand = np.min(candidates[survivors])
                            if np.any(candidates[j] < candidates[survivors]) == True:
                                wrong_deletions += 1
                            eliminations.append(no_task)
                            cand_elimination.append(j)

            elif race_type == "Bernstein4Sigma":
                range_sd_multiplier = 4
                means = np.mean(eval_matrix, axis=0)
                stds = np.std(eval_matrix, axis=0)
                b = stds * range_sd_multiplier
                e = stds * np.sqrt(2 * (((np.log(3 * no_comp) - np.log(delta))) / (2 * no_task))) + 3 * b * ( ((np.log(3 * no_comp) - np.log(delta))) / (2 * no_task))
                upper_bound = np.fmin(upper_bound, (means + e))
                lower_bound = np.fmax(lower_bound, (means - e))
                best = np.nanmin(upper_bound)
                best_bound_index = np.argsort(upper_bound)[0]
                if best_cand == best_bound_index:
                    tests_used += no_surv - 1
                else:
                    tests_used += 1
                for j in range(no_cand -1, -1, -1):
                    if survivors[j] == True:
                        if lower_bound[j] > best:
                            tests_used_order += 1
                            survivors[j] = False
                            if survivors[best_cand] == False:
                                best_deleted += 1
                                best_cand = np.min(candidates[survivors])
                            if np.any(candidates[j] < candidates[survivors]) == True:
                                wrong_deletions += 1
                            eliminations.append(no_task)
                            cand_elimination.append(j)


            elif race_type == "DeleteWorstImprovedRho0.10":
                rho = 0.10
                tests_used += no_surv - 1
                no_deletions = int(np.ceil(no_surv*rho))
                tests_used_order += no_deletions
                means = np.mean(eval_matrix, axis=0)
                worst = np.argsort(means)[no_surv-no_deletions:no_surv]
                rest = np.argsort(means)[0:no_surv-no_deletions]
                wrong_deletions += sum(np.max(candidates[rest]) > candidates[worst])
                for j in candidates[worst]:
                    survivors[candidates == [j]] = False
                    positive_tests += 1
                    if survivors[best_cand] == False:
                        best_deleted += 1
                        best_cand = np.min(candidates[survivors])
                    eliminations.append(no_task)
                    cand_elimination.append(j)

            elif race_type == "DeleteWorstImprovedRho0.20":
                rho = 0.20
                tests_used += no_surv - 1
                no_deletions = int(np.ceil(no_surv*rho))
                tests_used_order += no_deletions
                means = np.mean(eval_matrix, axis=0)
                worst = np.argsort(means)[no_surv-no_deletions:no_surv]
                rest = np.argsort(means)[0:no_surv - no_deletions]
                wrong_deletions += sum(np.max(candidates[rest]) > candidates[worst])

                for j in candidates[worst]:
                    survivors[candidates == [j]] = False
                    positive_tests += 1
                    if survivors[best_cand] == False:
                        best_deleted += 1
                        best_cand = np.min(candidates[survivors])
                    eliminations.append(no_task)
                    cand_elimination.append(j)



            elif race_type == "BlockingBayesFastGamma0":
                delta = delta/no_comp
                gamma = 0
                mean_best = np.argsort(np.mean(eval_matrix, axis=0))[0]
                if best_cand == mean_best:
                    tests_used += no_surv - 1
                else:
                    tests_used += 1
                for j in range(no_cand -1, -1, -1):
                    if survivors[j] == True:

                        diff = eval_matrix[:, mean_best] - eval_matrix[:, j]
                        T = (np.mean(diff) + gamma) / np.sqrt(np.var(diff, ddof=1)/no_task)
                        df = no_task - 1
                        C = scipy.stats.t.ppf(delta, df)
                        if T < C:
                            tests_used_order += 1
                            positive_tests += 1
                            survivors[j] = False
                            if survivors[best_cand] == False:
                                best_deleted += 1
                                best_cand = np.min(candidates[survivors])
                            if np.any(candidates[j] < candidates[survivors]) == True:
                                wrong_deletions += 1

                            eliminations.append(no_task)
                            cand_elimination.append(j)


            elif race_type == "BlockingBayesFastGamma1":
                delta = delta / no_comp
                gamma = 1
                mean_best = np.argsort(np.mean(eval_matrix, axis=0))[0]
                if best_cand == mean_best:
                    tests_used += no_surv - 1
                else:
                    tests_used += 1
                for j in range(no_cand -1, -1, -1):
                    if survivors[j] == True:
                        diff = eval_matrix[:, mean_best] - eval_matrix[:, j]
                        tests_used += 1
                        T = (np.mean(diff) + gamma) / np.sqrt(np.var(diff, ddof=1)/no_task)
                        df = no_task - 1
                        C = scipy.stats.t.ppf(delta, df)
                        if T < C:
                            tests_used_order += 1
                            positive_tests += 1
                            survivors[j] = False
                            if survivors[best_cand] == False:
                                best_deleted += 1
                                best_cand = np.min(candidates[survivors])
                            if np.any(candidates[j] < candidates[survivors]) == True:
                                wrong_deletions += 1
                            eliminations.append(no_task)
                            cand_elimination.append(j)



            elif race_type == "BayesImprovedGamma0":
                delta = delta / no_comp
                gamma = 0
                means = np.mean(eval_matrix, axis=0)
                var = np.var(eval_matrix, axis=0, ddof=1)
                mean_best = np.argsort(means)[0]
                var_best = var[mean_best]
                u_best = var_best/no_task
                if best_cand == mean_best:
                    tests_used += no_surv - 1
                else:
                    tests_used += 1
                for j in range(no_cand -1, -1, -1) :
                    if survivors[j] == True:
                        tests_used += 1
                        u = var[j] / no_task
                        T = (means[mean_best] - means[j] + gamma) / np.sqrt(u_best + u)
                        b = u_best / (u_best + u)
                        df = (b ** 2 / (no_task - 1) + (1 - b) ** 2 / (no_task - 1)) ** -1
                        C = scipy.stats.t.ppf(delta, df)
                        if T < C:
                            tests_used_order += 1
                            positive_tests += 1
                            survivors[j] = False
                            if survivors[best_cand] == False:
                                best_deleted += 1
                                best_cand = np.min(candidates[survivors])
                            if np.any(candidates[j] < candidates[survivors]) == True:
                                wrong_deletions += 1
                            eliminations.append(no_task)
                            cand_elimination.append(j)


            elif race_type == "BayesImprovedGamma1":
                delta = delta / no_comp

                gamma = 1
                means = np.mean(eval_matrix, axis=0)
                var = np.var(eval_matrix, axis=0, ddof=1)
                mean_best = np.argsort(means)[0]
                var_best = var[mean_best]
                u_best = var_best/no_task
                if best_cand == mean_best:
                    tests_used += no_surv - 1
                else:
                    tests_used += 1
                for j in range(no_cand -1, -1, -1) :
                    if survivors[j] == True:

                        u = var[j] / no_task
                        T = (means[mean_best] - means[j] + gamma) / np.sqrt(u_best + u)

                        b = u_best / (u_best + u)
                        df = (b ** 2 / (no_task - 1) + (1 - b) ** 2 / (no_task - 1)) ** -1
                        C = scipy.stats.t.ppf(delta, df)
                        if T < C:
                            tests_used_order += 1
                            positive_tests += 1
                            survivors[j] = False
                            if survivors[best_cand] == False:
                                best_deleted += 1
                                best_cand = np.min(candidates[survivors])
                            if np.any(candidates[j] < candidates[survivors]) == True:
                                wrong_deletions += 1
                            eliminations.append(no_task)
                            cand_elimination.append(j)


            elif race_type == "FriedmanT1Fast":
                temp_surv = copy.deepcopy(survivors)
                rank_matrix = np.apply_along_axis(scipy.stats.rankdata, 1, eval_matrix[:, survivors])
                R_vector = np.apply_along_axis(np.sum, 0, rank_matrix)
                A1 = np.sum(rank_matrix ** 2)
                sum_R= np.sum(R_vector**2)
                C1 = (no_task*no_surv*(no_surv+1)**2)/4
                T1 = ((no_surv-1)*(sum_R - no_task*C1))/(A1-C1)
                C = scipy.stats.chi2.ppf(1-delta, no_task-1)
                mean_best = np.argsort(R_vector)[0]
                df = (no_surv - 1)*(no_task - 1)
                if T1 > C:
                    if best_cand == mean_best:
                        tests_used += no_surv - 1
                    else:
                        tests_used += 1
                    for j in range(no_surv - 1, -1, -1):
                        t_test = (R_vector[mean_best] - R_vector[j])/np.sqrt((1-(T1/((no_task*(no_surv - 1)))))
                            *(((A1-C1)*2*no_task)/((no_surv - 1)*(no_task - 1))))
                        crit_value = scipy.stats.t.ppf(delta, df)
                        if t_test < crit_value:
                            tests_used_order += 1
                            positive_tests += 1
                            temp_surv[candidates == candidates[survivors][j]] = False
                            if temp_surv[best_cand] == False:
                                best_deleted += 1
                                best_cand = np.min(candidates[temp_surv])
                            if np.any(candidates[survivors][j] < candidates[temp_surv]) == True:
                                wrong_deletions += 1
                            eliminations.append(no_task)
                            cand_elimination.append(candidates[candidates == candidates[survivors][j]])
                    survivors = copy.deepcopy(temp_surv)


            elif race_type == "FriedmanT2Fast":
                temp_surv = copy.deepcopy(survivors)
                rank_matrix = np.apply_along_axis(scipy.stats.rankdata, 1, eval_matrix[:, survivors])
                R_vector = np.apply_along_axis(np.sum, 0, rank_matrix)
                A1 = np.sum(rank_matrix ** 2)
                sum_R= np.sum(R_vector**2)
                C1 = (no_task*no_surv*(no_surv+1)**2)/4
                T1 = ((no_surv-1)*(sum_R - no_task*C1))/(A1-C1)
                T2 = ((no_task-1)*T1)/(no_task*(no_surv-1)-T1)
                C = scipy.stats.f.ppf(1-delta,no_task-1,(no_surv - 1)*(no_task - 1))
                mean_best = np.argsort(R_vector)[0]
                df = (no_surv - 1)*(no_task - 1)
                if T2 > C:
                    if best_cand == mean_best:
                        tests_used += no_surv - 1
                    else:
                        tests_used += 1
                    for j in range(no_surv - 1, -1, -1) :
                        t_test = (R_vector[mean_best] - R_vector[j])/np.sqrt((1-(T1/((no_task*(no_surv - 1)))))
                            *(((A1-C1)*2*no_task)/((no_surv - 1)*(no_task - 1))))
                        crit_value = scipy.stats.t.ppf(delta, df)
                        if t_test < crit_value:
                            tests_used_order += 1
                            positive_tests += 1
                            temp_surv[candidates == candidates[survivors][j]] = False
                            if temp_surv[best_cand] == False:
                                best_deleted += 1
                                best_cand = np.min(candidates[temp_surv])
                            if np.any(candidates[survivors][j] < candidates[temp_surv]) == True:
                                wrong_deletions += 1
                            eliminations.append(no_task)
                            cand_elimination.append(candidates[candidates == candidates[survivors][j]])
                    survivors = copy.deepcopy(temp_surv)


            elif race_type == "BlockingHoeffdingFast4Sigma":
                range_sd_multiplier = 4
                mean_best = np.argsort(np.mean(eval_matrix, axis=0))[0]
                if best_cand == mean_best:
                    tests_used += no_surv - 1
                else:
                    tests_used += 1
                for j in range(no_cand -1, -1, -1) :
                    if survivors[j] == True:
                        diff = eval_matrix[:, mean_best] - eval_matrix[:, j]
                        means = np.mean(diff, axis=0)
                        b = np.std(diff, axis=0) * range_sd_multiplier
                        e = np.sqrt(((b ** 2) * (np.log(2 * no_comp) - np.log(delta))) / (2 * no_task))
                        if means + e < 0:
                            tests_used_order += 1
                            positive_tests += 1
                            survivors[j] = False
                            if survivors[best_cand] == False:
                                best_deleted += 1
                                best_cand = np.min(candidates[survivors])
                            if np.any(candidates[j] < candidates[survivors]) == True:
                                wrong_deletions += 1
                            eliminations.append(no_task)
                            cand_elimination.append(j)

            elif race_type == "BlockingBernsteinFast4Sigma":
                range_sd_multiplier = 4
                mean_best = np.argsort(np.mean(eval_matrix, axis=0))[0]
                if best_cand == mean_best:
                    tests_used += no_surv - 1
                else:
                    tests_used += 1
                for j in range(no_cand -1, -1, -1) :
                    if survivors[j] == True:
                        diff = eval_matrix[:, mean_best] - eval_matrix[:, j]
                        means = np.mean(diff, axis=0)
                        b = np.std(diff, axis=0) * range_sd_multiplier
                        e = np.std(diff, axis=0) * np.sqrt(2 * (((np.log(3 * no_comp) - np.log(delta)))
                            / (2 * no_task))) + 3 * b * (((np.log(3 * no_comp) - np.log(delta))) / (2 * no_task))

                        if means + e < 0:
                            positive_tests += 1
                            tests_used_order += 1
                            survivors[j] = False
                            if survivors[best_cand] == False:
                                best_deleted += 1
                                best_cand = np.min(candidates[survivors])
                            if np.any(candidates[j] < candidates[survivors]) == True:
                                wrong_deletions += 1
                            eliminations.append(no_task)
                            cand_elimination.append(j)


            elif race_type == "ANOVAFisherLSD":
                means = np.mean(eval_matrix, axis=0)
                grand_mean = np.nanmean(eval_matrix)
                mean_best = np.argsort(means)[0]
                ss_total = np.nansum((eval_matrix-grand_mean)**2)
                ss_treat = np.nansum(no_task*(means**2)) - no_task*no_surv*(grand_mean**2)
                ss_error = ss_total - ss_treat
                mean_ss_treat = ss_treat / (no_surv - 1)
                mean_ss_error = ss_error / (no_surv*(no_task - 1))
                F_stat =  mean_ss_treat / mean_ss_error
                C = scipy.stats.f.ppf(1-delta, no_surv - 1, no_surv*(no_task - 1))
                if F_stat > C:
                    if best_cand == mean_best:
                        tests_used += no_surv - 1
                    else:
                        tests_used += 1
                    for j in range(no_cand -1, -1, -1) :
                        if survivors[j] == True:
                            T = (means[mean_best] - means[j]) / np.sqrt(mean_ss_error*(2/no_task))
                            df = (no_surv*(no_task - 1))
                            C = scipy.stats.t.ppf(delta, df)
                            if T < C:
                                tests_used_order += 1
                                positive_tests += 1
                                survivors[j] = False
                                if survivors[best_cand] == False:
                                    best_deleted += 1
                                    best_cand = np.min(candidates[survivors])
                                if np.any(candidates[j] < candidates[survivors]) == True:
                                    wrong_deletions += 1
                                eliminations.append(no_task)
                                cand_elimination.append(j)
                                eval_matrix[:, j] = np.nan

            elif race_type == "pairedANOVAFisherLSD":
                treat_means = np.mean(eval_matrix, axis=0)
                sub_means = np.mean(eval_matrix, axis=1)
                grand_mean = np.nanmean(eval_matrix)
                mean_best = np.argsort(treat_means)[0]
                ss_total = np.nansum((eval_matrix - grand_mean) ** 2)
                ss_treat = np.nansum(no_task * (treat_means ** 2)) - no_task * no_surv * (grand_mean ** 2)
                ss_sub = np.nansum(no_surv * (sub_means ** 2)) - no_task * no_surv * (grand_mean ** 2)
                ss_error = ss_total - ss_treat - ss_sub
                mean_ss_treat = ss_treat / (no_surv - 1)
                mean_ss_error = ss_error / ((no_surv * (no_task - 1)) - (no_task - 1))
                F_stat = mean_ss_treat / mean_ss_error
                C = scipy.stats.f.ppf(1 - delta, no_surv - 1, (no_surv * (no_task - 1)) - (no_task - 1))
                if F_stat > C:
                    for j in range(no_cand -1, -1, -1):

                        if best_cand == mean_best:
                            tests_used += no_surv - 1
                        else:
                            tests_used += 1
                        if survivors[j] == True:
                            T = (treat_means[mean_best] - treat_means[j]) / np.sqrt(mean_ss_error * (2 / no_task))
                            df = (no_surv * (no_task - 1))
                            C = scipy.stats.t.ppf(delta, df)
                            if T < C:
                                tests_used_order += 1
                                positive_tests += 1
                                survivors[j] = False
                                if survivors[best_cand] == False:
                                    best_deleted += 1
                                    best_cand = np.min(candidates[survivors])
                                if np.any(candidates[j] < candidates[survivors]) == True:
                                    wrong_deletions += 1
                                eliminations.append(no_task)
                                cand_elimination.append(j)
                                eval_matrix[:, j] = np.nan
            else:
                print(race_type, "is invalid race_type")
                break

        avg_surv_ranks.append(np.mean(candidate_ranks[survivors]))

    avg_surv_ranks = np.array(avg_surv_ranks + [avg_surv_ranks[-1]]*int(max_inst_budget-len(avg_surv_ranks)),ndmin = 2)

    if out_type == "results":
        output_data = {"no_task": no_task, "no_eval": no_eval, "no_surv": sum(survivors), "correct": survivors[0],
                       "winner": np.array(range(no_cand))[survivors][0],
                       "tests_used" : tests_used, "best_deleted": best_deleted,
                       "max_eval": max_eval,
                       "gamma": gamma, "rho": rho, "range_sd_multiplier" : range_sd_multiplier,
                       "tests_used_order" : tests_used_order,
                       "wrong_deletions": wrong_deletions}

    elif out_type == "ranks":
        output_data = avg_surv_ranks

    return output_data

### FUNCTION TO GENERATE PARAMETERS

def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


### RACING RESULTS

sim_parameters = {"no_cand": [16], "max_task_cand": [20],
             "cand_diff" : [0.5], "sigma_task" : [np.sqrt(10)],
             "sigma_eval" : [np.sqrt(1)], "task_error_type" : ["Normal"],
                   "eval_error_type" : ["Normal"]}

race_parameters = {"max_eval_cand": [15],"start": [5],
                   "race_type": ["Hoeffding4Sigma","Bernstein4Sigma","DeleteWorstImprovedRho0.10","DeleteWorstImprovedRho0.20",
                                 "BlockingBayesFastGamma0",
                                 "BlockingBayesFastGamma1", "BayesImprovedGamma0",
                                 "BayesImprovedGamma1", "FriedmanT1Fast","FriedmanT2Fast",
                                 "BlockingHoeffdingFast4Sigma", "BlockingBernsteinFast4Sigma", "ANOVAFisherLSD",
                                 "pairedANOVAFisherLSD"], "alpha" : [0.95], "multi_comp_type" : ["None"],
                                 "out_type" : ["results"]}

counter = 0
start = time.time()
final_data = pd.DataFrame()
for sim_par in dict_product(sim_parameters):
    for seed in range(0, 50):
        seed_par = {"seed": seed}
        np.random.seed(seed)
        print(sim_par)
        sim_data = evaluations(**sim_par)
        for race_par in dict_product(race_parameters):
            print(race_par)
            start_run = time.time()
            results = race(sim_data, **race_par)
            stop_run = time.time()
            print("Run time:", stop_run - start_run, "sec" )
            result_append = {'race_id': counter,'run_time': stop_run - start_run, **seed_par, **sim_par, **race_par, **results}
            #print(result_append)
            final_data = final_data.append(result_append, ignore_index= True)
            counter += 1
            print(counter)
print(final_data)
final_data.to_csv('BASECaseXXX.csv')
stop = time.time()
print("Elapsed time:", stop - start)


### MEAN RANKS

sim_parameters = {"no_cand": [16], "max_task_cand": [20],
             "cand_diff" : [0.5], "sigma_task" : [np.sqrt(10)],
             "sigma_eval" : [np.sqrt(1)], "task_error_type" : ["Normal"],
                   "eval_error_type" : ["Normal"]}

race_parameters = {"max_eval_cand": [15],"start": [5],
                   "race_type": ["Hoeffding4Sigma","Bernstein4Sigma","DeleteWorstImprovedRho0.10","DeleteWorstImprovedRho0.20",
                                 "BlockingBayesFastGamma0",
                                 "BlockingBayesFastGamma1", "BayesImprovedGamma0",
                                 "BayesImprovedGamma1", "FriedmanT1Fast","FriedmanT2Fast",
                                 "BlockingHoeffdingFast4Sigma", "BlockingBernsteinFast4Sigma", "ANOVAFisherLSD",
                                 "pairedANOVAFisherLSD"], "alpha" : [0.95], "multi_comp_type" : ["None"],
                                 "out_type" : ["ranks"]}

start = race_parameters["start"][0]
no_cand = sim_parameters["no_cand"][0]
max_eval_cand = race_parameters["max_eval_cand"][0]
dimension = int(start + (no_cand*(max_eval_cand - start))/2)
base_data = np.empty((0, dimension ))
print(base_data.shape)
base_data_info = pd.DataFrame()

counter = 0
start = time.time()

for sim_par in dict_product(sim_parameters):
    for seed in range(0, 50):
        seed_par = {"seed": seed}
        np.random.seed(seed)
        print(sim_par)
        sim_data = evaluations(**sim_par)
        for race_par in dict_product(race_parameters):
            print(race_par)
            start_run = time.time()
            results = race(sim_data, **race_par)
            print(results.shape)
            stop_run = time.time()
            print("Run time:", stop_run - start_run, "sec")
            data_append = {'race_id': counter,'run_time': stop_run - start_run, **seed_par, **sim_par, **race_par}
            base_data_info = base_data_info.append(data_append, ignore_index= True)
            base_data = np.append(base_data, results.reshape((1,dimension )), axis=0)
            counter += 1
            print(counter)

print(base_data_info)
base_data_info.to_csv('AvgRankBASE_info.csv')
np.savetxt('AvgRankBASE.csv', base_data, delimiter=",")
stop = time.time()
print("Elapsed time:", stop - start)