#!/usr/bin/python3
import numpy as np
import scipy.stats
import scipy.misc
import pandas as pd
import itertools
import time
import copy
from IPython.core.debugger import Tracer; debug_here = Tracer()

### SIMULATION MODEL
## This function generates a matrix (max_task_cand * no_cand, no_cand) of
## simulated evaluations
def gen_simulation_matrix(no_cand = 16, max_task_cand = 15, cand_diff = 0.5,
                          sigma_task = np.sqrt(1), sigma_eval = np.sqrt(1),
                          task_error_type = "Normal", eval_error_type = "Normal"):
    # Generate list of candidate true expected values
    if cand_diff == 0:
        cand_means = [0] * no_cand
    else:
        cand_means = np.arange(0, no_cand * cand_diff, cand_diff)

    max_task = no_cand * max_task_cand

    # Generate list of task specific errors
    if task_error_type == "Normal":
        task_error = np.random.normal(0, sigma_task, (max_task, 1))
    elif task_error_type == "ExponentialZeroMean":
        task_error = np.random.exponential(sigma_task, (max_task, 1)) - sigma_task
    elif task_error_type == "ExponentialZeroMedian":
        task_error = np.random.exponential(sigma_task, (max_task, 1)) - sigma_task * np.log(2)

    # Generate within instance errors and evaluation
    if eval_error_type == "Normal":
        eval_error = np.random.normal(0, sigma_eval, (max_task, no_cand))
    if eval_error_type == "ExponentialZeroMean":
        eval_error = np.random.exponential(sigma_eval, (max_task, no_cand)) - sigma_eval
    if eval_error_type == "ExponentialZeroMedian":
        eval_error = np.random.exponential(sigma_eval, (max_task, no_cand)) - sigma_eval * np.log(2)

    # We could np.clip it to [- 2 * (sigma_task + sigma_eval), + 2 * (sigma_task + sigma_eval) ]
    simulation_matrix = cand_means + task_error + eval_error
    return simulation_matrix


# An object to store race statistics
class Race:
    def __init__(self, data_matrix, max_eval, alpha = 0.95, start = 5):
        self.data_matrix = np.copy(data_matrix)
        self.max_task, self.no_cand = self.data_matrix.shape
        self.max_eval = max_eval
        self.alpha = alpha
        self.start = start
        # Computed
        self.delta = 1 - alpha
        self.alive = np.ones(self.no_cand, dtype = bool)
        # candidates are ordered from best to worst
        self.true_best_cand = 0
        # Stats
        self.no_task = 0
        self.no_eval = 0
        self.tests_used = 0
        self.elimination_tests_used = 0
        self.start_time = time.time()
        # best_deleted: number of times we delete the best candidate remaining in the race
        self.best_deleted = 0
        # wrong_deletions: number of times we eliminate a candidate while a worse one survives
        self.wrong_deletions = 0

    def run(self):
        avg_surv_ranks = []
        while (self.no_eval + self.alive.sum()) <= self.max_eval \
              and self.alive.sum() > 1 and self.no_task < self.max_task:

            self.no_task += 1
            self.no_eval += self.alive.sum()
            # Load evaluations and store them temporarily
            if self.no_task >= self.start:
                self.data_matrix[:, np.logical_not(self.alive)] = np.nan
                self.elimination(self.data_matrix[:self.no_task, ])

            avg_surv_ranks.append(np.mean(1 + np.flatnonzero(self.alive)))

        return avg_surv_ranks, self.get_stats()

    def eliminate_candidate(self, j):
        assert self.alive[j] == True, "killing the dead!"
        self.alive[j] = False
        survivors = np.flatnonzero(self.alive)
        if j == self.true_best_cand:
            self.best_deleted += 1
            self.true_best_cand = np.min(survivors)
        # If we delete a candidate with an index smaller than any survivor,
        # then we deleted the wrong candidate.
        ## FIXME: This should probably be checked after all candidates are
        ## marked as surviving or not.
        if np.any(j < survivors) == True:
            self.wrong_deletions += 1

    def get_stats(self):
        return {"no_task"          : self.no_task,
                "no_eval"          : self.no_eval,
                "tests_used"       : self.tests_used,
                "elimination_tests_used" : self.elimination_tests_used,
                "best_deleted"     : self.best_deleted,
                "wrong_deletions"  : self.wrong_deletions,
                "time" : time.time() - self.start_time,
                "no_surv": self.alive.sum(),
                "correct": self.alive[0],
                "winner": np.flatnonzero(self.alive)[0] }


class BoundsRace(Race):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Maroon & Moore (1997)
        self.range_sd_multiplier = 4
        # Estimation of number of bounds that need to be computed
        ## It is not clear if no_bounds can be adjusted dynamically and still
        ## satisfy the bounds inequalities. This is a Conservative option. If
        ## max_eval is not known, then max_task * no_cand is the most
        ## conservative estimate.
        no_bounds = self.max_eval - (self.no_cand * (self.start - 1))
        # (1 - alpha) = (1 - alpha_race) / no_bounds
        self.delta = self.delta / no_bounds
        # For shrinking bounds
        self.upper_bound = np.array([float("Inf")] * self.no_cand)
        self.lower_bound = -1 * self.upper_bound

    def HoeffdingBounds(self, eval_matrix):
        # Number of rows
        no_task = eval_matrix.shape[0]
        # This is a rule of thumb when the true range is unknown
        R = np.std(eval_matrix, axis = 0) * self.range_sd_multiplier
        e = R * np.sqrt(np.log(2 / self.delta) / (2 * no_task))
        # 0.5 is the true difference
        # print("n > " + str((b**2) * np.log(2 / self.delta) / (2 *  0.5**2)) + "\n")
        ## Column means
        means = np.mean(eval_matrix, axis = 0)
        best = np.nanargmin(means)
        return best, means - e, means + e

    def BernsteinBounds(self, eval_matrix):
        # Number of rows
        no_task = eval_matrix.shape[0]
        stds = np.std(eval_matrix, axis = 0)
        # This is a rule of thumb when the true range is unknown
        R = stds * self.range_sd_multiplier
        ## This formula comes from:
        # Mnih, V., Szepesvari, C., Audibert, J.Y.: Empirical Bernstein
        # stopping. In Proceedings of the 25th International Conference on
        # Machine Learning, pp. 672-679. ACM Press, New York, NY (2008)
        e = stds * np.sqrt(2 * np.log(3 / self.delta) / no_task) \
            + 3 * R * np.log(3 / self.delta) / no_task
        ## Column means
        means = np.mean(eval_matrix, axis = 0)
        best = np.nanargmin(means)
        return best, means - e, means + e

class IndependentBoundsRace(BoundsRace):

    def eliminate_by_bounds(self, best, lbounds, ubounds):
        self.lower_bound = np.fmax(self.lower_bound, lbounds)
        self.upper_bound = np.fmin(self.upper_bound, ubounds)
        # for i in range(len(self.alive)):
        #     print("%i: [%f,%f]\n" % (i, self.lower_bound[i], self.upper_bound[i]))
        assert np.all(self.lower_bound < self.upper_bound), "lower bounds must be larger than upper bounds"
        #print(self.lower_bound)
        #print(self.upper_bound)
        best_ubound = self.upper_bound[best]
        # Each bound computation is a "test"
        self.tests_used += self.alive.sum()
        # But we actually do one fewer comparisons
        self.elimination_tests_used += self.alive.sum() - 1

        for j in reversed(np.flatnonzero(self.alive)):
            assert self.alive[j] == True, "survivor" + j + "should be survivors!"
            # We could skip best == j, but it is not worth it.
            if self.lower_bound[j] > best_ubound:
                self.eliminate_candidate(j)

    def elimination(self, eval_matrix):
        best, lbounds, ubounds = self.compute_bounds(eval_matrix)
        self.eliminate_by_bounds(best, lbounds, ubounds)


class HoeffdingRace(IndependentBoundsRace):
    def __init__(self, *args, **kwargs):
        # Call base class
        super().__init__(*args, **kwargs)
        self.compute_bounds = self.HoeffdingBounds

class BernsteinRace(IndependentBoundsRace):
    def __init__(self, *args, **kwargs):
        # Call base class
        super().__init__(*args, **kwargs)
        self.compute_bounds = self.BernsteinBounds

class BlockingRace(BoundsRace):
    def elimination(self, eval_matrix):
        best = np.nanargmin(np.mean(eval_matrix, axis = 0))
        assert self.alive[best] == True, "best is dead!"

        for j in reversed(np.flatnonzero(self.alive)):
            if best == j: continue
            assert self.alive[j] == True, "j is dead!"
            diff = eval_matrix[:, best] - eval_matrix[:, j]
            best, lbounds, ubounds = self.compute_bounds (diff)
            self.tests_used += 1
            self.elimination_tests_used += 1
            if ubounds < 0:
                self.eliminate_candidate(j)

class BlockingHoeffdingRace(BlockingRace):
    def __init__(self, *args, **kwargs):
        # Call base class
        super().__init__(*args, **kwargs)
        self.compute_bounds = self.HoeffdingBounds

class BlockingBernsteinRace(BlockingRace):
    def __init__(self, *args, **kwargs):
        # Call base class
        super().__init__(*args, **kwargs)
        self.compute_bounds = self.BernsteinBounds

class DeleteWorstRace(Race):
    def __init__(self, rho, **kwargs):
        # Call base class
        super().__init__(**kwargs)
        assert rho >= 0 and rho <= 1
        self.rho = rho

    def elimination(self, eval_matrix):
        no_surv = self.alive.sum()
        no_deletions = int(np.ceil(no_surv * self.rho))
        self.tests_used += no_surv
        self.elimination_tests_used += no_deletions
        # We keep only up to no_surv to remove NA entries.
        sorted_surv = np.argsort(np.mean(eval_matrix, axis = 0))[:no_surv]
        delete = np.sort(sorted_surv[ -no_deletions : ])
        for j in reversed(delete):
            self.eliminate_candidate(j)


class BlockingBayesianRace(Race):
    
    def elimination(self, eval_matrix):
        means = np.mean(eval_matrix, axis = 0)
        best = np.nanargmin(means)
        for j in reversed(np.flatnonzero(self.alive)):
            if j == best: continue
            self.tests_used += 1
            self.elimination_tests_used += 1
            # # All these should be equivalent to a paired t-test.
            # no_task = eval_matrix.shape[0]
            # df = no_task - 1
            # # FIXME: Not sure if this is correct
            # crit_value = scipy.stats.t.ppf(1 - self.delta, df)
            # diff = eval_matrix[:, best] - eval_matrix[:, j]
            # statistic = np.mean(diff) / np.sqrt(np.var(diff, ddof = 1) / no_task)
            # if statistic < crit_value:
            #     self.eliminate_candidate(j)
            ## FIXME: This could be tighter by using a one-tailed tests
            pvalue = scipy.stats.ttest_rel(eval_matrix[:,best], eval_matrix[:,j]).pvalue
            if pvalue < self.delta:
                self.eliminate_candidate(j)

class BayesianRace(Race):
    
    def elimination(self, eval_matrix):
        means = np.mean(eval_matrix, axis = 0)
        best = np.nanargmin(means)

        # Number of rows (tasks)
        # n = eval_matrix.shape[0]
        # u = np.var(eval_matrix, axis = 0, ddof = 1) / n
        for j in reversed(np.flatnonzero(self.alive)):
            if j == best: continue
            self.tests_used += 1
            self.elimination_tests_used += 1
            pvalue = scipy.stats.ttest_ind(eval_matrix[:, best], eval_matrix[:, j]).pvalue
            if pvalue < self.delta:
                self.eliminate_candidate(j)
            # # The above should be equivalent to the code below:
            # b = u[best] / (u[best] + u[j])
            # df = (n - 1) / (b ** 2 + (1 - b) ** 2)
            # # This is equivalent to R's qt() function
            # crit_value = scipy.stats.t.ppf(1 - self.delta, df)
            # statistic = (means[best] - means[j]) / np.sqrt(u[best] + u[j])
            # pvalue = scipy.stats.t.sf(np.abs(statistic), df) * 2
            # if statistic < crit_value:
            #     self.eliminate_candidate(j)
            #     assert pvalue < 1 - self.delta, \
            #         "statistic < crit_value but pvalue >= alpha: statistic = %f ; df = %f ; crit_value = %f ; pvalue = %g ; alpha = %f\n " % (statistic, df, crit_value, pvalue, 1 - self.delta)
            # else:
            #     assert pvalue >= 1 - self.delta, \
            #         "statistic >= crit_value but pvalue < alpha: statistic = %f ; df = %f ; crit_value = %f ; pvalue = %g ; alpha = %f\n " % (statistic, df, crit_value, pvalue, 1 - self.delta)


class FRace(Race):
    def __init__(self, fstat, **kwargs):
        # Call base class
        super().__init__(**kwargs)
        self.fstat = fstat

    def elimination(self, eval_matrix):
        survivors = np.flatnonzero(self.alive)
        no_surv = survivors.shape[0]
        no_task = eval_matrix.shape[0]
        # Conover, "Practical Nonparametric Statistics",
        # 1999, pages 369-371.
        r_matrix = np.apply_along_axis(scipy.stats.rankdata, 1, eval_matrix[:, self.alive])
        #debug_here()
        R = np.sum(r_matrix, axis = 0)
        A = np.sum(r_matrix ** 2)
        sumR2 = np.sum(R ** 2)
        C = no_task * no_surv * ((no_surv + 1) ** 2) / 4
        T1 = (no_surv - 1) * (sumR2 - no_task * C) / (A - C)
        T2 = (no_task - 1) * T1 / (no_task * (no_surv - 1) - T1)
        df1 = no_surv - 1
        df2 = (no_task - 1) * (no_surv - 1)
        if self.fstat == "T1":
            statistic = T1
            crit_value = scipy.stats.chi2.ppf(1 - self.delta, df1)
        elif self.fstat == "T2":
            statistic = T2
            crit_value = scipy.stats.f.ppf(1 - self.delta, df1, df2)

        self.tests_used += 1
        if statistic <= crit_value:
            return

        order = np.argsort(R)
        best = order[0]
        crit_diff = scipy.stats.t.ppf(1 - self.delta / 2, df2) \
                     * np.sqrt(2 * (no_task * A - sumR2) / df2)
        delete = []
        for j in range(1, len(order)):
            self.elimination_tests_used += 1
            if np.abs(R[order[j]] - R[order[best]]) > crit_diff:
                delete = np.sort(survivors[order[j:]])
                break
        # Delete in reverse order
        for j in reversed(delete):
            self.eliminate_candidate(j)

class AnovaRace(Race):
    def __init__(self, blocking = False, **kwargs):
        # Call base class
        super().__init__(**kwargs)
        self.blocking = blocking

    def elimination(self, eval_matrix):
        survivors = np.flatnonzero(self.alive)
        eval_surv = eval_matrix[:, self.alive]
        no_surv = survivors.shape[0]
        no_task = eval_matrix.shape[0]
        means = np.mean(eval_surv, axis = 0)
        grand_mean = np.mean(eval_surv)
        # One-way ANOVA
        ss_total = np.sum((eval_surv - grand_mean) ** 2)
        ss_treat = no_task * np.sum((means - grand_mean) ** 2)
        ss_tasks = 0
        df_treat = no_surv - 1
        df_error = no_surv * (no_task - 1)
        if self.blocking == True:
            # Repeated measures ANOVA (task as blocking factor)
            ss_tasks = no_surv * np.sum((np.mean(eval_surv, axis = 1) - grand_mean) ** 2)
            df_error = (no_surv - 1) * (no_task - 1)

        ss_error = ss_total - ss_treat - ss_tasks
        ms_treat = ss_treat / df_treat
        ms_error = ss_error / df_error
        F_stat = ms_treat / ms_error
        crit_value = scipy.stats.f.ppf(1 - self.delta, df_treat, df_error)

        self.tests_used += 1
        if F_stat <= crit_value:
            return

        best = np.argsort(means)[0]
        #debug_here()
        t_crit_value = scipy.stats.t.ppf(1 - self.delta, df_error) \
                       * np.sqrt(2 * ms_error / no_task)
        for j in reversed(range(len(survivors))):
            if best == j: continue
            self.elimination_tests_used += 1
            # Least Significant Difference:
            # Snedecor, G. W. and Cochran, W. G. (1967). Statistical Methods,
            # 6th edn, Iowa State University Press, Ames, IA, USA.
            t_stat = np.abs(means[j] - means[best])
            if t_stat > t_crit_value:
                self.eliminate_candidate(survivors[j])


def run_race(data_matrix, race_type = None, start = 5, max_eval_cand = 15,
             alpha = 0.95,
             gamma = None, rho = None):
    max_task, no_cand = data_matrix.shape
    # FIXME: This should be defined outside to simplify this function.
    max_eval = max_eval_cand * no_cand
    if race_type == "Hoeffding":
        race = HoeffdingRace(data_matrix, max_eval, alpha)
    elif race_type == "Bernstein":
        race = BernsteinRace(data_matrix, max_eval, alpha)
    elif race_type == "Bayesian":
        race = BayesianRace(data_matrix, max_eval, alpha)
    elif race_type == "BlockingHoeffding":
        race = BlockingHoeffdingRace(data_matrix, max_eval, alpha)
    elif race_type == "BlockingBernstein":
        race = BlockingBernsteinRace(data_matrix, max_eval, alpha)
    elif race_type == "BlockingBayesian":
        race = BlockingBayesianRace(data_matrix, max_eval, alpha)
    elif race_type == "FRaceT1":
        race = FRace(fstat = "T1", data_matrix = data_matrix, max_eval = max_eval, alpha = alpha)
    elif race_type == "FRaceT2":
        race = FRace(fstat = "T2", data_matrix = data_matrix, max_eval = max_eval, alpha = alpha)
    elif race_type == "AnovaRace":
        race = AnovaRace(data_matrix = data_matrix, max_eval = max_eval, alpha = alpha)
    elif race_type == "BlockingAnovaRace":
        race = AnovaRace(blocking = True, data_matrix = data_matrix, max_eval = max_eval, alpha = alpha)
    elif race_type == "DeleteWorst":
        race = DeleteWorstRace(rho = rho, data_matrix = data_matrix, max_eval = max_eval, alpha = alpha)
    else:
        raise NameError('race_type' + race_type + 'unrecognized!')

    return race.run()

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def dict_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def run_race_experiment(filename, data_params, eval_matrix_generator):
    for data_param in dict_product(data_params):
        for seed in seeds:
            np.random.seed(seed)
            eval_matrix = eval_matrix_generator(**data_param)
            for race_type in race_types:
                for rho in rhos:
                    if race_type != 'DeleteWorst':
                        rho = None
                    race_params = {'seed': seed,
                                   'race_type' : race_type,
                                   'rho' : rho,
                                   **data_param}
                    print(race_params)
                    ranks, stats = run_race(eval_matrix, race_type = race_type, rho = rho)
                    print("Run time:", stats['time'], "sec" )
                    ranks_table.append(ranks)
                    race_params_table.append(race_params)
                    stats_table.append(merge_dicts(race_params, stats))
                    if rho == None: break
                        
    ranks_df = pd.DataFrame(race_params_table).join(pd.DataFrame(ranks_table).fillna(method='ffill', axis=1))
    stats_df = pd.DataFrame(stats_table)
    ranks_df.to_csv(filename + '_ranks.csv')
    stats_df.to_csv(filename + '_stats.csv')
    return ranks_df, stats_df

def acotsp_csv(no_cand):
    return choose_from_cvs(no_cand, "ACOTSP-RUE-1k.csv")
def acotspvar_csv(no_cand):
    return choose_from_cvs(no_cand, "ACOTSP-VAR-TSP3000-10K-Anytime.csv")
def spear_csv(no_cand):
    return choose_from_cvs(no_cand, "spear-train.csv")

def choose_from_cvs(no_cand, filename):
    eval_matrix = np.genfromtxt(filename, delimiter=",", skip_header=1)
    columns = np.random.choice(eval_matrix.shape[1], no_cand)
    eval_matrix = eval_matrix[:, columns]
    means = np.mean(eval_matrix, axis = 0)
    eval_matrix = eval_matrix[:, np.argsort(means)]
    return eval_matrix

race_types = [ "DeleteWorst",
               "Hoeffding", "Bernstein", "Bayesian",
               "BlockingHoeffding", "BlockingBernstein", "BlockingBayesian",
               "FRaceT1", "FRaceT2",
               "AnovaRace", "BlockingAnovaRace"]
race_params_table = []
ranks_table = []
stats_table = []
rhos = [0.1, 0.2]
nreps = 100
seeds = np.random.randint(1, 2**30, size = nreps)

no_cands = [16, 64, 256]
data_params = { 'no_cand' : no_cands,
                'max_task_cand' : [15],
                'cand_diff' : [0.5],
                'sigma_task' : [1],
                'sigma_eval' : [1],
                'eval_error_type' : ["Normal", "ExponentialZeroMean"],
                'task_error_type' : ["Normal", "ExponentialZeroMean"] }


ranks_df, stats_df = run_race_experiment(data_params, gen_simulation_matrix)
ranks_df, stats_df = run_race_experiment({ 'no_cand' : no_cands}, acotsp_csv)
ranks_df, stats_df = run_race_experiment({ 'no_cand' : no_cands}, acotspvar_csv)
ranks_df, stats_df = run_race_experiment({ 'no_cand' : no_cands}, spear_csv)
