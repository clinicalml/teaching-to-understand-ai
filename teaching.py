import numpy as np
import math
import random
from tqdm import tqdm
from collections import Counter
from human_learner import HumanLearner
from metrics_hai import compute_metrics

class TeacherExplainer():
    """ Returns top examples that best teach a learner when to defer to a classifier.
    Given a tabular dataset with classifier predictions, human predictions and a similarity metric,
    the method returns the top k images that best describe when to defer to the AI.    
     """

    def __init__(self,
                data_x,
                data_y,
                hum_preds,
                ai_preds,
                prior_rejector_preds,
                sim_kernel,
                metric_y,
                alpha = 1,
                teaching_points = 10):
            """Init function.
            Args:
                data_x: 2d numpy array of the features
                data_y: 1d numpy array of labels
                hum_preds:  1d array of the human predictions 
                ai_preds:  1d array of the AI predictions 
                prior_rejector_preds: 1d binary array of the prior rejector preds 
                sim_kernel: function that takes as input two inputs and returns a positive number
                metric_y: metric function (positive, the higher the better) between predictions and ground truths, must behave like rbf_kernel from sklearn
                alpha: parameter of selection algorithm, 0 for double greedy and 1 for consistent radius
                teaching_points: number of teaching points to return
            """
            self.data_x = data_x
            self.data_y = data_y
            self.hum_preds = hum_preds
            self.data_y = data_y
            self.ai_preds = ai_preds
            self.kernel = sim_kernel
            self.prior_rejector_preds = prior_rejector_preds
            self.metric_y = metric_y
            self.alpha = alpha
            self.teaching_points = teaching_points

    def get_optimal_deferral(self):
        '''
        gets optimal deferral decisions computed emperically
        Return:
            opt_defer: optimal deferral decisions 

        '''
        opt_defer_teaching = []
        for ex in range(len(self.hum_preds)):
            score_hum = self.metric_y([self.data_y[ex]], [self.hum_preds[ex]])
            score_ai = self.metric_y([self.data_y[ex]], [self.ai_preds[ex]])
            if score_hum >= score_ai:
                opt_defer_teaching.append(0)
            else:
                opt_defer_teaching.append(1)
        self.opt_defer = np.array(opt_defer_teaching)
        return np.array(opt_defer_teaching)


    def get_optimal_consistent_gammas(self):
        '''
        get optimal consistent gammas
        Return:
            optimal_consistent_gammas: array of optimal consistent gamma values
        '''
        optimal_consistent_gammas = []
        with tqdm(total=len(self.data_x)) as pbar:
            similarities_embeds_all = self.kernel( np.asarray(self.data_x), np.asarray(self.data_x)) # kernel matrix
            self.similarities_embeds_all = similarities_embeds_all # save kernel matrix
            for i in range(len(self.data_x)):
                # get all similarities
                similarities_embeds = similarities_embeds_all[i]
                opt_defer_ex = self.opt_defer[i]
                opt_gamma = 1
                sorted_sim = sorted([(similarities_embeds[k], self.opt_defer[k]) for k in range(len(self.data_x))], key=lambda tup: tup[0])
                indicess = list(range(1, len(self.opt_defer)))
                indicess.reverse()
                for k in indicess:
                    if sorted_sim[k][1] == opt_defer_ex and sorted_sim[k- 1][1] != opt_defer_ex:
                        opt_gamma = sorted_sim[k][0]
                        break
                optimal_consistent_gammas.append(opt_gamma)
                pbar.update(1)
        self.optimal_consistent_gammas = np.array(optimal_consistent_gammas)
        return np.array(optimal_consistent_gammas)

    def get_improvement_defer_consistent(self, current_defer_preds):
        '''
        Gets how much would score improve in terms of metric_y if you add each point to the teaching set 
        Assumption: assumes metric_y over all data is decomposed as the average of metric_y for each data point e.g. 0-1 loss
        Relaxation: instead of simulating human learner, we assume if point added to the human's set, human will follow the points optimal decision
        Note: for the consistent gamma strategy, the relaxation does not affect the result
        Args:
            current_defer_preds: current predictions of human learner on teaching set
        Return:
            error_improvements: improvement of adding point i for each i in data to the human training set
        '''
        error_improvements = []
        error_at_i = 0
        for i in range(len(self.data_x)):
            error_at_i = 0
            similarities_embeds = self.similarities_embeds_all[i]
            # get the ball for x
            # in this ball how many does the current defer not match the optimal
            for j in range(len(similarities_embeds)):
                if similarities_embeds[j] >= self.optimal_consistent_gammas[i]:
                    score_hum = self.metric_y([self.data_y[j]], [self.hum_preds[j]])
                    score_ai = self.metric_y([self.data_y[j]], [self.ai_preds[j]])
                    if self.opt_defer[i] == 1:
                        if current_defer_preds[j] == 0:
                            error_at_i += score_ai - score_hum
                    else:
                        if current_defer_preds[j] == 1:
                            error_at_i += score_hum - score_ai
            error_improvements.append(error_at_i)

        return error_improvements


    def teach_consistent(self, to_print = False, plotting_interval = 2):
        '''
        our greedy consistent selection algorithm, updates human learner
        returns:
            errors: training errors after adding each teaching point
            indices_used: indices used for teaching 
        '''
        errors = []
        data_sizes  = []
        indices_used = []
        points_chosen = []
        plotting_interval = plotting_interval # plotting interval
        for itt in tqdm(range(self.teaching_points)):
            best_index = -1
            # predict with current human learner
            if itt == 0:
                preds_teach = self.prior_rejector_preds
            else:
                preds_teach = self.human_learner.predict(self.data_x, self.prior_rejector_preds)
            # get improvements for each point if added
            error_improvements = self.get_improvement_defer_consistent(preds_teach)
            # pick best point and add it
            best_index = np.argmax(error_improvements)
            indices_used.append(best_index) # add found element to set used
            ex_embed = self.data_x[best_index]
            ex_label = self.opt_defer[best_index]
            gamma = self.optimal_consistent_gammas[best_index]
            if to_print:
                print(f'got improvements with max {max(error_improvements)}')
            self.human_learner.add_to_teaching([ex_embed, ex_label, gamma])

            # evaluate on teaching points
            if to_print and itt % plotting_interval == 0:
                print("####### train eval " + str(itt)+ " ###########")
                preds_teach = self.human_learner.predict(self.data_x, self.prior_rejector_preds)
                _, metricss, __, ___ = compute_metrics(self.hum_preds, self.ai_preds, preds_teach, self.data_y, self.metric_y, to_print)
                errors.append(metricss['score'])   
                print("##############################")
            elif itt % plotting_interval:
                _, metricss, __, ___ = compute_metrics(self.hum_preds, self.ai_preds, preds_teach, self.data_y, self.metric_y, False)
                errors.append(metricss['score'])   
        return errors, indices_used


    
    def get_improvement_defer_greedy(self, current_defer_preds, seen_indices):
        '''
        Gets how much would score improve in terms of metric_y if you add each point and optimize radius to the teaching set 
        Assumption: assumes metric_y over all data is decomposed as the average of metric_y for each data point e.g. 0-1 loss
        Relaxation: instead of simulating human learner, we assume if point added to the human's set, human will follow the points optimal decision
        Args:
            current_defer_preds: current predictions of human learner on teaching set
            seen_indices: current predictions of human learner on teaching set

        Return:
            error_improvements: improvement of adding point i for each i in data to the human training set
            found_gammas: optimal gammas found for each point
        '''
        error_improvements = []
        error_at_i = 0
        found_gammas = []
        indicess = list(range(1, len(self.data_x) -1 ))
        indicess.reverse()
        for i in range(len(self.data_x)):
            similarities_embeds = self.similarities_embeds_all[i]
            sorted_sim = self.sorted_sims[i] 

            max_improve = -1000
            gamma_value = self.optimal_consistent_gammas[i]
            current_improve = 0
            so_far = 0
            for j in indicess:
                if i in seen_indices:
                    continue
                so_far += 1
                idx = int(sorted_sim[j][1])
                score_hum = self.metric_y([self.data_y[idx]], [self.hum_preds[idx]])
                score_ai = self.metric_y([self.data_y[idx]], [self.ai_preds[idx]])
                if self.opt_defer[i] == 1:
                    if current_defer_preds[idx] == 0:
                        current_improve += score_ai - score_hum
                else:
                    if current_defer_preds[idx] == 1:
                        current_improve += score_hum - score_ai

                if current_improve >= max_improve:
                    max_improve = current_improve 
                    gamma_value = min(self.optimal_consistent_gammas[i], sorted_sim[j][0] )

            error_improvements.append(max_improve)
            found_gammas.append(gamma_value)
        return error_improvements, found_gammas


    
    def teach_doublegreedy(self, to_print = False, plotting_interval = 2):
        '''
        our double greedy selection algorithm, updates human learner
        returns:
            errors: training errors after adding each teaching point
            indices_used: indices used for teaching 
        '''
        sorted_sims = []
        for i in range(len(self.similarities_embeds_all)):
            sorted_sim = sorted([(self.similarities_embeds_all[i][k], k) for k in range(len(self.data_x))], key=lambda tup: tup[0])
            sorted_sims.append(np.asarray(sorted_sim))
        self.sorted_sims = sorted_sims
        errors = []
        data_sizes  = []
        indices_used = []
        points_chosen = []
        found_gammas = []
        plotting_interval = plotting_interval # plotting interval
        for itt in tqdm(range(self.teaching_points)):
            best_index = -1
            # predict with current human learner
            if itt == 0:
                preds_teach = self.prior_rejector_preds
            else:
                preds_teach = self.human_learner.predict(self.data_x, self.prior_rejector_preds)
                
            # get improvements for each point if added
            error_improvements, best_gammas = self.get_improvement_defer_greedy(preds_teach, indices_used)
            # pick best point and add it
            best_index = np.argmax(error_improvements)
            indices_used.append(best_index) # add found element to set used
            ex_embed = self.data_x[best_index]
            ex_label = self.opt_defer[best_index]
            gamma = best_gammas[best_index]
            found_gammas.append(gamma)
            if to_print:
                print(f'got improvements with max {max(error_improvements)}')
            self.human_learner.add_to_teaching([ex_embed, ex_label, gamma])

            # evaluate on teaching points
            if to_print and itt % plotting_interval == 0:
                print("####### train eval " + str(itt)+ " ###########")
                preds_teach = self.human_learner.predict(self.data_x, self.prior_rejector_preds)
                _, metricss, __, ___ = compute_metrics(self.hum_preds, self.ai_preds, preds_teach, self.data_y, self.metric_y, to_print)
                errors.append(metricss['score'])   
                print("##############################")
            elif itt % plotting_interval:
                _, metricss, __, ___ = compute_metrics(self.hum_preds, self.ai_preds, preds_teach, self.data_y, self.metric_y, False)
                errors.append(metricss['score'])   
        return errors, indices_used, np.array(found_gammas)
    

    def get_teaching_examples(self, to_print = False, plot_interval = 2):
        """ obtains teaching points. Currently only implemented for consistent strategy
        Args:
            to_print: display details of teaching process
            plot_interval: how often to plot results
        Return:
            teaching_x: 2d numpy array of teaching points features
            teaching_indices: indices of the teaching points in self.data_x
            teaching_gammas: 1d numpy of gamma values used
            teaching_labels: 1d array of deferral labels where 1 signifies defer to AI and 0 signifies don't defer to AI
        
        """
        if self.alpha not in [0,1]:
            assert self.alpha != 1, "Only consistent or double-greedy strategy implemented with alpha =1 or =0"

        # run algorithm to get examples
        # get optimal deferrall points
        print("getting gammas and optimal deferral decisions on teaching set")
        self.get_optimal_deferral()
        # get consistentg
        self.get_optimal_consistent_gammas()
        self.human_learner = HumanLearner(self.kernel)
        
        if self.alpha == 1:
            print("starting the teaching process with consistent radius ...")
            errors, indices_used = self.teach_consistent(to_print, plot_interval)
            teaching_x = self.data_x[indices_used]
            teaching_indices = indices_used
            teaching_gammas = self.optimal_consistent_gammas[indices_used]
            teaching_labels = self.opt_defer[indices_used]
        else:
            print("starting the teaching process with greedy radius ...")
            errors, indices_used, best_gammas = self.teach_doublegreedy(to_print, plot_interval)
            teaching_x = self.data_x[indices_used]
            teaching_indices = indices_used
            teaching_gammas = best_gammas
            teaching_labels = self.opt_defer[indices_used]
        print("got teaching points")
        return teaching_x, teaching_gammas, teaching_labels, teaching_indices

    





    
    
