import numpy as np
class HumanLearner:
    """ Model of Human Learner.
    Learner has a list of training points each with a radius and label.
    Learner follows the radius nearest neighbor assumption.
    """
    def __init__(self, kernel):
        '''
        Args:
            kernel: function that takes two inputs and returns a similarity
        '''
        self.teaching_set = []
        self.kernel = kernel

    def predict(self, xs, prior_rejector_preds, to_print = False):
        '''
        Args:
            xs: x points 
            prior_rejector_preds: predictions of prior rejector
        Return:
            preds: posterior human learner rejector predictions
        '''
        preds = []
        idx = 0
        used_posterior = 0 
        for x in xs:
            ball_at_x = []
            similarities = self.kernel(x.reshape(1,-1), np.asarray([self.teaching_set[kk][0] for kk in range(len(self.teaching_set))]))[0]
            for i in range(len(self.teaching_set)):
                similarity = similarities[i]
                if similarity >=  self.teaching_set[i][2]:
                    ball_at_x.append(self.teaching_set[i])
            if len(ball_at_x) == 0: 
                preds.append(prior_rejector_preds[idx])
            else:
                used_posterior += 1
                ball_similarities = self.kernel(x.reshape(1,-1), np.asarray([ball_at_x[kk][0] for kk in range(len(ball_at_x))]))[0]
                normalization = np.sum([ball_similarities[i] for i in range(len(ball_at_x))])
                score_one = np.sum([ball_similarities[i]*ball_at_x[i][1] for i in range(len(ball_at_x))])
                pred = score_one / normalization
                if pred >= 0.5:
                    preds.append(1)
                else:
                    preds.append(0)
            idx += 1
        
        return preds

    def add_to_teaching(self, teaching_example):
        '''
        adds teaching_example to training set
        args:
            teaching_example: (x, label, gamma)
        '''
        self.teaching_set.append(teaching_example)

    def remove_last_teaching_item(self):
        """ removes last placed teaching example from training set"""
        self.teaching_set = self.teaching_set[:-1]
