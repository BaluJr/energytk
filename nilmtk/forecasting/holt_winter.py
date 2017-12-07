from __future__ import division
from sys import exit
from math import sqrt
from numpy import array
from scipy.optimize import fmin_l_bfgs_b

class HoltWinterModel(object):
    params = {
        alpha: None, 
        beta: None, 
        gamma: None
    }

class HoltWinterForecaster(Forecaster):
    """ Forecaster based on popular Holt-Winter exponential smoothing.
    
    Core originally coded in Python 2 by: Andre Queiroz
    Description: This module contains three exponential smoothing algorithms. They are Holt's linear trend method and Holt-Winters seasonal methods (additive and multiplicative).
    https://gist.github.com/andrequeiroz/5888967    
    References:
    Hyndman, R. J.; Athanasopoulos, G. (2013) Forecasting: principles and practice. http://otexts.com/fpp/. Accessed on 07/03/2013.
    Byrd, R. H.; Lu, P.; Nocedal, J. A Limited Memory Algorithm for Bound Constrained Optimization, (1995), SIAM Journal on Scientific and Statistical Computing, 16, 5, pp. 1190-1208.
    """

    model_class = ArimaForecasterModel

    def __init__(self, model = None):
        """
        Constructor of this class which takes an optional model as input.
        If no model is given, it createsa default one.
        """
        super(HoltWinterForecaster, self).__init__(model)




