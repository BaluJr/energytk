class SlpForecasterModel(object):
    '''
    This model class contains the slp models which are used for the calculation.
    '''
    params = {
        # The number of lag observations included in the model, also called the lag order.
        'p': 3,
        #The number of times that the raw observations are differenced, also called the degree of differencing.
        'd': 2,
        #The size of the moving average window, also called the order of moving average
        'q': 1 
    }

class SlpForecaster(object):
    """
    Description of class
    This forecaster works the same as the grid operators. It takes each building and applies a 
    standard load profiles depending on its asset type.
    """
    Requirement = {'building_type':'ANY VALUE'}

    def __init__(self, model):
        super(SlpForecaster, self).__init__(model)