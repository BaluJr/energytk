from __future__ import print_function, division
from datetime import datetime
from nilmtk.timeframe import merge_timeframes, TimeFrame
from nilmtk.elecmeter import ElecMeter
from nilmtk.processing import Processing


class ClusterModel(object):
    '''
    As for the ther classes this model contains the paramters and 
    the models.
    '''
    #parameters = {}


class Clusterer(Processing):
    """ Baseclass for all clustering classes.

    Clustring is used to cluster metering devices. Different subclasses may introduce 
    different ways to cluster the meters. One option is to cluster the elements only 
    by using the information from the elements themselves. The other opportunity is 
    to give extra information which are included into the clustering process.
    The predictions should return a set of MeterGroups with the smart meters bound together.

    Attributes
    ----------
    model :
        Each subclass should internally store models learned from training.
        For ANN approaches eg. it would be the parameters for the ANN

    MODEL_NAME : string
        A short name for this type of model.
        e.g. 'CO' for combinatorial optimisation.
        Used by self._save_metadata_for_disaggregation.
    """


    '''
    This attribute declares which data is necessary to use the forecaster.
    Whenever a forecasting or training is performed, the dataset is checked 
    for the fullfillment of these requirements
    '''
    Requirements = {
        'max_sample_period': 900,
        'physical_quantities': [['power','active']]
    }

    ''' 
    This attribute has to be overwritten with the 
    corresponding model of the disaggregator.
    '''
    model_class = None
        
    def __init__(self, model):
        '''
        The init function offers the possibility to overwrite the 
        default model by an own model, which can contain own parameters 
        or even already a trained model.

        Paramters
        ---------
        model: Model of type model_class
            The model which shall be used.
        '''
        super(Clusterer, self).__init__()

        if model == None:
            model = self.model_class();
        self.model = model