from __future__ import print_function, division
from datetime import datetime
from nilmtk.timeframe import merge_timeframes, TimeFrame
from nilmtk.disaggregate import Disaggregator, DisaggregatorModel


class UnsupervisedDisaggregatorModel(DisaggregatorModel):
    
    '''
    Amount of meters detected within the trainingdata
    '''
    num_meters = None,
    
    '''
    This is the model used for the UnsupevisedDisaggregator.
    It contains the learned model data as well as the 
    paramterization of how it has been created. In contrast to 
    the other models, the all unsupervised models are always 
    anonymous, that means no ids of meters are stored.
    '''
    def extendModels(otherModel):
        '''
        For unsupervised learning, the extend function for the models 
        is even more important as for the supervised/transfer case,
        because each data to disaggregate can be seen as new 
        training data.        
        '''
        raise NotImplementedError("Not yet created!")



class UnsupervisedDisaggregator(Disaggregator):
    """Provides a common interface to all disaggregation classes.

    See https://github.com/nilmtk/nilmtk/issues/271 for discussion, and
    nilmtk/docs/manual/development_guide/writing_a_disaggregation_algorithm.md
    for the development guide.

    Attributes
    ----------
    model :
        Each subclass should internally store models learned from training.

    MODEL_NAME : string
        A short name for this type of model.
        e.g. 'CO' for combinatorial optimisation.
        Used by self._save_metadata_for_disaggregation.
    """
    pass