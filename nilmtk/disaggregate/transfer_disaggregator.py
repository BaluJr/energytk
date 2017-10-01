from __future__ import print_function, division
from datetime import datetime
from nilmtk.timeframe import merge_timeframes, TimeFrame
from nilmtk.disaggregate import Disaggregator, DisaggregatorModel

class TransferDisaggregatorModel(DisaggregatorModel):
    '''
    This is the model used for the TransferDisaggregator.
    It contains the learned model data as well as the 
    paramterization of how it has been created.
    '''
    pass


class TransferDisaggregator(Disaggregator):
    """Provides a common interface for all disaggregators based on transfer learning.
    This has the difference towards supervised learning, that the disaggregated data
    to learn on does not have to be from the same meters, which are later disaggregated.

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