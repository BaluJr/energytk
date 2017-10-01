from __future__ import print_function, division
from datetime import datetime
from nilmtk.timeframe import merge_timeframes, TimeFrame
from nilmtk.disaggregate import Disaggregator, DisaggregatorModel


class SupervisedDisaggregatorModel(DisaggregatorModel):
    '''
    This is the model used for the SupevisedDisaggregator.
    It contains the learned model data as well as the 
    paramterization of how it has been created.
    '''
    pass


class SupervisedDisaggregator(Disaggregator):
    """ Provides a common interface to all supoervised disaggregation classes.
    Supervised algorithms include a training function.

    Attributes
    ----------
    model :
        Each subclass should internally store models learned from training.

    MODEL_NAME : string
        A short name for this type of model.
        e.g. 'CO' for combinatorial optimisation.
        Used by self._save_metadata_for_disaggregation.
    """

    def train(self, appliance_meters):
        """Trains the model given a metergroup containing appliance meters
        (supervised) or a site meter (unsupervised).  Will have a
        default implementation in super class.  Can be overridden for
        simpler in-memory training, or more complex out-of-core
        training.

        Parameters
        ----------
        metergroup : a nilmtk.MeterGroup object
        """
        _pre_training_checks(appliance_meters)
        raise NotImplementedError()

    def train_on_chunk(self, chunk, meter):
        """Signature is fine for site meter dataframes (unsupervised
        learning). Would need to be called for each appliance meter
        along with appliance identifier for supervised learning.
        Required to be overridden to provide out-of-core
        disaggregation.

        Parameters
        ----------
        chunk : pd.DataFrame where each column represents a
            disaggregated appliance
        meter : ElecMeter for this chunk
        """
        raise NotImplementedError()

    
    def import_model(self, filename):
        """Loads learned model from file.
        Required to be overridden for learned models to persist.

        Parameters
        ----------
        filename : str path to file to load model from
        """
        raise NotImplementedError()

    def export_model(self, filename):
        """Saves learned model to file.
        Required to be overridden for learned models to persist.

        Parameters
        ----------
        filename : str path to file to save model to
        """
        raise NotImplementedError()

    
    def _pre_disaggregation_checks(self, site_meter, load_kwargs):
        '''
        This is the basic check, which is called before disaggregation is performed.
        It takes care.
        '''
        if not self.model:
            raise RuntimeError(
                "The model needs to be instantiated before"
                " calling `disaggregate`.  For example, the"
                " model can be instantiated by running `train`.")
        
        Disaggregator._pre_disaggregation_checks(self, site_meter, load_kwargs)
        
        return load_kwargs

    
    def _pre_training_checks(self, meters, load_kwargs):
        '''
        This is the basic check, which is called before disaggregation is performed.
        It takes care.
        '''
        
        for meter in meters.all_meters:
            Disaggregator._check_meter(self, meter)
        
        return load_kwargs