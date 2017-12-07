from __future__ import print_function, division
from datetime import datetime
from nilmtk.timeframe import merge_timeframes, TimeFrame
from nilmtk.elecmeter import ElecMeter
from nilmtk.processing import Processing
import pandas as pd 
import cntk as C


class ForecasterModel(object):
    '''
    As for the disaggregators this model contains the paramters and 
    the models. Additional attributes are defined in the model subclasses.
    '''
    parameters = {}


class Forecaster(Processing):
    """ Provides the baseclass for all forecasting classes.
    It takes Elecmeter or a metergroup as input and returns a 
    elecmeter with values for a given timeframe in the future.

    Attributes
    ----------
    model :
        Each subclass should internally store models learned from training.
        For ANN approaches eg. it would be the parameters for the ANN

    MODEL_NAME : string
        A short name for this type of model.
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
        if model == None:
            model = self.model_class();
        self.model = model;


        
    def mape(self, z, l):
        ''' Small helpfunction implementing mape.
        Used as an error metric during optimization.
        
        Parameters
        ----------
        z: vector<float>
            prediction
        l: vector<float>
            label
        
        Returns
        -------
        errors:
            mape
        '''
        return C.reduce_mean(C.abs(z - l)/l) 
    

    def mae(self, z, l):
        ''' Small helpfunction implementing mae.
        Used as an error metric during optimization.
        
        Parameters
        ----------
        z: vector<float>
            prediction
        l: vector<float>
            label
        
        Returns
        -------
        errors:
            mape
        '''
        return C.reduce_mean(C.abs(z - l))



    #region Data augmentation help functions

    def _addShiftsToChunkAndReduceToValid(self, chunk, past_shifts, model_horizons):
        '''Add shifts to the chunk.
        This function takes the current chunk of the meter and adapts it sothat
        it can be used for training. That means it extends the dataframe by the 
        missing features we want to learn.
        Everything done in memory. Not out of memory computation.

        Paramters
        ---------
        chunk: pd.DataFrame
            The input which have to be augmented 
        past_shifts: [int,...]
            The set of shifts which have to be incorporated from the past due to the 
            history, which are incorporated into the forecast.
        model_horizons:
            The horizons into the future which are trained. Also influences the shifts 
            which have to be prepared.

        Returns
        -------
        chunk: pd.DataFrame
            The input chunk augmented by the fields given in weekday_features 
            and hour_features.
        '''
        # Determine the shifts that are required
        chunk = chunk.copy()
        all_shifts = set()
        for shift in shifts:
            all_shifts.update(range(shift, shift + model_horizons))

        # Create the shifts and return
        for i in all_shifts:
            chunk[('shifts', str(i))] = chunk[('power','active')].shift(i)
        return chunk.drop(chunk.index[chunk[('shifts',str(max(all_shifts)))].isnull()])
           
    

    def _addTimeRelatedFeatures(self, chunk, weekday_features, hour_features):
        ''' Add the time related features.
        Todo: one could also include the day of year when using longer training
        periods.

        Paramters
        ---------
        chunk: pd.DataFrame
            The input which have to be augmented 
        weekday_features: [indexes, ...]
            For the layout of the indexes have a look for the weekday_features in the 
            parameters of lstm_forecaster.
        hour_features: [indexes, ...]
            For the layout of the indexes have a look for the hour_features in the 
            parameters of lstm_forecaster.

        Returns
        -------
        chunk: pd.DataFrame
            The input chunk augmented by the fields given in weekday_features 
            and hour_features.
        '''
        chunk = chunk.copy()
        idxs = weekday_features
        for idx in idxs:
            weekday = idx[1]
            days_of_group = set(range(int(weekday[0]),int(weekday[2])))
            chunk[idx] = chunk.index.weekday
            chunk[idx] = chunk[idx].apply(lambda e, dog=days_of_group: e in dog)
        idxs = hour_features     #self.model.params['hour_features']
        for idx in idxs:
            hour = idx[1]
            hours_of_group = set(range(int(hour[:2]),int(hour[3:])))
            chunk[idx] = chunk.index.hour
            chunk[idx] = chunk[idx].apply(lambda e, hog = hours_of_group: e in hog)
        return chunk



    def _addExternalData(self, chunk, ext_dataset, section, external_features):
        '''
        Currently coming from 820 (for all the meters I do consider)
    
        Paramters
        ---------
        chunk: pd.DataFrame
            The input which have to be augmented 
        ext_dataset: nilmtk.Dataset
            The Dataset, where the external Data can be found.
        section: nilmtk.Timeframe
            The timeframe for which the data shall be retrieved.
        external_features: [indixes,... ]
            The indexes which shall be retrieved.

        Returns
        -------
        chunk: pd.DataFrame
            The input chunk extended by the features given in 
            external_features.
        '''
        if len(external_features) > 0: 
            extData = ext_dataset.get_data_for_group('820', section, 60*15, external_features)[1:]
        return pd.concat([chunk, extData], axis=1)


    #endregion