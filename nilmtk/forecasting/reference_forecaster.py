import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from . import Forecaster
from nilmtk import DataSet, ExternDataSet, TimeFrame, TimeFrameGroup
import pickle as pckl

class ReferenceForecasterModel(object):
    params = {
        # The strategy to use: ["last_day_same_time", "last_week_same_time"]
        'strategy': "last_day_same_time"
    }

    model = None

class ReferenceForecaster(Forecaster):
    """ Very simple forecaster following fixed strategies. 
    This forecaster does the forecasting by following simple rules, as 
    for example the value of the last day of the same type.
    Can be therfore also seen as an expert system.
    """

    model_class = ReferenceForecasterModel

    def __init__(self, model = None):
        """
        Constructor of this class which takes an optional model as input.
        If no model is given, it createsa default one.
        """
        super(ReferenceForecaster, self).__init__(model)



    def _apply_strategy(self, strategy, orginal_load):
        '''
        This method uses the learned model to predict the future
        For each forecaster the forecast horizon is derived from its 
        smallest shift value.
        All meters that contain a powerflow are considered part of the 
        group to forecast.

        Parameters
        ----------
        strategy: str
            The strategy to use. For descriptions see model.params.strategy.
        original_load:
            The original load, which is adapted.
        Returns
        -------
        The adapted original timeframe sothat it fits the chosen strategy.
        '''
        if strategy == "last_day_same_time":
            orginal_load = orginal_load.shift(96)
        elif strategy == "last_week_same_time":
            orginal_load = orginal_load.shift(96*7)
        else:
            raise Exception("Strategy unknown!")
        return orginal_load.dropna()


    def forecast(self, meters, ext_dataset, timestamps, horizon = pd.Timedelta('1d'), resolution = "15m", verbose = False):
        '''
        This method uses the learned model to predict the future
        For each forecaster the forecast horizon is derived from its 
        smallest shift value.
        All meters that contain a powerflow are considered part of the 
        group to forecast.

        Parameters
        ----------
        meters: nilmtk.DataSet
            The meters from which the demand is loaded.
        ext_dataset: nilmtk.DataSet
            The storage with external data
        timestamps: pd.Timestamp, [pd.TimeStamp,...] or pd.DatetimeIndex
            A single point or a list of points for which the forecasting is performed.
            All contained model horizonts are applied to each point in time.
        horizon: pd.Timedelta (optional)
            The horizon in the future to forecast for.
        resolution: str
            A freq_str representing the frequency with which results
            are returned.
        verbose: bool (optional)
            Whether additional output shall be printed during training.

        Returns
        -------
        forecasts: pd.DataFrame
            A DataFrame containing the forecasts for each Timestamp. 
            One column for each timestamp and one row for each forecaster 
            horizon.
        '''
        params = self.model.params
        forecast = pd.DataFrame(columns = timestamps)

        
        # Load the data and apply strategy
        chunk = self._load_data(meters)
        chunk = self._apply_strategy(params['strategy'], chunk)

        # Forecast
        for timestamp in timestamps:
            forecast[timestamp] = chunk[timestamp: timestamp + horizon].reset_index(drop=True)
            
        forecast['horizon'] = forecast.index.values * pd.Timedelta(resolution)
        forecast = forecast.set_index('horizon')
        return forecast