import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from . import Forecaster
from nilmtk import DataSet, ExternDataSet, TimeFrame, TimeFrameGroup
import pickle as pckl

class ReferenceForecasterModel(object):
    params = {
        # The strategy to use: ["last_day_same_time", "last_week_same_time", "keep_constant"]
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

    def train(self, meters, verbose = False):
        section = TimeFrame(start=pd.Timestamp("1.1.2016", tz = 'UTC'), end = pd.Timestamp("30.05.2016", tz = 'UTC'))
        #powerflow = meters.power_series_all_data(verbose = True, sample_period=3600, sections=sections).dropna()
        powerflow = pckl.load(open("./ForecastingBenchmark.pckl", "rb"))        
        prediction_error = (powerflow - powerflow.shift(24)) / (powerflow)

    def forecast(self):
        pass     