import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from . import Forecaster
from nilmtk import DataSet, ExternDataSet, TimeFrame, TimeFrameGroup
import pickle as pckl

class ReferenceForecasterModel(object):
    params = {
        # The number of lag observations included in the model, also called the lag order.
        'p': 3,
        #The number of times that the raw observations are differenced, also called the degree of differencing.
        'd': 2,
        #The size of the moving average window, also called the order of moving average
        'q': 1 
    }

    model = None

class ReferenceForecaster(Forecaster):
    """ This is a stupid forecaster which follows fixed 
    strategies. Last day same time.
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