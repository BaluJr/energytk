import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from matplotlib import pyplot
from . import Forecaster
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from nilmtk import TimeFrame, TimeFrameGroup
import pickle as pckl
from sklearn.preprocessing import StandardScaler

class SarimaxForecasterModel(object):
    '''
    This is the model used for forecasting. It is an enumerable with one set of the below attributes
    for each entry. Each entry represents one forecasting horizon.
    #https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
    #http://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/

    Attributes
    ----------
    model_sarimax: sstatsmodels.tsa.statespace.sarimax.SARIMAX
        The SARIMAX model used for forecasting
    '''

    params = {
        # The number of lag observations included in the model, also called the lag order.
        'p': 2,
        #The number of times that the raw observations are differenced, also called the degree of differencing.
        'd': 1,
        #The size of the moving average window, also called the order of moving average
        'q': 3,

        # Seasonal
        'P': 2,
        # Seasonal
        'D': 1,       
        # Seasonal
        'Q': 3,       

        # Seasonal
        'S': 96,

        # The feature, which is used from the external data
        'external_feature': ('temperature', ''),  

        # Only the last elements of original_data are used for training (default = 10 days)
        "trainingdata_limit": 960 * 3
    }

    model = None


class SarimaxForecaster(Forecaster):
    """
    This is a forecaster based on Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors:
    https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3
    """

    model_class = SarimaxForecasterModel

    def __init__(self, model = None):
        """
        Constructor of this class which takes an optional model as input.
        If no model is given, it createsa default one.
        
        Paramters
        ---------
        model: Model of type model_class
            The model which shall be used.
        """
        super(SarimaxForecaster, self).__init__(model)


    def _stationary_data(self, load):
        ''' Makes the data stationary by different techniques.
        This is necessary as a preprocessing step to appy ARIMA models 
        afterwards.

        Parameters:
        ----------
        load: pd.Series
            The original load which might be seasonal and not stationary.

        Returns
        -------
        stationary_load:
            The load in its stationary form
        '''
        self.model.standardscaler = StandardScaler()
        load.iloc[:,0] = self.model.standardscaler.fit_transform(load.values)
        return load


    def forecast(self, meters, ext_dataset, horizon = pd.Timedelta('1d'), return_residuals=False, verbose = False):
        '''
        Does the forecasting for an ARIMA model. As the model is only valid 
        for a certain dataset and cannot be generalized, 
        
        Parameters
        ----------
        meters: nilmtk.DataSet or str
            The meters from which the demand is loaded. 
            Alternatively a path to a PickleFile. 
        ext_dataset: nilmtk.DataSet or str
            The External Dataset containing the fitting data.
            Alternatively a path to a PickleFile. 
        horizon: pd.Timedelta (optional)
            The horizon in the future to forecast for.
        return_residuals: bool (optional)
            Whether to return the residuals as additional information
        verbose: bool (optional)
            Whether additional output shall be printed during training.
        '''

        params = self.model.params
        
        # Load and preprocess the data
        chunk = self._load_data(meters)
        chunk = self._stationary_data(chunk)
        chunk = chunk[-params['trainingdata_limit']:]

        ext_chunk, ext_chunk_future = None, None
        if not params['external_feature'] is None:
            ext_chunk = self._add_external_data(chunk.index, ext_dataset, [params['external_feature']], horizon)
            ext_chunk, ext_chunk_future = ext_chunk[:chunk.index[-1]], ext_chunk[chunk.index[-1]:]
            ext_chunk = ext_chunk[-params['trainingdata_limit']:]

        # Fit the model if not already done
        model = SARIMAX(endog = chunk, order=(params['p'],params['d'],params['q']), 
                        seasonal_order = (params['P'],params['D'],params['Q'], params['S']), 
                        enforce_stationarity = False, enforce_invertibility = False)#, exog = ext_chunk)
        self.model.model_sarimax = model
        model_fit = model.fit(disp=verbose)
        if return_residuals:
            residuals = DataFrame(model_fit.resid)

        # Do a forecast for so far unknown region and scale back
        forecast = model_fit.predict(start=len(data), end=len(data)+horizon, dynamic=True, exog = ext_chunk_future)
        
        # Print the summaries if verbose
        if verbose == True:
            print(model_fit.summary())
            if return_residuals:
                print(residuals.describe())

        if return_residuals:
            return forecast, residuals
        else:
            return forecast

        
    def test(self):
        '''
        This is just an exemplanatory test that shows the usage of 
        the SARIMAX model. Can be removed.
        '''

        series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
        X = series.values
        size = int(len(X) * 0.66)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=(5,1,0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            print('predicted=%f, expected=%f' % (yhat, obs))
            error = mean_squared_error(test, predictions)
        print('Test MSE: %.3f' % error)

        # plot
        pyplot.plot(test)
        pyplot.plot(predictions, color='red')
        pyplot.show()
