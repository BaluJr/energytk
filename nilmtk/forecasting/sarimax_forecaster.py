import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from . import Forecaster
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from nilmtk import TimeFrame, TimeFrameGroup
import pickle as pckl

#https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/

class SarimaxForecasterModel(object):
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

        # Die externen Daten werden direkt bei fit mit reingegeben
        ## The feature, which is used from the external data
        #'ext_data': 'temperature',  
        ## The external, which is used to create the vectors for each element, DataSet
        #'ext_data_dataset': "C:\\Users\\maxim_000\\Documents\\InformatikStudium_ETH\\Masterarbeit\\6_Data\\Tmp\\ExtData.hdf",
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
        """
        super(SarimaxForecaster, self).__init__(model)


    def train(self, meters, extDataSet, verbose = False):
        params = self.model.params
        
        # 1. Load the data
        timeframe = TimeFrame(start=pd.Timestamp("1.1.2016", tz = 'UTC'), end = pd.Timestamp("15.03.2017", tz = 'UTC'))
        sections = TimeFrameGroup([TimeFrame(start=pd.Timestamp("1.1.2016", tz = 'UTC'), end = pd.Timestamp("15.03.2017", tz = 'UTC'))]) #30.05.2016
        #powerflow = meters.power_series_all_data(verbose = True, sample_period=900, sections=sections).dropna()
        #pckl.dump(powerflow, open("./ForecastingBenchmark15min.pckl", "wb"))
        #return

        # Kommt von 820
        powerflow = pckl.load(open("./ForecastingBenchmark15min.pckl", "rb"))
        learn = powerflow[-1920:-96]
        extData = extDataSet.get_data_for_group('820', timeframe, 60*60, [('temperature','')])

        # Load the external data specified in the params
        #periodsExtData = [int(dev['sample_period']) for dev in extDataSet.metadata['meter_devices'].values()]
        #min_sampling_period = min(periodsExtData + [self.model.params['self_corr_freq']]) * 2

        # 2. Make data stationary   


        # 3. Define the best paramters


        # 4. Fit the model
        self.model.model_sarimax = model = SARIMAX(endog = learn, order=(params['p'],params['d'],params['q']), 
                                                   seasonal_order = (params['P'],params['D'],params['Q'], params['S']),
                                                   enforce_stationarity = False, enforce_invertibility = False)#, exog = extData[-960:-96])
        model_fit = model.fit(disp=True)
        #self.model.model_arima = model = ARIMA(endog = learn.values, order=(params['p'],params['d'],params['q']), exog = extData[1:-96])
        #model_fit2 = model.fit(disp=True)

        # Do a forecast for so far unknown region and scale back
        forecast = model_fit.predict(start=len(powerflow)-96, end=len(powerflow), dynamic=True)
        #forecast2 = model_fit2.predict(start=len(powerflow)-96, end=len(powerflow), dynamic=True)
        
        # Plot the forecast
        series_to_plot = pd.concat([powerflow, forecast], axis = 1).fillna(0)
        series_to_plot.plot()
        pyplot.show()
        i = abs['tst']

        # Plot residual errors
        residuals =pd.DataFrame(model_fit.resid)
        residuals.plot()
        pyplot.show()
        residuals.plot(kind='kde')
        pyplot.show()
        print(residuals.describe())
        
        # Print the summary 
        print(model_fit.summary())
        print("############ ############ ############ ############ ############")
        print(model_fit2.summary())

        

    def forecast(self):
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
