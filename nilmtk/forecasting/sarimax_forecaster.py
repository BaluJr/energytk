import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from . import Forecaster
import matplotlib.pyplot as plt
import requests
from io import BytesIO

class SarimaxForecasterModel(object):
    params = {
        # The number of lag observations included in the model, also called the lag order.
        'p': 1,
        #The number of times that the raw observations are differenced, also called the degree of differencing.
        'd': 1,
        #The size of the moving average window, also called the order of moving average
        'q': 1 ,

        # Seasonal
        'P': 3,
        # Seasonal
        'D': 2,       
        # Seasonal
        'Q': 2,       
        # Seasonal
        'S': 2,

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


    def train(self, meters, verbose = False):
        #powerflow = pd.read_csv('15minSampledLoadProfileForForecast.csv') #meters.power_series_all_data(sample_period=900, verbose = True)
        #powerflow[:96*7].plot()
        #plt.show()
        #powerflow.dropna(inplace = True)
        #powerflow = powerflow[:10000]
        #load = powerflow.iloc[:,1].values
        #extern = powerflow.set_index(['2011-03-21 23:00:00'])
        params = self.model.params

        wpi1 = requests.get('http://www.stata-press.com/data/r12/wpi1.dta').content
        data = pd.read_stata(BytesIO(wpi1))
        data.index = data.t
        #data.plot()
        #plt.show()
        self.model.model = model = SARIMAX(endog = data.wpi, order=(params['p'],params['d'],params['q'])) #exog = extern
        model_fit = model.fit(disp=True)
        
        self.model.model = model = ARIMA(endog = data.wpi, order=(params['p'],params['d'],params['q'])) #exog = extern
        model_fit2 = model.fit(disp=True)

        i = 1
        
        if verbose:
            # plot summary and residual errors
            print(model_fit.summary())
            print("############ ############ ############ ############ ############")
            print(model_fit2.summary())
            residuals =pd.DataFrame(model_fit.resid)
            residuals.plot()
            pyplot.show()
            residuals.plot(kind='kde')
            pyplot.show()
            print(residuals.describe())
        

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
