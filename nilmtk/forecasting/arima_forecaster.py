import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from . import Forecaster

class ArimaForecasterModel(object):
    params = {
        # The number of lag observations included in the model, also called the lag order.
        'p': 3,
        #The number of times that the raw observations are differenced, also called the degree of differencing.
        'd': 2,
        #The size of the moving average window, also called the order of moving average
        'q': 1 
    }

    model = None

class ArimaForecaster(Forecaster):
    """This is a forecaster based on the
    popular AutoRegressive Integrated Moving Average
    https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
    """

    model_class = ArimaForecasterModel

    def __init__(self, model = None):
        """
        Constructor of this class which takes an optional model as input.
        If no model is given, it createsa default one.
        """
        super(ArimaForecaster, self).__init__(model)

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
        self.model.model = ARIMA(series, order=(params.p,params.params.d,params.p))
        model_fit = model.fit(disp=0)
        
        if verbose:
            # plot summary and residual errors
            print(model_fit.summary())
            residuals = DataFrame(model_fit.resid)
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
