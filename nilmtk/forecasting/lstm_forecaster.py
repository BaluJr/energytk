from matplotlib import pyplot as plt
import math
import numpy as np
import os
import pandas as pd
import random
import time
from cntk.debugging import debug_model
import pickle as pckl
from .forecaster import Forecaster
from nilmtk import DataSet, ExternDataSet, TimeFrame, TimeFrameGroup
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections.abc import Sequence
import copy

import cntk as C
from cntk.layers import Sequential
from cntk.layers.typing import Tensor, Sequence

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve



class LstmForecasterModel(Sequence):
    params = {
        # The target storage to store the data to
        'forecasting_storage': "C:\\Users\\maxim_000\\Documents\\InformatikStudium_ETH\\Masterarbeit\\6_Data\\Tmp\\forecasting.hdf",
        
        # How many timesteps into the future we predict
        'models': [192],

        # Learning Rate
        'learning_rate': 0.01,

        # How many errors are taken together until training step
        'size_minibatches': 10,
        
        # Whether to use (100 for is_fast, else 2000)
        'epochs': 100,

        # Maximum value sothat we devide by this
        'normalize': 30000,

        # Dimensionality of LSTM cell
        'h_dim': 14,

        # The hidden dimensionality of the LSTM-NN
        'lstm_dim': 25,
        
        # 10 Tage auf einmal
        'batch_size': 14 * 10,

        # How far to the past the network looks
        'horizon': 92,

        # The features which are used as input
        'externalFeatures': [('temperature', '')],#, ('national', ''), ('school', '')],#, ('dewpoint', '')], #, 'time'
        
        # How the daytime is regarded
        'hour_features': [('hour', '00-06'), ('hour', "06-09"), ('hour', "09-12"), ('hour', "12-15"), ('hour', "15-18"), ('hour', "18-21"), ('hour', "21-24")],

        # How the weekdays are paired
        'weekday_features': [('week', "0-5"),('week', "5-6"),('week',"6-7")],

        # Unfortunately debug_model not yet working in CNTK 2.2
        'debug' : False,

        # Amount of data used for validation
        'validation_quota': 0.1,

        # Define output
        'training_progress_output_freq': 40,

        # Forecasting without external data
        'forecast_without_external': False
    }

    
    def __init__(self, **args):
        '''
        The paramters are set to the optional paramters handed in by the constructor.
        '''
        for arg in args:
            self.params[arg] = args[arg]
        self.model = [
            {'trainer':None, 
             'input':None, 
             'labels':None, 
             'plotdata': {"batchnumber":[], "loss":[], "error":[], "avg_power":[]}
            }] * len(self.params['models'])
    

    def set_model(self, i, element, new_model):
        self.model[i] = new_model


    def __getitem__(self, i):
        '''
        Makes it possible to access the data of the i-th submodel.
        key: str
        '''
        return self.model[i]


    def __len__(self):
        return self.model.__len__()


    def plot_training_progress(self, mean = False):
        ''' 
        This method plots the training progress which has been recorded during training.

        The model for which the plot shall be drawn. When set to -1 then all are drawn
        '''
        myrange = range(len(self.params['models']))

        plt.subplot(211)
        plt.figure(1)
        for i in myrange:
            plt.plot(self[i]['plotdata']["batchnumber"], self[i]['plotdata']["loss"], 'r--')
            plt.plot(self[i]['plotdata']["batchnumber"], pd.rolling_mean(pd.Series(self.model[0]['plotdata']["loss"]),20).values, 'b--')
        plt.xlabel('Minibatch number')
        plt.ylabel('Loss')
        plt.title('Minibatch run vs. Training loss ')
        
        plt.subplot(212)
        for i in myrange:
            plt.plot(self[i]['plotdata']["batchnumber"], self[i]['plotdata']["error"], 'r')
            plt.plot(self[i]['plotdata']["batchnumber"], pd.rolling_mean(pd.Series(self.model[0]['plotdata']["error"]),20).values, 'b--')
        plt.xlabel('Minibatch number')
        plt.ylabel('Label Prediction Error')
        plt.title('Minibatch run vs. Label Prediction Error ')

        plt.show()



class LstmForecaster(Forecaster):
    """ Forecaster using a single lstm. The future values are not incorporated.

    This is a forecaster based on a distinct artificial neural network (ANN) 
    for each of the the forecasting distances.

    Attributes
    ----------
    model_class: type  
        The model type, which belonging to the forecaster.
    """
    
    # The related model
    model_class = LstmForecasterModel
    

    def __init__(self, model = None):
        """
        Constructor of this class which takes an optional model as input.
        If no model is given, it creates a default one.

        Paramters
        ---------
        model: Model of type model_class
            The model which shall be used.
        """
        super(LstmForecaster, self).__init__(model)

        
   
    
    def _model_creation(self, modelnumber):
        """ Create the model for time series prediction.
        Since there are multiple models, the parameter defines 
        the number of the model.

        Parameters
        ----------
        modelnumber: int
            Forecasting horizon this model is trained for.
        """
        
        # Define input
        params = self.model.params
        input_dim = 1 + len(params['externalFeatures']) + len(params['hour_features']) + len(params['weekday_features'])
        self.model[modelnumber]['input'] = x = C.sequence.input_variable(shape=input_dim, is_sparse = False)     # Das fuegt glaube sofort schon eine Dyn Achse ein

        # Create the network
        with C.layers.default_options(initial_state = 0.1):
            m = C.layers.Recurrence(C.layers.LSTM(self.model.params['lstm_dim'], self.model.params['h_dim'], name="Recur"))(x)
            m = C.sequence.last(m, name="RecurLast")
            m = C.layers.Dropout(0.2, name="Dropout")(m)
            z = C.layers.Dense(1, name="Dense")(m)
        if self.model.params['debug']:
            z = debug_model(z)

        # Define output
        self.model[modelnumber]['label'] = l = C.input_variable(1, dynamic_axes=z.dynamic_axes, name="y") 
        
        # Define error metric and trainer
        loss = C.squared_error(z, l)
        error = self.mae(z, l)
        lr_schedule = C.learning_rate_schedule(params['learning_rate'], C.UnitType.minibatch)
        momentum_time_constant = C.momentum_as_time_constant_schedule(params['batch_size'] / -math.log(0.9)) 
        learner = C.fsadagrad(z.parameters, lr = lr_schedule, momentum = momentum_time_constant)
        self.model[modelnumber]['trainer']  = C.Trainer(z, (loss, error), [learner])


    def train(self, meters, extDataSet, verbose = False):
        '''
        Does the training of the neural network. Each load chunk is translated into 
        multiple minibatches used for training the network.
        
        Parameters
        ----------
        meters: nilmtk.DataSet
            The meters from which the demand is loaded.
        extDataSet: nilmtk.DataSet
            The External Dataset containing the fitting data.
        verbose: bool
            Whether additional output shall be printed during training.
        '''

        params = self.model.params
        
        # A) Load the data
        section = TimeFrame(start=pd.Timestamp("1.1.2016", tz = 'UTC'), end = pd.Timestamp("15.03.2017", tz = 'UTC'))
        #section = meters.get_timeframe(intersection_instead_union=True)
        sections = TimeFrameGroup([TimeFrame(start=pd.Timestamp("1.1.2016", tz = 'UTC'), end = pd.Timestamp("15.03.2017", tz = 'UTC'))])
        #powerflow = meters.power_series_all_data(verbose = True, sample_period=3600, sections=sections).dropna()
        chunk = pd.DataFrame(pckl.load(open("./ForecastingBenchmark15min.pckl", "rb")))
        
        # B) Prepare the data
        shifts = list(range(params['horizon'], 0, -1))
        chunk = self._addTimeRelatedFeatures(chunk, params['weekday_features'], params['hour_features'])
        chunk = self._addExternalData(chunk, extDataSet, section, self.model.params['externalFeatures'])
        #chunk = self._addShiftsToChunkAndReduceToValid(chunk, shifts, params['models'])
        
        # C) Create the model
        models = self.model.params['models']
        for modelnumber in range(len(models)):
            self._model_creation(modelnumber)

        # D) Arrange the features and labels
        train_test_mask = np.random.rand(len(chunk)) < 0.8
        self.model.featurescaler = scaler = StandardScaler()
        features = chunk.values #.drop([('power','active')], axis = 1)
        #training_features = list(np.expand_dims(training_features,2))
        features = scaler.fit_transform(features)        
        self.model.labelscaler = scaler = StandardScaler()
        labels = chunk[('power','active')].values
        labels = scaler.fit_transform(labels)
        labels = np.expand_dims(labels, 2)

        # Determine the indices for training and testing
        valid_indexes = np.arange(max(params['models']) + params['horizon'], len(chunk))
        testing_indexes = np.random.choice(valid_indexes,len(valid_indexes) * params['validation_quota'])
        testing_indexes.sort()
        training_indexes = np.delete(valid_indexes, testing_indexes)
        training_indexes.sort()

        # Go through all horizons
        for modelnumber, horizon in enumerate(params['models']):

            # Do the training batch per batch            
            num_minibatches = features.shape[0] // params['size_minibatches']
            for i in range(num_minibatches * params['epochs']):
                # Put together the features an labels
                items = np.random.choice(training_indexes, self.model.params['size_minibatches'])
                cur_input = []
                for item in items:
                    cur_input.append(features[item-horizon-params['horizon']:item-horizon])
                # features = list(np.expand_dims(training_features[items],2)) #np.ascontiguousarray(training_features[items])
                # labels =  list( np.expand_dims(training_labels[items], 2))#training_labels[items])#np.ascontiguousarray(training_labels[items])
                cur_labels = list(labels[items])

                # iterate the minibatches
                model = self.model[modelnumber]
                model['trainer'].train_minibatch({model['input']: cur_input, model['label']: cur_labels})
                if i % params['training_progress_output_freq'] == 0:
                    self._track_training_progress(modelnumber, horizon , i, features, labels, testing_indexes, verbose = verbose)

            print("Training took {:.1f} sec".format(time.time() - start))
            model['trainer'].save_checkpoint("lstm.checkpoint", self.model[0]['plotdata'])

        # Print the train and validation errors
        for labeltxt in ["train", "val"]:
            print("mse for {}: {:.6f}".format(labeltxt, self.get_mse(X, Y, labeltxt, trainer, x, l)))

        # Print the test error
        labeltxt = "test"
        print("mse for {}: {:.6f}".format(labeltxt, self.get_mse(X, Y, labeltxt, trainer, x, l)))


    def forecast(self, meters, extData, timestamp, verbose = False):
        '''
        This method uses the learned model to predict the future
        For each forecaster the forecast horizon is derived from its 
        smallest shift value.
        All meters that contain a powerflow are considered part of the 
        group to forecast.

        Paramters:
        -------------
        
        Parameters
        ----------
        meters: nilmtk.DataSet
            The meters from which the demand is loaded.
        timestamp: [pd.TimeStamp,...] or pd.DatetimeIndex
            The point in time from which the prognoses is performed.
        '''

       # Predict
        f, a = plt.subplots(2, 1, figsize=(12, 8))
        for j, ds in enumerate(["val", "test"]):
            results = []
            for x_batch, _ in next_batch(X, Y, ds):
                pred = z.eval({x: x_batch})
                results.extend(pred[:, 0])
            # because we normalized the input data we need to multiply the prediction
            # with SCALER to get the real values.
            a[j].plot((Y[ds] * NORMALIZE).flatten(), label=ds + ' raw');
            a[j].plot(np.array(results) * NORMALIZE, label=ds + ' pred');
            a[j].legend();


                    
    def _track_training_progress(self, modelnumber, horizon, mb, features, labels, testing_indices, verbose = False):
        ''' Writes training progress into the buffer.
        Calculates the error per minibatch and outputs the error with a given frequency.

        Parameters
        ----------
        modelnumber: int
            The model, which shall be used for the forecast
        horizon: int
            The horizon, the model is forecasting for.
        mb: int
            The number of the minibatch
        ann_features: 
            The input features for the ann_input
        lstm_features:
            The input features for the lstm_input
        labels: 
            The real output taken from the ground_truth
        testing_indices: [int,...]
            All indices in the main data, reserved for testing. 
            Elements of these indices are not used for training.
        verbose: bool
            Whether to print additional output
        '''

        selection = np.random.choice(testing_indices, 10)
        cur_input = []
        for item in selection:
            cur_input.append(features[item-horizon-self.model.params['horizon']:item-horizon])
        cur_labels = list(labels[selection])

        model =  self.model[modelnumber]
        training_loss = model['trainer'].previous_minibatch_loss_average
        eval_error = model['trainer'].test_minibatch({model['input'] : cur_input, model['label'] : cur_labels})
        eval_error = eval_error * self.model.labelscaler.scale_
        avg_power = (self.model.labelscaler.inverse_transform(labels[selection])).mean()

        model['plotdata']["batchnumber"].append(mb)
        model['plotdata']["loss"].append(training_loss)
        model['plotdata']["error"].append(eval_error)
        model['plotdata']["avg_power"].append(avg_power)
        if verbose: 
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}, AvgPower: {3}".format(mb, training_loss, eval_error, avg_power))