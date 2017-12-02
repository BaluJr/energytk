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
from cntk.logging.progress_print import TensorBoardProgressWriter
from cntk.layers import Sequential
from cntk.layers.typing import Tensor, Sequence

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve



class AnnLstmForecasterModel(Sequence):
    params = {
        # The target storage to store the data to
        'forecasting_storage': "C:\\Users\\maxim_000\\Documents\\InformatikStudium_ETH\\Masterarbeit\\6_Data\\Tmp\\forecasting.hdf",
        
        # Which timesteps into the future we predict
        'models': [1],
        
        # Amount of data used for validation
        'validation_quota': 0.1,

        # Define output
        'training_progress_output_freq': 50,

        # Learning Rate
        'learning_rate': 0.005,

        # How many errors are taken together until training step
        'size_minibatches': 10,
        
        # Whether to use (100 for is_fast, else 2000)
        'epochs': 100,


        # Dimensionality of LSTM cell
        'lstm_h_dim': 10,

        # The hidden dimensionality of the LSTM-NN
        'lstm_dim': 10,
        
        # How far to the past the network looks
        'lstm_horizon': 192,
            
        # The features which are used as input (for the dense ANN)
        'lstm_externalFeatures': [], #('temperature', '')],#, ('national', ''), ('school', '')],#, ('dewpoint', '')], #, 'time'
        
        # How the daytime is regarded (for the dense ANN)
        'lstm_hourFeatures': [],#('hour', '00-06'), ('hour', "06-09"), ('hour', "09-12"), ('hour', "12-15"), ('hour', "15-18"), ('hour', "18-21"), ('hour', "21-24")],

        # How the weekdays are paired (for the dense ANN)
        'lstm_weekdayFeatures': [],#('week', "0-5"),('week', "5-6"),('week',"6-7")],


        # The hidden layers of the dense ANN
        'ann_num_hidden_layers': 2,

        # Dimensionality of the dense ANNs hidden layers
        'ann_hidden_layers_dim': 50,
        
        # The features which are used as input (for the dense ANN)
        'ann_externalFeatures': [('temperature', '')],#, ('national', ''), ('school', '')],#, ('dewpoint', '')], #, 'time'
        
        # How the daytime is regarded (for the dense ANN)
        'ann_hourFeatures': [('hour', '00-06'), ('hour', "06-09"), ('hour', "09-12"), ('hour', "12-15"), ('hour', "15-18"), ('hour', "18-21"), ('hour', "21-24")],

        # How the weekdays are paired (for the dense ANN)
        'ann_weekdayFeatures': [('week', "0-5"),('week', "5-6"),('week',"6-7")],

        
        # Unfortunately lstm debug not yet working in CNTK 2.2
        'debug' : False,
        
        # The folder where the tensorflow logs are placed
        'tensorboard_dict': "E:Tensorboard/"
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


class AnnLstmForecaster(Forecaster):
    """This is a fance forecaster which combines an rnn and an ANN. The previous load is regarded by using an RNN.
    That output is then inserted into an ANN, which subsequently incorporates the influence of the additional data.

    Ggf kann man das ANN behalten.

    ----------
    model_class : The model type, which belonging to the forecaster.
    
    """
    
    # The related model
    model_class = AnnLstmForecasterModel
    

    def __init__(self, model = None):
        """
        Constructor of this class which takes an optional model as input.
        If no model is given, it creates a default one.
        """
        super(AnnLstmForecaster, self).__init__(model)

        
    def mape(self, z, l):
        return C.reduce_mean(C.abs(z - l)/l) 
    
    def mae(self, z, l):
        return C.reduce_mean(C.abs(z - l))


    def _model_creation(self, modelnumber):
        """
        Create the model for time series prediction
        """
        
        # Define input
        params = self.model.params
        lstm_input_dim = 1 + len(params['lstm_externalFeatures']) + len(params['lstm_hourFeatures']) + len(params['lstm_weekdayFeatures'])
        self.model[modelnumber]['lstm_input'] = lstm_input = C.sequence.input_variable(shape=lstm_input_dim, is_sparse = False, name="lstm_input")     # Das fuegt glaube sofort schon eine Dyn Achse ein

        ann_input_dim = len(params['ann_externalFeatures']) + len(params['ann_hourFeatures']) + len(params['ann_weekdayFeatures'])
        self.model[modelnumber]['ann_input'] = ann_input = C.input_variable(shape=ann_input_dim, is_sparse = False, name="ann_input")     # Das fuegt glaube sofort schon eine Dyn Achse ein

        # Create the network
        with C.layers.default_options(initial_state = 0.1):
            # First the lstm
            lstm = C.layers.Recurrence(C.layers.LSTM(self.model.params['lstm_dim'], self.model.params['lstm_h_dim'], name="recur"))(lstm_input)
            lstm = C.sequence.last(lstm, name="recurlast")
            lstm = C.layers.Dropout(0.2, name="dropout")(lstm)
            # Combine with external data
            h = C.splice(ann_input, lstm, name = 'combination')
            # Create a dense network in behind
            with C.layers.default_options(init = C.glorot_uniform(), activation=C.relu):
                for i in range(params['ann_num_hidden_layers']):
                    h = C.layers.Dense(params['ann_hidden_layers_dim'], name='ann'+str(i))(h)
            z = C.layers.Dense(1, activation=None, name="ann_last")(h)
        if self.model.params['debug']:
            z = debug_model(z)

        # Define output
        self.model[modelnumber]['label'] = l = C.input_variable(1, dynamic_axes=z.dynamic_axes, name="y") 
        
        # Define error metric and trainer
        loss = C.squared_error(z, l)
        error = self.mae(z, l)
        lr_schedule = C.learning_rate_schedule(params['learning_rate'], C.UnitType.minibatch)
        momentum_time_constant = C.momentum_as_time_constant_schedule(params['size_minibatches'] / -math.log(0.9)) 
        learner = C.fsadagrad(z.parameters, lr = lr_schedule, momentum = momentum_time_constant)
        tensorboard_writer = TensorBoardProgressWriter(freq=10, log_dir=self.model.params['tensorboard_dict'], model=z)
        self.model[modelnumber]['trainer']  = C.Trainer(z, (loss, error), [learner], tensorboard_writer )


    def train(self, meters, extDataSet, verbose = False):
        '''
        Does the training of the neural network. Each load chunk is translated into 
        multiple minibatches used for training the network.
        '''
        params = self.model.params
        
        # A) Load the data
        section = TimeFrame(start=pd.Timestamp("1.1.2016", tz = 'UTC'), end = pd.Timestamp("15.03.2017", tz = 'UTC'))
        #section = meters.get_timeframe(intersection_instead_union=True)
        sections = TimeFrameGroup([TimeFrame(start=pd.Timestamp("1.1.2016", tz = 'UTC'), end = pd.Timestamp("15.03.2017", tz = 'UTC'))])
        #powerflow = meters.power_series_all_data(verbose = True, sample_period=3600, sections=sections).dropna()
        chunk = pd.DataFrame(pckl.load(open("./ForecastingBenchmark15min.pckl", "rb")))
        
        # B) Prepare the data
        weekday_features = set(params['ann_weekdayFeatures']+params['lstm_weekdayFeatures'])
        hourFeatures = set(params['ann_hourFeatures'] + params['lstm_hourFeatures'])
        chunk = self._addTimeRelatedFeatures(chunk, weekday_features, hourFeatures)
        externalFeatures = set(self.model.params['ann_externalFeatures'] + self.model.params['lstm_externalFeatures']) 
        chunk = self._addExternalData(chunk, extDataSet, section, externalFeatures)
        
        # C) Create the model
        models = self.model.params['models']
        for modelnumber in range(len(models)):
            self._model_creation(modelnumber)

        # D) Arrange the input for LSTM and ANN and labels
        train_test_mask = np.random.rand(len(chunk)) < 0.8
        ann_features = params['ann_externalFeatures'] + params['ann_hourFeatures'] + params['ann_weekdayFeatures']
        self.model.ann_featurescaler = scaler = StandardScaler()
        ann_features = chunk[ann_features].values
        ann_features = scaler.fit_transform(ann_features)    
        lstm_features = params['lstm_externalFeatures'] + params['lstm_hourFeatures'] + params['lstm_weekdayFeatures'] + [('power','active')]
        self.model.lstm_featurescaler = scaler = StandardScaler()
        lstm_features = chunk[lstm_features].values
        lstm_features = scaler.fit_transform(lstm_features)        
        #training_features = list(np.expand_dims(training_features,2))
        self.model.labelscaler = scaler = StandardScaler()
        labels = chunk[('power','active')].values
        labels = scaler.fit_transform(labels)
        labels = np.expand_dims(labels, 2)

        # Determine the indices for training and testing
        valid_indexes = np.arange(max(params['models']) + params['lstm_horizon'], len(chunk))
        testing_indexes = np.random.choice(valid_indexes,len(valid_indexes) * params['validation_quota'])
        testing_indexes.sort()
        training_indexes = np.delete(valid_indexes, testing_indexes)
        training_indexes.sort()

        # Go through all horizons
        for modelnumber, horizon in enumerate(params['models']):

            # Do the training batch per batch            
            num_minibatches = len(testing_indexes) // params['size_minibatches']
            for i in range(num_minibatches * params['epochs']):
                # Put together the features an labels
                items = np.random.choice(training_indexes, self.model.params['size_minibatches'])
                cur_lstm_input = []
                for item in items:
                    cur_lstm_input.append(lstm_features[item-horizon-params['lstm_horizon']:item-horizon])
                cur_ann_input = ann_features[items]
                cur_labels = list(labels[items])

                # iterate the minibatches
                model = self.model[modelnumber]
                model['trainer'].train_minibatch({model['label']: cur_labels, model['lstm_input']: cur_lstm_input, model['ann_input']: cur_ann_input})
                if i % params['training_progress_output_freq'] == 0:
                    self._track_training_progress(modelnumber, horizon , i, ann_features, lstm_features, labels, testing_indexes, verbose = verbose)

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
        meters:     The meter group to predict
        timestamp:  The point in time from which the prognoses is performed.
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


                    
    def _track_training_progress(self, modelnumber, horizon, mb, ann_features, lstm_features, labels, testing_indices, verbose = False):
        '''
        Calculates the error per minibatch. 
        Outputs the error with a given frequency.
        '''
        selection = np.random.choice(testing_indices, 10)
        cur_lstm_input = []
        for item in selection:
            cur_lstm_input.append(lstm_features[item-horizon-self.model.params['lstm_horizon']:item-horizon])
        cur_ann_input = ann_features[selection]
        cur_labels = list(labels[selection])

        model =  self.model[modelnumber]
        training_loss = model['trainer'].previous_minibatch_loss_average
        eval_error = model['trainer'].test_minibatch({model['label']: cur_labels, model['lstm_input']: cur_lstm_input, model['ann_input']: cur_ann_input})
        eval_error = eval_error * self.model.labelscaler.scale_
        avg_power = (self.model.labelscaler.inverse_transform(labels[selection])).mean()

        model['plotdata']["batchnumber"].append(mb)
        model['plotdata']["loss"].append(training_loss)
        model['plotdata']["error"].append(eval_error)
        model['plotdata']["avg_power"].append(avg_power)
        if verbose: 
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}, AvgPower: {3}".format(mb, training_loss, eval_error, avg_power))
        
      


            
    #def _addShiftsToChunkAndReduceToValid(self, chunk):
    #    '''
    #    This function takes the current chunk of the meter and adapts it sothat
    #    it can be used for training. That means it extends the dataframe by the 
    #    missing features we want to learn.
    #    '''        
    #    # Determine the shifts that are required. All in memory since only a few values
    #    all_shifts = set()
    #    for shift in range(self.model.params['horizon'], 0, -1):
    #        all_shifts.update(range(shift, shift+self.model.params['models']))

    #    # Compute the stock being up (1) or down (0) over different day offsets compared to current dat closing price
    #    for i in all_shifts:
    #        chunk[('shifts', str(i))] = chunk[('power','active')].shift(i)    # i: number of look back days

    #    chunk.drop(chunk.index[chunk[('shifts',str(max(all_shifts)))].isnull()], inplace=True)
          
