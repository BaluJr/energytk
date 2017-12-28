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
    '''
    This is the model used for forecasting. It is an enumerable with one set of the below attributes
    for each entry. Each entry represents one forecasting horizon

    Attributes:
    ann_featurescaler: sklearn.preprocessing.StandardScaler
        The scaler used to scale the ann inputs.
    lstm_featurescaler: sklearn.preprocessing.StandardScaler
        The scaler used to scale the lstm inputs.
    labelscaler: sklearn.preprocessing.StandardScaler
        The scaler used to scale the labels.

    Attributes per horizon
    ----------------------
    trainer: cntk.train.Trainer
        The trainer used for the model
    ann_input: C.input_variable
        The input which goes into the ann, which is responsible to incorporate the external features.
    lstm_input: C.sequence.input_variable
        The input for the rnn, which reads the past entry values
    tensorboard_writer: C.logging.progress_print.TensorBoardProgressWriter
        The tensorboard writer to write the data to.
    model: C
        The model z, which is also included in the trainer. But since the reloading is not yet 
        working properly, also stored separatly.
    plotdata: {"batchnumber":[], "loss":[], "error":[], "avg_power":[]}
        This is the buffer for storing the training results. Has been more or less replaced by the
        Tensorflow progress writer.
    '''

    params = {
        # Which timesteps into the future we predict
        'models': list(range(12, 97, 12)), #[1] + list(range(2, 92, 12)),
        
        # Amount of data used for validation
        'validation_quota': 0.1,

        # Learning Rate
        'learning_rate': 0.01,

        # How many errors are taken together until training step
        'size_minibatches': 10,
        
        # Whether to use (100 for is_fast, else 2000)
        'epochs': 15,

        # Resolution of one step
        'resolution': '15m',

        # Dimensionality of LSTM cell
        'lstm_h_dim': 10,

        # The hidden dimensionality of the LSTM-NN
        'lstm_dim': 10,
        
        # How far to the past the network looks
        'lstm_horizon': 96 * 7,
            
        # The features which are used as input (for the dense ANN)
        'lstm_external_features': [], #('temperature', '')],#, ('national', ''), ('school', '')],#, ('dewpoint', '')], #, 'time'
        
        # Uses continuous features for weekdays and hours
        'continuous_time_features': False,

        # How the daytime is regarded (for the dense ANN)
        'lstm_hourFeatures': [('hour', '00-06'), ('hour', "06-09"), ('hour', "09-12"), ('hour', "12-15"), ('hour', "15-18"), ('hour', "18-21"), ('hour', "21-24")],

        # How the weekdays are paired (for the dense ANN)
        'lstm_weekdayFeatures': [('week', "0-5"),('week', "5-6"),('week',"6-7")],


        # The hidden layers of the dense ANN
        'ann_num_hidden_layers': 2,

        # Dimensionality of the dense ANNs hidden layers
        'ann_hidden_layers_dim': 20,
        
        # The features which are used as input (for the dense ANN)
        'ann_external_features': [],#('temperature', '')],#, ('national', ''), ('school', '')],#, ('dewpoint', '')], #, 'time'
        
        # How the daytime is regarded (for the dense ANN)
        'ann_hourFeatures': [('hour', '00-06'), ('hour', "06-09"), ('hour', "09-12"), ('hour', "12-15"), ('hour', "15-18"), ('hour', "18-21"), ('hour', "21-24")],

        # How the weekdays are paired (for the dense ANN)
        'ann_weekdayFeatures': [('week', "0-5"),('week', "5-6"),('week',"6-7")],

        
        # Unfortunately lstm debug not yet working in CNTK 2.2
        'debug' : False,
        
        # Define output
        'training_progress_output_freq': 50
    }
    

    def __init__(self, **args):
        '''
        The paramters are set to the optional paramters handed in by the constructor.
        
        Parameters
        ----------
        args: dic
            Parameters which are optionaly set
        '''
        for arg in args:
            self.params[arg] = args[arg]
        self.model = [
            {'trainer':None, 
                'ann_input':None, 
                'lstm_input':None, 
                'label':None, 
                'model':None,
                'plotdata': {"batchnumber":[], "loss":[], "error":[], "avg_power":[]}
            }] * len(self.params['models'])
    
    def set_model(self, i, new_model):
        self.model[i] = new_model

    def __getitem__(self, i):
        '''
        Makes it possible to access the data of the i-th submodel.
        Paramters
        ---------
        key: int
            Model to access
        '''
        return self.model[i]

    def __len__(self):
        return self.model.__len__()


    def plot_training_progress(self, mean = False):
        ''' 
        This method plots the training progress which has been recorded during training.
        The model for which the plot shall be drawn. When set to -1 then all are drawn.
        This function is more or less deprecated since the tensorflow progress writer has been
        in use

        Paramters
        ---------
        mean: bool 
            Whether to plot the rolling mean of length 20 over the results. Smoothing.
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


    def store(self, folder):
        ''' Stores the model in a certain folder.
        A folder is given as certain models cannot be stored within a single file.
        
        folder: str
            Folder in which the model is persistently stored
        '''
        # Cancel if folder already exists and contains elements
        if os.path.isdir(folder):
            pass
            #if len(os.listdir("E:/Tensorboard/0")) != 5:
            #    raise Exception("Folder must be empty.")
        else:
            os.mkdir(folder)

        # Store the cntk model and remove
        for i, cur in enumerate(self):
            cur["trainer"].save_checkpoint(folder + str(i) + "/.trainer")
        del cur['trainer']
        del cur['ann_input']
        del cur['lstm_input']
        del cur['label']
        del cur['model']
        del cur['tensorboard_writer']
        # The rest can be pickled normally 
        pckl.dump(self, open(folder+ "basemodel", "wb")) 
        pckl.dump(self, open(folder+ "basemodel", "wb")) 
    

    def load(folder):
        ''' Loads the model from the designated folder.
        A folder is given as certain models cannot be stored within a single file.
        
        folder: str
            Folder in which the model can be found
        '''
        # Cancel if folder already exists and contains elements
        if not os.path.isdir(folder):
            raise Exception("The designated folder does not exist.")

        # First reload the base model
        try:
            model = pckl.load(open(folder+ "basemodel", "rb")) 
        except:
            model = AnnLstmForecasterModel()
        
        # Store the cntk model and remove (not working yet)
        for i, cur in enumerate(model):
            cur['model'] = C.load_model(folder + str(i) + '/model.dnn')
            #cur['ann_input'] = cur['model']['ann_input']
            #cur['lstm_input'] = cur['model']['lstm_input']
            #cur["trainer"].load_checkpoint(folder + str(i) + "/.trainer")
            #cur["trainer"].restore_from_checkpoint(folder + str(i) + "/.trainer")
            #cur['ann_input'] = cur["trainer"].model['ann_input']
            #cur['lstm_input'] = cur['trainer'].model['lstm_input']
        return model





class AnnLstmForecaster(Forecaster):
    """Forecaster which combines a RNN and an ANN. 
    
    The previous load is regarded by using an RNN. That output is then inserted 
    into an ANN, which subsequently incorporates the influence of the additional data.
    It offers the same forecaster interface like the other forecasters.
    
    Attributes
    ----------
    model_class: type  
        The model type, which belonging to the forecaster.
    """
    
    # The related model
    model_class = AnnLstmForecasterModel

    def __init__(self, model = None, folder = None):
        """
        Constructor of this class which takes an optional model as input.
        If no model is given, it creates a default one.

        Paramters
        ---------
        model: Model of type model_class
            The model which shall be used.
        folder: str
            When set, the model is reconstructed from the content of the designated folder.
        """
        if not folder is None:
            model = AnnLstmForecasterModel.load(folder)

        super(AnnLstmForecaster, self).__init__(model)
        

    def _prepare_data(self, chunk, ext_dataset):
        ''' Adds the additional featues to the data
        
        Parameters
        ----------
        chunk: pd.DataFrame
            Chunk to extend
        ext_dataset: nilmtk.DataSet or str
            The External Dataset containing the fitting data.
            Alternatively a path to a PickleFile. 
        
        Returns
        -------
        chunk: pd.DataFrame
            The extended power series
        '''
        params = self.model.params
        if params['continuous_time_features']:
            chunk = self._add_continuous_time_related_features(chunk)
        else:
            weekday_features = set(params['ann_weekdayFeatures']+params['lstm_weekdayFeatures'])
            hourFeatures = set(params['ann_hourFeatures'] + params['lstm_hourFeatures'])
            chunk = self._add_time_related_features(chunk, weekday_features, hourFeatures)
        external_features = set(self.model.params['ann_external_features'] + self.model.params['lstm_external_features']) 
        if len(external_features) != 0:
            chunk = self._add_external_data(chunk, ext_dataset, external_features)
        return chunk

    def _model_creation(self, modelnumber, tensorboard_folder):
        """ Create the model for time series prediction.
        Since there are multiple models, the parameter defines 
        the number of the model.

        Parameters
        ----------
        modelnumber: int
            Forecasting horizon this model is trained for.
        tensorboard_folder: str
            path where the tensorboard output shall be stored.
        """
        
        # Define input
        params = self.model.params
        if params['continuous_time_features']:
            ann_input_dim = len(params['ann_external_features']) + 3
            lstm_input_dim = 1 + len(params['lstm_external_features']) + 3
        else:
            ann_input_dim = len(params['ann_external_features']) + len(params['ann_hourFeatures']) + len(params['ann_weekdayFeatures'])
            lstm_input_dim = 1 + len(params['lstm_external_features']) + len(params['lstm_hourFeatures']) + len(params['lstm_weekdayFeatures'])
        self.model[modelnumber]['lstm_input'] = lstm_input = C.sequence.input_variable(shape=lstm_input_dim, is_sparse = False, name="lstm_input")

        self.model[modelnumber]['ann_input'] = ann_input = C.input_variable(shape=ann_input_dim, is_sparse = False, name="ann_input")

        # Create the network
        with C.layers.default_options(initial_state = 0.1):
            # First the lstm
            lstm = C.layers.Recurrence(C.layers.LSTM(self.model.params['lstm_dim'], self.model.params['lstm_h_dim'], name="recur"))(lstm_input)
            lstm = C.sequence.last(lstm, name="recurlast")
            #lstm = C.layers.Dropout(0.1, name="dropout")(lstm)
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
        if not tensorboard_folder is None:
            tensorboard_writer = TensorBoardProgressWriter(freq=10, log_dir=tensorboard_folder + str(modelnumber) + "/", model=z)
            self.model[modelnumber]['tensorboard_writer']  = tensorboard_writer 
            self.model[modelnumber]['trainer']  = C.Trainer( z, (loss, error), [learner], tensorboard_writer)
        else:
            self.model[modelnumber]['trainer']  = C.Trainer( z, (loss, error), [learner])
        self.model[modelnumber]['model'] = z

    def _arrange_features_and_labels(self, chunk, for_forecast = False):
        ''' Arrange the inputs and labels as needed by the networks.
        It selects from the loaded data the features which are used and prepares
        them by removing the mean and scaling to unit variance.

        Parameters
        ----------
        chunk: pd.DataFrame
            Chunk to convert to arrays for network input
        for_forecast: bool
            If set to true only the features are returned and
            the scalers are kept as set.

        Returns
        -------
        ann_features: np.ndarray
            Scaled features for ann prepared as array
        lstm_features: np.ndarray
            Scaled features for lstm prepared as array
        labels: np.ndarray (only for features_only = False)
            Scaled result labels prepared for trainng as array
        '''
        params = self.model.params
        train_test_mask = np.random.rand(len(chunk)) < 0.8

        if params['continuous_time_features']:
            ann_features = params['ann_external_features'] + [('weekday',''), ('time',''),('dayofyear','')]
            lstm_features = params['lstm_external_features'] + [('weekday',''), ('time',''), ('dayofyear','')] + [('power','active')]
        else:
            ann_features = params['ann_external_features'] + params['ann_hourFeatures'] + params['ann_weekdayFeatures']
            lstm_features = params['lstm_external_features'] + params['lstm_hourFeatures'] + params['lstm_weekdayFeatures'] + [('power','active')]
        ann_features = chunk[ann_features].values
        lstm_features = chunk[lstm_features].values
        
        if for_forecast:
            ## Nur Temporaer da gleicher Datensatz wie training - Remove later
            self.model.ann_featurescaler = StandardScaler()
            self.model.ann_featurescaler.fit(ann_features)    
            self.model.lstm_featurescaler = StandardScaler()
            self.model.lstm_featurescaler.fit(lstm_features)
            self.model.labelscaler = StandardScaler()
            self.model.labelscaler.fit(chunk[('power','active')].values)
            ## Ende remove
            ann_features = self.model.ann_featurescaler.transform(ann_features)    
            lstm_features = self.model.lstm_featurescaler.transform(lstm_features)   
            return ann_features, lstm_features
        
        labels = chunk[('power','active')].values
        self.model.ann_featurescaler = scaler = StandardScaler()
        self.model.lstm_featurescaler = scaler = StandardScaler()
        self.model.labelscaler = scaler = StandardScaler()
        ann_features = scaler.fit_transform(ann_features)    
        lstm_features = scaler.fit_transform(lstm_features)        
        labels = scaler.fit_transform(labels)
        labels = np.expand_dims(labels, 2)
        return ann_features, lstm_features, labels


    def train(self, meters, ext_dataset, section = None, tensorboard_folder = None, verbose = False):
        '''
        Does the training of the neural network. Each load chunk is translated into 
        multiple minibatches used for training the network.
        
        Parameters
        ----------
        meters: nilmtk.DataSet or str
            The meters from which the demand is loaded. 
            Alternatively a path to a PickleFile. 
        ext_dataset: nilmtk.DataSet or str
            The External Dataset containing the fitting data.
            Alternatively a path to a PickleFile. 
        section: nilmtk.TimeFrame
            The timeframe used for training. Meters have to be valid within this region.
        tensorboard_folder: str
            path where the tensorboard output shall be stored.
        verbose: bool
            Whether additional output shall be printed during training.
        '''
        params = self.model.params
        
        # A) Load the data
        chunk = self._load_data(meters, section)
        
        # B) Prepare the data
        chunk = self._prepare_data(chunk, ext_dataset)
        
        # C) Create the model
        models = self.model.params['models']
        for modelnumber in range(len(models)):
            self._model_creation(modelnumber, tensorboard_folder)

        # D) Arrange the input for LSTM and ANN and labels
        ann_features, lstm_features, labels = self._arrange_features_and_labels(chunk)

        # Determine the indices for training and testing
        valid_indexes = np.arange(max(params['models']) + params['lstm_horizon'], len(chunk))
        testing_indexes = np.random.choice(valid_indexes,len(valid_indexes) * params['validation_quota'])
        testing_indexes.sort()
        training_indexes = np.delete(valid_indexes, testing_indexes)
        training_indexes.sort()

        # Go through all horizons
        for modelnumber, horizon in enumerate(params['models']):

            # Do the training batch per batch            
            start = time.time()
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

            self.model[modelnumber]['trainer'].model.save(tensorboard_folder + str(modelnumber) + "/model.dnn")
            self.model[modelnumber]['tensorboard_writer'].flush()
            open(tensorboard_folder + str(modelnumber) + "/" + str(horizon) ,"w+").close()
            self.model[modelnumber]['trainer'].save_checkpoint(tensorboard_folder + str(modelnumber) + "/lstm.checkpoint")
            print("Training took {:.1f} sec".format(time.time() - start))
        
        self.model.store(tensorboard_folder)
        tmp = open(tensorboard_folder + "/model.txt" ,"w+")
        tmp.write(str(self.model.params))
        tmp.close()
        return self.model



    def forecast(self, meters, ext_dataset, timestamps, verbose = False):
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
        verbose: bool
            Whether additional output shall be printed during training.

        Returns
        -------
        forecasts: pd.DataFrame
            A DataFrame containing the forecasts for each Timestamp. 
            One column for each timestamp and one row for each forecaster 
            horizon.
        '''
        if type(timestamps) is list:
            timestamps = pd.DatetimeIndex(timestamps)
        params = self.model.params

        forecast = pd.DataFrame(index = timestamps)

        # Prepare the input
        chunk = self._load_data(meters)
        chunk = self._prepare_data(chunk, ext_dataset)
        #chunk = self._append_future_data(chunk) # This would be necessary to really forecast the future
        ann_features, lstm_features = self._arrange_features_and_labels(chunk, for_forecast = True)

        # Determine the indices inside the dataset
        items = pd.Series(index = chunk.index, data= range(len(chunk)))
        items = items.loc[timestamps].values

        # Go through all horizons
        for modelnumber, cur in enumerate(self.model):
            horizon = params['models'][modelnumber]

            # Put together the features an labels
            cur_lstm_input = []
            for item in items:
                cur_lstm_input.append(lstm_features[item-params['lstm_horizon']:item])
            cur_ann_input = ann_features[items + horizon]            
            
            z = cur['model']
            pred = z(cur_ann_input, cur_lstm_input) #z.eval({ann_features: cur_ann_input, lstm_input: cur_lstm_input})
            forecast[horizon] = self.model.labelscaler.inverse_transform(pred[:, 0])
        
        forecast = forecast.transpose()
        forecast['horizon'] = forecast.index.values * pd.Timedelta(params['resolution'])
        forecast = forecast.set_index('horizon')
        return forecast

                    
    def _track_training_progress(self, modelnumber, horizon, mb, ann_features, lstm_features, labels, testing_indices, verbose = False):
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
        