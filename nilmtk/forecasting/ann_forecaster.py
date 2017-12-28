from statsmodels.tsa.arima_model import ARIMA
import cntk as C
from cntk.logging.progress_print import TensorBoardProgressWriter
from cntk.debugging import debug_model
import numpy as np
from .forecaster import Forecaster
from pandas import DataFrame
from nilmtk import DataSet, ExternDataSet, TimeFrame, TimeFrameGroup
from collections.abc import Sequence
import pandas as pd
import pickle as pckl
import copy
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

#TODO
# ALSO habe hier gerae den zentralen Featuerescaler gebaut -> Ausprobieren auch mit forecast
# SARIMAX anpassen, dass gleiches forecasting wie die anderen, bzw. formalisieren was dort anders ist
# Schauen ob HoltWinter bauen
# Alle durchlaufen lassen
# Pruefen warum die Verschiebung statt fand
# Error metriken berechnen (check)



class AnnForecasterModel(Sequence):
    '''
    This is the model used for forecasting. It is an enumerable with one set of the below attributes
    for each entry. Each entry represents one forecasting horizon

    Attributes
    ----------
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
    input: C.input_variable
        The input going into the ann.
    label: C.input_variable
        The output of the network
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
        'models': [1] + list(range(2, 98, 24)),
        
        # Amount of data used for validation
        'validation_quota': 0.1,
        
        # Learning Rate
        'learning_rate': 0.005,

        # How many errors are taken together until training step
        'size_minibatches': 10,

        # How often each minibatch is trained
        'epochs': 1,

        # Resolution of one step
        'resolution': '15m',


        # The features which are used as input
        'external_features': [],#[('temperature', '')],#, ('national', ''), ('school', '')],#, ('dewpoint', '')], #, 'time'
        
        # How the daytime is regarded
        'hourFeatures': [('hour', '00-03'), ('hour', '03-06'), ('hour', "06-09"), ('hour', "09-12"), ('hour', "12-15"), ('hour', "15-18"), ('hour', "18-21"), ('hour', "21-24")],

        # Shifts are the prev days
        'shifts': list(range(96,192,4)) + [cur*96 for cur in range(3,7)], #+ [cur*24*7 for cur in range(4)],
        
        # How the weekdays are paired
        'weekdayFeatures': [('week', "0-5"),('week', "5-6"),('week',"6-7")],
        
        # Number of hidden layers of the ANN
        'num_hidden_layers': 2,

        # Dimensionality of the ANN's hidden layers
        'hidden_layers_dim': 50,


        # Whether to give debug output during training.
        'debug': False,
        
        # Define output
        'training_progress_output_freq': 40,
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
             'model':None,
             'plotdata': {"batchnumber":[], "loss":[], "error":[]}
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

        
    def store(self, folder):
        ''' Stores the model in a certain folder.
        A folder is given as certain models cannot be stored within a single file.
        
        folder: str
            Folder in which the model is persistently stored
        '''
        # Cancel if folder already exists and contains elements
        if os.path.isdir(folder):
            pass
            #if len(os.listdir(folder)) != 0:
            #    raise Exception("Folder must be empty.")
        else:
            os.mkdir(folder)

        # Store the cntk model and remove
        for i, cur in enumerate(self):
            cur["trainer"].save_checkpoint(folder + str(i) + "/.trainer")
            
        del cur['trainer'] # Don't ask me why these are all the same objects for all iterates
        del cur['input']
        del cur['label']
        
        # The rest can be pickled normally 
        pckl.dump(self, open(folder+ "model", "wb")) 
    

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
            model = pckl.load(open(folder+ "/basemodel", "rb")) 
        except:
            model = AnnForecasterModel()
        
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





class AnnForecaster(Forecaster):
    """This is a forecaster based on a distinct artificial neural network (ANN) 
    for each of the the forecasting distances.
    
    Attributes
    ----------
    model_class: type  
        The model type, which belonging to the forecaster.
    
    """

    # The related model
    model_class = AnnForecasterModel

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
            model = AnnForecasterModel.load(folder)

        super(AnnForecaster, self).__init__(model)

        
    def get_mse(X,Y,labeltxt):
        '''
        At the moment don't know where used.
        '''
        result = 0.0
        for x1, y1 in next_batch(X, Y, labeltxt):
            eval_error = self.model.trainer.test_minibatch({x : x1, l : y1})
            result += eval_error
        return result/len(X[labeltxt])

    
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
        power_column = [('power', 'active')] #meters._convert_physical_quantity_and_ac_type_to_cols(physical_quantity = 'power', ignore_missing_columns = True)['cols']
        chunk = self._add_time_related_features(chunk, params['weekdayFeatures'], params['hourFeatures'])
        chunk = self._add_external_data(chunk, ext_dataset, self.model.params['external_features'])
        chunk = self._addShiftsToChunkAndReduceToValid(chunk, params['shifts'], params['models'])
        return chunk

    def _model_creation(self, modelnumber, tensorboard_folder = None):
        ''' Create the ANN model for time series forecasting 

        Parameters
        ----------
        modelnumber: int
            The number of model to create. Needed to look up its params.
        tensorboard_folder: str
            path where the tensorboard output shall be stored.
        '''
        

        # Define input and output which his fixed
        params = self.model.params
        input_dim = len(params['external_features']) + len(params['shifts']) + len(params['hourFeatures']) + len(params['weekdayFeatures'])
        input_dynamic_axes = [C.Axis.default_batch_axis()]
        self.model[modelnumber]['input'] = input = C.input_variable(input_dim)#, dynamic_axes=input_dynamic_axes)
        self.model[modelnumber]['label'] = label = C.input_variable(1)#, dynamic_axes=input_dynamic_axes)
        
        # Create the network
        h = input
        with C.layers.default_options(init = C.glorot_uniform(), activation=C.relu):
            for i in range(params['num_hidden_layers']):
                h = C.layers.Dense(params['hidden_layers_dim'])(h)
            z = C.layers.Dense(1, activation=None)(h)   
        if params['debug']:
            z = debug_model(z)

        # Define error metric and trainer
        loss = C.squared_error(z, label)        # cross_entropy_with_softmax(z, label)      # For training on training data
        label_error = self.mae(z, label)       # C.classification_error(z, label)          # For error on test data
        lr_schedule = C.learning_rate_schedule(params['learning_rate'], C.UnitType.minibatch)
        moment_schedule = C.momentum_schedule([0.99,0.9], 1000)
        learner = C.adam(z.parameters, lr_schedule, minibatch_size = 1, momentum = moment_schedule)
        if not tensorboard_folder is None:
            tensorboard_writer = TensorBoardProgressWriter(freq=10, log_dir=tensorboard_folder + str(modelnumber) + "/", model=z)
            self.model[modelnumber]['tensorboard_writer'] = tensorboard_writer
            self.model[modelnumber]['trainer'] = C.Trainer(z, (loss, label_error), [learner], tensorboard_writer)
        else:
            self.model[modelnumber]['trainer'] = C.Trainer(z, (loss, label_error), [learner])
        self.model[modelnumber]['model'] = z



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
        chunk = self._load_data(meters)

        # B) Prepare the data
        chunk = self._prepare_data(chunk, ext_dataset)        

        # C) Create the model
        horizons = self.model.params['models']
        for horizon in range(len(horizons)):
            self._model_creation(horizon, tensorboard_folder)

        # D) Arrange the labels
        train_test_mask = np.random.rand(len(chunk)) < 0.8
        self.model.labelscaler = scaler = StandardScaler()
        labels = np.asarray(chunk[[('power','active')]], dtype = "float32")
        labels = scaler.fit_transform(labels)
        training_labels = labels[train_test_mask]
        testing_labels = labels[~train_test_mask]
        
        self.model.featurescaler = scaler = StandardScaler() 
        chunk = chunk.dropna()
        chunk.loc[:,:] = scaler.fit_transform(chunk.dropna())

        # Go through all horizons
        for j, horizon in enumerate(params['models']):
            start = time.time()
            model = self.model[j]

            # E) Arrange the input for model j
            columns = copy.deepcopy(params['external_features'])
            for shift in params['shifts']:
                columns.append(('shifts',str(shift+horizon)))
            columns.extend(params['hourFeatures'] + params['weekdayFeatures'])
            features = np.asarray(chunk[columns], dtype = "float32")
            training_features = features[train_test_mask]
            testing_features = features[~train_test_mask]

            # Do the training batch per batch
            num_minibatches = training_features.shape[0] // params['size_minibatches']
            for mb in range(num_minibatches*self.model.params['epochs']):
                items = np.random.choice(len(training_features), params['size_minibatches'])
                features = np.ascontiguousarray(training_features[items])
                labels = np.ascontiguousarray(training_labels[items])
                    
                # Specify the mapping of input variables in the model to actual minibatch data to be trained with
                
                model['trainer'].train_minibatch({model['input'] : features, model['label'] : labels})                    
                if mb % params['training_progress_output_freq'] == 0:
                    self._track_training_progress(j, mb, testing_features, testing_labels, verbose = verbose)

            model['trainer'].model.save(tensorboard_folder + str(j) + "/model.dnn")
            model['tensorboard_writer'].flush()
            model['trainer'].save_checkpoint(tensorboard_folder + str(j) + "/lstm.checkpoint")
            print("Training took {:.1f} sec".format(time.time() - start))
            
            self.model.store(tensorboard_folder)
            return model


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

        # Determine the indices inside the dataset
        items = pd.Series(index = chunk.index, data= range(len(chunk)))
        items = items.loc[timestamps].values

        # Go through all horizons
        for j, horizon in enumerate(params['models']):
            
            # E) Arrange the input for model j
            columns = copy.deepcopy(params['external_features'])
            for shift in params['shifts']:
                columns.append(('shifts',str(shift+horizon)))
            columns.extend(params['hourFeatures'] + params['weekdayFeatures'])
            features = np.asarray(chunk[columns], dtype = "float32")
            z = self.model[j]['model']
            pred = z(features[items]) #z.eval({ann_features: cur_ann_input, lstm_input: cur_lstm_input})
            forecast[horizon] = self.model.labelscaler.inverse_transform(pred[:, 0])
        
        forecast = forecast.transpose()
        forecast['horizon'] = forecast.index.values * pd.Timedelta(params['resolution'])
        forecast = forecast.set_index('horizon')
        return forecast
       

                    
    def _track_training_progress(self, j, mb, testing_features, testing_labels, verbose = False):
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

        selection = np.random.choice(len(testing_features), 100)
        training_loss = self.model[j]['trainer'].previous_minibatch_loss_average
        eval_error = self.model[j]['trainer'].test_minibatch({self.model[j]['input'] : testing_features[selection], self.model[j]['label'] : testing_labels[selection]})
        eval_error = eval_error * self.model.labelscaler.scale_[0]

        self.model[j]['plotdata']["batchnumber"].append(mb)
        self.model[j]['plotdata']["loss"].append(training_loss)
        self.model[j]['plotdata']["error"].append(eval_error)
        if verbose: 
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}".format(mb, training_loss, eval_error))
            
        

