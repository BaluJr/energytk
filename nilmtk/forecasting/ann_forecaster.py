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

    ATTENTION: Currently not checking for valid horizonts.Only until 1day supported currently.

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
        'models': list(range(16, 98, 16)),
        
        # Amount of data used for validation
        'validation_quota': 0.1,
        
        # Learning Rate
        'learning_rate': 0.015,

        # How many errors are taken together until training step
        'size_minibatches': 10,

        # How often each minibatch is trained
        'epochs': 250,

        # Resolution of one step
        'resolution': '15m',


        # The features which are used as input
        'external_features': [],#[('temperature', '')],#, ('national', ''), ('school', '')],#, ('dewpoint', '')], #, 'time'
        
        # How the daytime is regarded
        'hourFeatures': [('hour', '00-01'), ('hour', "01-02"), ('hour', "02-03"), ('hour', "03-04"), ('hour', "04-05"),
                     ('hour', "05-06"), ('hour', "06-07"), ('hour', "07-08"), ('hour', "08-09"), ('hour', "09-10"),
                     ('hour', "10-11"), ('hour', "11-12"), ('hour', "12-13"), ('hour', "13-14"), ('hour', "14-15"),
                     ('hour', "15-16"), ('hour', "16-17"), ('hour', "17-18"), ('hour', "18-19"), ('hour', "19-20"),
                     ('hour', "20-21"), ('hour', "21-22"), ('hour', "22-23"), ('hour', "23-24")],

        # Shifts relative to the previous available
        'shifts_lastentry_based': list(range(0, 17, 4)),

        # Shifts relative to the target
        'shifts_target_based': [cur*4*24 for cur in range(1,7)] + [cur*4*24*7 for cur in range(1,5)],

        # How the weekdays are paired
        'weekdayFeatures': [('week', "0-1"),('week', "1-2"),('week', "2-3"),('week', "3-4"),('week', "4-5"),('week', "5-6"),('week',"6-7")],
        
        # Number of hidden layers of the ANN
        'num_hidden_layers': 2,

        # Dimensionality of the ANN's hidden layers
        'hidden_layers_dim': 30,


        # Whether to give debug output during training.
        'debug': False,
        
        # Define output
        'training_progress_output_freq': 10900,
    }


    def __init__(self, path = None, args = {}):
        '''
        The paramters are set to the optional paramters handed in by the constructor.
        '''
        self.untrained = True
        self.target_folder = None
        if not path is None:
            self.target_folder = path
            self.load()
        else:
            for arg in args:
                self.params[arg] = args[arg]
            self.model = [
                {   'trainer':None,
                    'input':None,
                    'labels':None,
                    'model':None,
                    'plotdata': {"batchnumber":[], "loss":[], "error":[], "avg_power":[]}
                } for _ in range(len(self.params['models']))]


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

        
    def store(self):
        ''' Stores the model in a certain folder.
        A folder is given as certain models cannot be stored within a single file.
        '''

        # Cancel if folder already exists and contains elements
        folder = self.target_folder
        if os.path.isdir(folder):
            pass
            #if len(os.listdir(folder)) != 0:
            #    raise Exception("Folder must be empty.")
        else:
            os.mkdir(folder)

        # Store the cntk model and remove
        for i, cur in enumerate(self):
            cur['trainer'].model.save(folder + str(i) + "/model.dnn")
            cur["trainer"].save_checkpoint(folder + str(i) + "/.trainer")
            cur['tensorboard_writer'].flush()
            open(folder + str(i) + "/" + str(self.params['models'][i]) ,"w+").close()

        # Store the basemodel
        all_to_store = {}
        models_to_store = []
        for cur in self:
            to_store = {}
            for cur_to_store in cur:
                if cur_to_store in ['trainer', 'input','label', 'model', 'tensorboard_writer']:
                    continue
                to_store[cur_to_store] = cur[cur_to_store]
            models_to_store.append(to_store)
        all_to_store["model"] = models_to_store
        all_to_store["labelscaler"] = self.labelscaler
        all_to_store["featurescaler"] = self.featurescaler
        all_to_store["params"] = self.params
        all_to_store["target_folder"] = self.target_folder
        all_to_store["untrained"] = self.untrained
        pckl.dump(all_to_store, open(folder+ "basemodel", "wb"))

        # Store meta information
        tmp = open(folder + "model.txt" ,"w+")
        tmp.write(str(self.params))
        tmp.write("\n##################\n")
        tmp.write(str(list(list(map(lambda m: m['plotdata'], self)))))
        tmp.close()
    

    def load(self):
        ''' Loads the model from the designated folder.
        A folder is given as certain models cannot be stored within a single file.
        
        folder: str
            Folder in which the model can be found
        '''
        # Cancel if folder already exists and contains elements
        folder = self.target_folder
        if not folder.endswith("/"):
            folder += "/"
        if not os.path.isdir(folder):
            raise Exception("The designated folder does not exist.")

        # First reload the base model
        model_data = pckl.load(open(folder+ "basemodel", "rb"))
        self.model = model_data["model"]
        self.labelscaler = model_data["labelscaler"]
        self.featurescaler = model_data["featurescaler"]
        self.params = model_data["params"]
        self.target_folder = model_data["target_folder"]
        self.untrained = model_data["untrained"]

        # Load the cntk model and remove (not working yet)
        for i, cur in enumerate(self.model):
            cur['model'] = m = C.load_model(folder + str(i) + '/model.dnn')
            cur['input'] = cur['model'].arguments[0]  # cur['model']['ann_input']
            cur["label"] = m(cur['model'].arguments[0])
            #cur["trainer"].load_checkpoint(folder + str(i) + "/.trainer")
            #cur["trainer"].restore_from_checkpoint(folder + str(i) + "/.trainer")





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

    def __init__(self, model = None):
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

    
    def _prepare_data(self, chunk, ext_dataset, for_forecasting):
        ''' Adds the additional featues to the data
        
        Parameters
        ----------
        chunk: pd.DataFrame
            Chunk to extend
        ext_dataset: nilmtk.DataSet or str
            The External Dataset containing the fitting data.
            Alternatively a path to a PickleFile.
        for_forecasting: bool
            Whether it is for forecasting. This changes how the shifts are arranged.
            This is due to the fact that during training the given index is oriented at the
            forecasting target and for the forecasting it is oriented at the last available
            index.
        
        Returns
        -------
        chunk: pd.DataFrame
            The extended power series
        '''
        params = self.model.params
        power_column = [('power', 'active')] #meters._convert_physical_quantity_and_ac_type_to_cols(physical_quantity = 'power', ignore_missing_columns = True)['cols']
        chunk = self._add_time_related_features(chunk, params['weekdayFeatures'], params['hourFeatures'])
        chunk = self._add_external_data(chunk, ext_dataset, self.model.params['external_features'])

        #allshifts = params['shifts_target_based'] + list(map(lambda e: e + , params['shifts_lastentry_based']))
        chunk = self._addShiftsToChunkAndReduceToValid(chunk, params['shifts_target_based'], params['shifts_lastentry_based'], params['models'], for_forecasting)
        return chunk

    def _model_creation(self, modelnumber, target_folder = None):
        ''' Create the ANN model for time series forecasting 

        Parameters
        ----------
        modelnumber: int
            The number of model to create. Needed to look up its params.
        target_folder: str
            path where the tensorboard output shall be stored.
        '''
        

        # Define input and output which his fixed
        params = self.model.params
        input_dim = len(params['external_features']) + len(params['hourFeatures']) + len(params['weekdayFeatures']) \
                    + len(params['shifts_target_based']) + len(params['shifts_lastentry_based'])
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

        # Define output
        #self.model[modelnumber]['label'] = l = C.input_variable(1, dynamic_axes=z.dynamic_axes, name="y")
        self.model[modelnumber]['model'] = z


    def _setup_metric_and_trainer(self, modelnumber, z, l, target_folder=None):
        """ Create the trainer for a model.

        Parameters
        ----------
        modelnumber: int
            Forecasting horizon this model is trained for.
        z: Cntk.model
            The model to train.
        l: str
            path where the tensorboard output shall be stored.
        target_folder: str
            path where the tensorboard output shall be stored.
        """
        params = self.model.params

        loss = C.squared_error(z, l)        # cross_entropy_with_softmax(z, label)      # For training on training data
        label_error = self.mae(z, l)       # C.classification_error(z, label)          # For error on test data
        lr_schedule = C.learning_rate_schedule(params['learning_rate'], C.UnitType.minibatch)
        moment_schedule = C.momentum_schedule([0.99,0.9], 1000)
        learner = C.adam(z.parameters, lr_schedule, momentum = moment_schedule)
        tensorboard_writer = TensorBoardProgressWriter(freq=10, log_dir=target_folder + str(modelnumber) + "/", model=z)
        self.model[modelnumber]['tensorboard_writer'] = tensorboard_writer
        self.model[modelnumber]['trainer'] = C.Trainer(z, (loss, label_error), [learner], tensorboard_writer)


    def train(self, meters, ext_dataset, section = None, target_folder = None, verbose = False):
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
        target_folder: str
            path where the tensorboard output shall be stored.
        verbose: bool
            Whether additional output shall be printed during training.
        '''
        global CANCELLED
        CANCELLED = False

        params = self.model.params
        if target_folder != None:
            if not target_folder.endswith("/"):
                target_folder += "/"
            self.model.target_folder = target_folder
        else:
            target_folder = self.model.target_folder
        self.model.untrained = False

        # A) Load the data
        chunk = self._load_data(meters)

        # B) Prepare the data
        chunk = self._prepare_data(chunk, ext_dataset, for_forecasting=False)
        chunk = chunk.dropna()

        # C) Create the model
        horizons = self.model.params['models']
        for horizon in range(len(horizons)):
            self._model_creation(horizon, target_folder)
            self._setup_metric_and_trainer(horizon, self.model[horizon]['model'], self.model[horizon]['label'], target_folder=target_folder)

        # D) Arrange the labels
        train_test_mask = np.random.rand(len(chunk)) < 0.8
        self.model.labelscaler = scaler = StandardScaler()
        labels = np.asarray(chunk[[('power','active')]], dtype = "float32")
        labels = scaler.fit_transform(labels)
        training_labels = labels[train_test_mask]
        testing_labels = labels[~train_test_mask]
        
        self.model.featurescaler = scaler = StandardScaler()
        chunk[('power', 'active')] = scaler.fit_transform(chunk[[('power','active')]])
        chunk['shifts'] = scaler.transform(chunk['shifts'])

        # Repeat epochs
        num_minibatches = train_test_mask.sum() // params['size_minibatches']
        epoch = 0
        while not CANCELLED:
            # Go through all horizons
            start = time.time()
            for j, horizon in enumerate(params['models']):
                model = self.model[j]

                # E) Arrange the input for model j
                columns = copy.deepcopy(params['external_features'])
                for shift in params['shifts_target_based']:
                    columns.append(('shifts',str(shift)))
                for shift in params['shifts_lastentry_based']:
                    columns.append(('shifts',str(shift+horizon)))
                columns.extend(params['hourFeatures'] + params['weekdayFeatures'])
                features = np.asarray(chunk[columns], dtype = "float32")
                training_features = features[train_test_mask]
                testing_features = features[~train_test_mask]

                # Do the training batch per batch
                for mb in range(num_minibatches):
                    items = np.random.choice(len(training_features), params['size_minibatches'])
                    features = np.ascontiguousarray(training_features[items])
                    labels = np.ascontiguousarray(training_labels[items])

                    # Specify the mapping of input variables in the model to actual minibatch data to be trained with
                    model['trainer'].train_minibatch({model['input'] : features, model['label'] : labels})
                    if mb % params['training_progress_output_freq'] == 0:
                        self._track_training_progress(j, mb, testing_features, testing_labels, verbose = verbose)

            # After each epoch, store the progress
            self.model.store()
            print("Epoch {0} took {1:.1f} sec".format(epoch, time.time() - start))

            epoch += 1
            if (params['epochs'] > 0 and epoch >= params['epochs']):
                CANCELLED = True

        self.model.store()
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
        chunk = self._prepare_data(chunk, ext_dataset, for_forecasting=True)
        chunk['shifts'] = self.model.featurescaler.transform(chunk['shifts'])


        # Determine the indices inside the dataset
        items = pd.Series(index = chunk.index, data= range(len(chunk)))
        items = items.loc[timestamps].values

        # Go through all horizons
        for j, horizon in enumerate(params['models']):
            
            # E) Arrange the input for model j
            columns = copy.deepcopy(params['external_features'])
            for shift in params['shifts_target_based']:
                columns.append(('shifts',str(shift-horizon)))
            for shift in params['shifts_lastentry_based']:
                columns.append(('shifts',str(shift)))
            columns.extend(params['hourFeatures'] + params['weekdayFeatures'])
            features = np.asarray(chunk[columns], dtype = "float32")
            z = self.model[j]['model']
            pred = z(features[items]) #z.eval({ann_features: cur_ann_input, lstm_input: cur_lstm_input})
            forecast[horizon] = self.model.labelscaler.inverse_transform(pred[:, 0])
        
        forecast = forecast.transpose()
        forecast['horizon'] = forecast.index.values * pd.Timedelta(params['resolution'])
        forecast = forecast.set_index('horizon')
        return forecast
       

                    
    def _track_training_progress(self, modelnumber, mb, testing_features, testing_labels, verbose = False):
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
        model =  self.model[modelnumber]

        training_loss = model['trainer'].previous_minibatch_loss_average
        eval_error = model['trainer'].test_minibatch({model['input'] : testing_features[selection], model['label'] : testing_labels[selection]})
        eval_error = eval_error * self.model.labelscaler.scale_[0]
        avg_power = (self.model.labelscaler.inverse_transform(testing_labels[selection])).mean()

        model['plotdata']["batchnumber"].append(mb)
        model['plotdata']["loss"].append(training_loss)
        model['plotdata']["error"].append(eval_error)
        model['plotdata']["avg_power"].append(avg_power)
        if verbose:
            print("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}, AvgPower: {3}".format(mb, training_loss, eval_error, avg_power))

        

