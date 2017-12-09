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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class AnnForecasterModel(Sequence):
    params = {
        # The features which are used as input
        'externalFeatures': [('temperature', '')],#, ('national', ''), ('school', '')],#, ('dewpoint', '')], #, 'time'
        
        # How the daytime is regarded
        'hourFeatures': [('hour', '00-03'), ('hour', '03-06'), ('hour', "06-09"), ('hour', "09-12"), ('hour', "12-15"), ('hour', "15-18"), ('hour', "18-21"), ('hour', "21-24")],

        # Shifts are the prev days
        'shifts': list(range(96,192,4)) + [cur*96 for cur in range(3,7)], #+ [cur*24*7 for cur in range(4)],
        
        # How the weekdays are paired
        'weekdayFeatures': [('week', "0-5"),('week', "5-6"),('week',"6-7")],

        # Describes the amount of models which is used to do the forecasting.
        # Each additional model is trained for a time, one step further in the
        # future.
        # Setting this to 1 means that only one 15 minute prediction is performed.
        # To create a 24 hour forecasting series the amount_of_models has to be 
        # set to 96.
        'amount_of_models': 1,

        # Architecture of the ANN
        'num_hidden_layers': 2,
        'hidden_layers_dim': 50,

        # How many errors are taken together until training step
        'size_minibatches': 25,

        # Define output
        'training_progress_output_freq': 40,

        # How often each minibatch is trained
        'epochs': 10000,

        # The target storage to store the data to
        'forecasting_storage': "C:\\Users\\maxim_000\\Documents\\InformatikStudium_ETH\\Masterarbeit\\6_Data\\Tmp\\forecasting.hdf",

        # Whether to give debug output during training.
        'debug': False,
        
        # Amount of data used for validation
        'validation_quota': 0.1,

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
             'plotdata': {"batchnumber":[], "loss":[], "error":[]}
            }] * self.params['amount_of_models']

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
        myrange = range(self.params['amount_of_models'])

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
        """
        super(AnnForecaster, self).__init__(model)

    def mape(self, z, l):
        return C.reduce_mean(C.abs(z - l)/l) 
    
    def mae(self, z, l):
        return C.reduce_mean(C.abs(z - l))

    def get_mse(X,Y,labeltxt):
        result = 0.0
        for x1, y1 in next_batch(X, Y, labeltxt):
            eval_error = self.model.trainer.test_minibatch({x : x1, l : y1})
            result += eval_error
        return result/len(X[labeltxt])

    def generate_random_data_sample(sample_size, feature_dim, num_classes):
        # Create synthetic data using NumPy.
        Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

        # Make sure that the data is separable
        X = (np.random.randn(sample_size, feature_dim)+3) * (Y+1)
        X = X.astype(np.float32)
        # converting class 0 into the vector "1 0 0",
        # class 1 into vector "0 1 0", ...
        class_ind = [Y==class_number for class_number in range(num_classes)]
        Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
        return X, Y


    def train(self, meters, extDataSet, verbose = False):
        '''
        Does the training of the neural network. Each load chunk is translated into 
        multiple minibatches used for training the network.
        '''
        params = self.model.params

        # A) Load the data
        section = TimeFrame(start=pd.Timestamp("1.1.2016", tz = 'UTC'), end = pd.Timestamp("15.03.2017", tz = 'UTC'))
        sections = TimeFrameGroup([TimeFrame(start=pd.Timestamp("1.1.2016", tz = 'UTC'), end = pd.Timestamp("30.05.2016", tz = 'UTC'))])
        powerflow = pckl.load(open("./ForecastingBenchmark15min.pckl", "rb"))
        powerflow = pd.DataFrame(powerflow)
        #sections = TimeFrameGroup([meters.get_timeframe(intersection_instead_union=True)])
        #powerflow = meters.power_series_all_data(verbose = True, sample_period=3600, sections=sections).dropna()
        #powerflow = next(meters.load(verbose = True, sample_period=3600, sections=sections)).dropna()
        
        # C) Create the model
        modelnumber = self.model.params['amount_of_models']
        for forecasthorizont in range(modelnumber):
            self._model_creation(forecasthorizont)

        # B) Prepare the data
        power_column = [('power', 'active')] #meters._convert_physical_quantity_and_ac_type_to_cols(physical_quantity = 'power', ignore_missing_columns = True)['cols']
        chunk = self._add_time_related_features(powerflow, params['weekdayFeatures'], params['hourFeatures'])
        chunk = self._add_external_data(chunk, extDataSet, self.model.params['externalFeatures'])
        chunk = self._addShiftsToChunkAndReduceToValid(chunk, params['shifts'], params['amount_of_models'])
        

        # D) Arrange the labels
        train_test_mask = np.random.rand(len(chunk)) < 0.8
        self.model.labelscaler = scaler = StandardScaler()
        labels = np.asarray(chunk[[('power','active')]], dtype = "float32")
        labels = scaler.fit_transform(labels)
        training_labels = labels[train_test_mask]
        testing_labels = labels[~train_test_mask]

        # Go through all horizonts
        for j in range(params['amount_of_models']):
            # E) Arrange the input for model j
            columns = copy.deepcopy(params['externalFeatures'])
            for shift in copy.deepcopy(params['shifts']):
                columns.append(('shifts',str(shift+j)))
            columns.extend(params['hourFeatures'] + params['weekdayFeatures'])
            features = np.asarray(chunk[columns], dtype = "float32")
            self.model.featurescaler = scaler = StandardScaler()
            features = scaler.fit_transform(features)
            training_features = features[train_test_mask]
            testing_features = features[~train_test_mask]

            # Do the training batch per batch
            num_minibatches = training_features.shape[0] // params['size_minibatches']
            for mb in range(num_minibatches*self.model.params['epochs']):
                items = np.random.choice(len(training_features), params['size_minibatches'])
                features = np.ascontiguousarray(training_features[items])
                labels = np.ascontiguousarray(training_labels[items])
                    
                # Specify the mapping of input variables in the model to actual minibatch data to be trained with
                model = self.model[j]
                model['trainer'].train_minibatch({model['input'] : features, model['label'] : labels})                    
                if mb % params['training_progress_output_freq'] == 0:
                    self._track_training_progress(j, mb, testing_features, testing_labels, verbose = verbose)

            model['trainer'].save_checkpoint("Test.checkpoint", self.model[0]['plotdata'])


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

        result = pd.DataFrame(columns=['power','active'])
        for i in amount_of_models:
            model = self.model[i].trainers

        test_features = np.ascontiguousarray(test_data[predictor_names], dtype = "float32")
        test_labels = np.ascontiguousarray(test_data[["next_day","next_day_opposite"]], dtype="float32")

        avg_error = self.model[i].trainer.test_minibatch({input : test_features, label : test_labels})
        if verbose:
            print("Average error: {0:2.2f}%".format(avg_error * 100))
        
        # Write data to target file: Brauche ich glaueb ich erst mal nicht, da nicht so viele Ergebnisse
        # forecastingStore = HDFDataStore("C:\\Users\\maxim_000\\Documents\\InformatikStudium_ETH\\Masterarbeit\\6_Data\\Tmp\\forecasting.hdf'", 'w')



    def _model_creation(self, forecasthorizon):
        '''
        Build the ANN

        Parameters:
        forecasthorizon: The amount of steps into the future of this model. Don't need it!
        '''

        # Define input and output which his fixed
        params = self.model.params
        input_dim = len(params['externalFeatures']) + len(params['shifts']) + len(params['hourFeatures']) + len(params['weekdayFeatures'])
        input_dynamic_axes = [C.Axis.default_batch_axis()]
        self.model[forecasthorizon]['input'] = input = C.input_variable(input_dim)#, dynamic_axes=input_dynamic_axes)
        self.model[forecasthorizon]['label'] = label = C.input_variable(1)#, dynamic_axes=input_dynamic_axes)
        
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
        lr_schedule = C.learning_rate_schedule(0.01, C.UnitType.minibatch)
        moment_schedule = C.momentum_schedule([0.99,0.9], 1000)
        learner = C.adam(z.parameters, lr_schedule, minibatch_size = 1, momentum = moment_schedule)
        tensorboard_writer = TensorBoardProgressWriter(freq=10, log_dir=self.model.params['tensorboard_dict'], model=z)
        self.model[forecasthorizon]['trainer'] = C.Trainer(z, (loss, label_error), [learner], tensorboard_writer) # label_error



                    
    def _track_training_progress(self, j, mb, testing_features, testing_labels, verbose = False):
        '''
        Calculates the error per minibatch. 
        Outputs the error with a given frequency.
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
            
        

