from statsmodels.tsa.arima_model import ARIMA
import cntk as C
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

class AnnForecasterModel(Sequence):
    params = {
        # The features which are used as input
        'externalFeatures': [('temperature', '')],#, ('national', ''), ('school', '')],#, ('dewpoint', '')], #, 'time'
        
        # How the daytime is regarded
        'hourFeatures': [('hour', '00-06'), ('hour', "06-09"), ('hour', "09-12"), ('hour', "12-15"), ('hour', "15-18"), ('hour', "18-21"), ('hour', "21-24")],

        # Shifts are the prev days
        'shifts': list(range(12,36)) + [12 + cur*24 for cur in range(7)], #+ [cur*24*7 for cur in range(4)],
        
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
        'debug': False
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
             'plotdata': {"batchsize":[], "loss":[], "error":[]}
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
            plt.plot(self[i]['plotdata']["batchsize"], self[i]['plotdata']["loss"], 'b--')
            plt.plot(self[i]['plotdata']["batchsize"], pd.rolling_mean(pd.Series(self.model[0]['plotdata']["loss"]),20).values, 'b--')
        plt.xlabel('Minibatch number')
        plt.ylabel('Loss')
        plt.title('Minibatch run vs. Training loss ')
        
        plt.subplot(212)
        for i in myrange:
            plt.plot(self[i]['plotdata']["batchsize"], self[i]['plotdata']["error"], 'r--')
        plt.xlabel('Minibatch number')
        plt.ylabel('Label Prediction Error')
        plt.title('Minibatch run vs. Label Prediction Error ')

        plt.show()




class AnnForecaster(Forecaster):
    """This is a forecaster based on a distinct artificial neural network (ANN) 
    for each of the the forecasting distances.
    ----------
    model_class : The model type, which belonging to the forecaster.
    
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

        section = TimeFrame(start=pd.Timestamp("1.1.2016", tz = 'UTC'), end = pd.Timestamp("30.05.2016", tz = 'UTC'))
        sections = TimeFrameGroup([TimeFrame(start=pd.Timestamp("1.1.2016", tz = 'UTC'), end = pd.Timestamp("30.05.2016", tz = 'UTC'))])
        #powerflow = meters.power_series_all_data(verbose = True, sample_period=3600, sections=sections).dropna()
        powerflow = pckl.load(open("./ForecastingBenchmark.pckl", "rb"))        
        # Kommt von 820
        extData = pd.DataFrame() 
        if len(self.model.params['externalFeatures']) > 0: 
            extData = extDataSet.get_data_for_group('820', section, 60*60, self.model.params['externalFeatures'])[1:]

        # Create the model
        modelnumber = self.model.params['amount_of_models']
        for forecasthorizont in range(modelnumber):
            self._model_creation(forecasthorizont)

        #Initialize the parameters for the trainer, we will train in large minibatches in sequential order
        columns = copy.deepcopy(self.model.params['externalFeatures'])
        power_column = [('power', 'active')] #meters._convert_physical_quantity_and_ac_type_to_cols(physical_quantity = 'power', ignore_missing_columns = True)['cols']
        columns += power_column
        #section = meters.get_timeframe(intersection_instead_union=True)
        
        #for chunk in meters.load(sample_period=900, cols=columns, ignore_missing_columns=True):
            #chunk = chunk[chunk[('power','active')].notnull()]
        for i in range(1):
            chunk = pd.concat([extData, powerflow], axis=1) #('temperature',''):extData[('temperature','')]})
            self._addShiftsToChunkAndReduceToValid(chunk)
            self._addTimeRelatedFeatures(chunk)

            # Go through all models
            min_max_scaler = StandardScaler()
            training_labels = np.asarray(chunk[[('power','active')]], dtype = "float32")
            training_labels = min_max_scaler.fit_transform(training_labels)
            for j in range(self.model.params['amount_of_models']):
                # Define input for model j
                columns = copy.deepcopy(self.model.params['externalFeatures'])
                for shift in copy.deepcopy(self.model.params['shifts']):
                    columns.append(('shifts',str(shift+j)))
                columns.extend(self.model.params['hourFeatures'] + self.model.params['weekdayFeatures'])
                training_features = np.asarray(chunk[columns], dtype = "float32")

                # Scale the data
                training_features = min_max_scaler.fit_transform(training_features) # Later use inverse_transform

                # Do the training batch per batch
                num_minibatches = training_features.shape[0] // self.model.params['size_minibatches']
                for i in range(num_minibatches*self.model.params['epochs']):
                    items = np.random.choice(len(training_features), self.model.params['size_minibatches'])
                    features = np.ascontiguousarray(training_features[items]) #tf[i%num_minibatches])
                    labels = np.ascontiguousarray(training_labels[items]) #tl[i%num_minibatches])
                    
                    # Specify the mapping of input variables in the model to actual minibatch data to be trained with
                    model = self.model[j]
                    model['trainer'].train_minibatch({model['input'] : features, model['label'] : labels})                    
                    if i%self.model.params['training_progress_output_freq'] == 0:
                        print(labels.mean())
                        self._track_training_progress(j, i, verbose = verbose)
                    if False:
                        self.model.plot_training_progress(True)



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



    def _addShiftsToChunkAndReduceToValid(self, chunk):
        '''
        This function takes the current chunk of the meter and adapts it sothat
        it can be used for training. That means it extends the dataframe by the 
        missing features we want to learn.
        '''        
        # Determine the shifts that are required. All in memory since only a few values
        all_shifts = set()
        for shift in self.model.params['shifts']:
            all_shifts.update(range(shift, shift+self.model.params['amount_of_models']))

        # Compute the stock being up (1) or down (0) over different day offsets compared to current dat closing price
        for i in all_shifts:
            chunk[('shifts', str(i))] = chunk[('power','active')].shift(i)    # i: number of look back days

        chunk.drop(chunk.index[chunk[('shifts',str(self.model.params['shifts'][-1]))].isnull()], inplace=True)
           
    
    def _addTimeRelatedFeatures(self, chunk):
        '''
        Add the time related features
        '''
        idxs = self.model.params['weekdayFeatures']
        for idx in idxs:
            weekday = idx[1]
            days_of_group = set(range(int(weekday[0]),int(weekday[2])))
            chunk[idx] = chunk.index.weekday
            chunk[idx] = chunk[idx].apply(lambda e, dog=days_of_group: e in dog)
        idxs = self.model.params['hourFeatures']
        for idx in idxs:
            hour = idx[1]
            hours_of_group = set(range(int(hour[:2]),int(hour[3:])))
            chunk[idx] = chunk.index.hour
            chunk[idx] = chunk[idx].apply(lambda e, hog = hours_of_group: e in hog)


    def _model_creation(self, forecasthorizont):
        '''
        Build the ANN

        Parameters:
        forecasthorizont: The amount of steps into the future of this model. Don't need it!
        '''

        # Define input and output which his fixed
        params = self.model.params
        input_dim = len(params['externalFeatures']) + len(params['shifts']) + len(params['hourFeatures']) + len(params['weekdayFeatures'])
        input_dynamic_axes = [C.Axis.default_batch_axis()]
        input = C.input_variable(input_dim)#, dynamic_axes=input_dynamic_axes)
        self.model[forecasthorizont]['input'] = input
        label = C.input_variable(1)#, dynamic_axes=input_dynamic_axes)
        self.model[forecasthorizont]['label'] = label
        
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
        label_error = self.mape(z, label)       # C.classification_error(z, label)          # For error on test data
        lr_schedule = C.learning_rate_schedule(0.001, C.UnitType.minibatch)
        moment_schedule = C.momentum_schedule([0.99,0.9], 1000)
        learner = C.adam(z.parameters, lr_schedule, minibatch_size = 1, momentum = moment_schedule)
        self.model[forecasthorizont]['trainer'] = C.Trainer(z, (loss, label_error), [learner]) # label_error



                    
    def _track_training_progress(self, j, mb, verbose = False):
        '''
        Calculates the error per minibatch. 
        Outputs the error with a given frequency.
        '''
        training_loss = self.model[j]['trainer'].previous_minibatch_loss_average
        eval_error = self.model[j]['trainer'].previous_minibatch_evaluation_average
        self.model[j]['plotdata']["loss"].append(training_loss)
        self.model[j]['plotdata']["error"].append(eval_error)
        self.model[j]['plotdata']["batchsize"].append(mb)
        if verbose: 
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error*100))
            
        

