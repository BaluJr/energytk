from statsmodels.tsa.arima_model import ARIMA
import cntk as C
import numpy as np
from .forecaster import Forecaster
from pandas import DataFrame
from nilmtk import DataSet, ExternDataSet
from collections.abc import Sequence
import copy

class AnnForecasterModel(Sequence):
    params = {
        # The features which are used as input
        'externalFeatures': [('temperature', ''), ('dewpoint', ''), ('national', ''), ('school', '')], #, 'time'
        'shifts': list(range(1,5)),#48)), # +[cur*96 for cur in range(7)] + [cur*96*7 for cur in range(4)],
        

        # Describes the amount of models which is used to do the forecasting.
        # Each additional model is trained for a time, one step further in the
        # future.
        # Setting this to 1 means that only one 15 minute prediction is performed.
        # To create a 24 hour forecasting series the amount_of_models has to be 
        # set to 96.
        'amount_of_models': 3,

        # Architecture of the ANN
        'num_hidden_layers': 2,
        'hidden_layers_dim': 2,

        # How many errors are taken together until training step
        'size_minibatches': 100,

        # Define output
        'training_progress_output_freq': 1,

        # How often each minibatch is trained
        'num_passes': 100,

        # The target storage to store the data to
        'forecasting_storage': "C:\\Users\\maxim_000\\Documents\\InformatikStudium_ETH\\Masterarbeit\\6_Data\\Tmp\\forecasting.hdf",
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

    def plot_training_progress(model = -1):
        ''' 
        This method plots the training progress which has been recorded during training.

        The model for which the plot shall be drawn. When set to -1 then all are drawn
        '''
        range = range(self.model.params['amount_of_models']) if model == -1 else [model]

        plt.subplot(211)
        plt.figure(1)
        for i in range:
            plt.plot(self.model[i]['plotdata']["batchsize"], self.model[i]['plotdata']["loss"], 'b--')
        plt.xlabel('Minibatch number')
        plt.ylabel('Loss')
        plt.title('Minibatch run vs. Training loss ')
        
        plt.subplot(212)
        for i in range:
            plt.plot(self.model[i]['plotdata']["batchsize"], plotdata[model]["error"], 'r--')
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

        
    def train(self, meters, verbose = False):
        '''
        Does the training of the neural network. Each load chunk is translated into 
        multiple minibatches used for training the network.
        '''

        # Create the model
        modelnumber = self.model.params['amount_of_models']
        for forecasthorizont in range(modelnumber):
            self._model_creation(forecasthorizont)
        #self.model.plotdata = [{"batchsize":[], "loss":[], "error":[]} for i in modelnumber]

        #Initialize the parameters for the trainer, we will train in large minibatches in sequential order
        columns = copy.deepcopy(self.model.params['externalFeatures'])
        power_column = meters._convert_physical_quantity_and_ac_type_to_cols(physical_quantity = 'power', ignore_missing_columns = True)['cols']
        columns += power_column
        section = meters.get_timeframe(intersection_instead_union=True)
        for chunk in meters.load(sample_period=900, cols=columns, ignore_missing_columns=True):
            chunk = chunk[chunk[('power','active')].notnull()]
            self._addShiftsToChunkAndReduceToValid(chunk)

            # Go through all models
            training_labels = DataFrame(chunk.loc[:,('power','active')])
            training_labels = np.asarray(training_labels, dtype = "float32")    
            for j in range(self.model.params['amount_of_models']):
                # Define input for model j
                columns = copy.deepcopy(self.model.params['externalFeatures'])
                for shift in copy.deepcopy(self.model.params['shifts']):
                    columns.append(('shifts',str(shift+j)))
                training_features = chunk[columns]#drop(('power','active'),axis=1)
                training_features = np.asarray(training_features, dtype = "float32")
                num_minibatches = training_features.shape[0] // self.model.params['size_minibatches'] #len(training_data.index) // minibatch_size
                tf = np.array_split(training_features,num_minibatches)
                tl = np.array_split(training_labels, num_minibatches)

                # Do the training batch per batch
                for i in range(num_minibatches*self.model.params['num_passes']):
                    features = np.ascontiguousarray(tf[i%num_minibatches])
                    labels = np.ascontiguousarray(tl[i%num_minibatches])
    
                    # Specify the mapping of input variables in the model to actual minibatch data to be trained with
                    model = self.model[j]
                    model['trainer'].train_minibatch({model['input'] : features, model['label'] : labels})                    
                    if i%self.model.params['training_progress_output_freq'] == 0:
                        self._track_training_progress(j, i, verbose = verbose)



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
        all_shifts = {1}
        for shift in self.model.params['shifts']:
            all_shifts.update(range(shift, shift+self.model.params['amount_of_models']))

        # Compute the stock being up (1) or down (0) over different day offsets compared to current dat closing price
        for i in all_shifts:
            chunk[('shifts', str(i))] = chunk[('power','active')].shift(i)    # i: number of look back days

        chunk.drop(chunk.index[chunk[('shifts',str(self.model.params['shifts'][-1]))].isnull()], inplace=True)
           


    def _model_creation(self, forecasthorizont):
        '''
        Build the ANN

        Parameters:
        forecasthorizont: The amount of steps into the future of this model. Don't need it!
        '''

        # Define input and output which his fixed
        params = self.model.params
        num_output_classes = 1 
        input_dim = len(params['externalFeatures']) + len(params['shifts'])
        input_dynamic_axes = [C.Axis.default_batch_axis()]
        input = C.input_variable(input_dim, dynamic_axes=input_dynamic_axes)
        self.model[forecasthorizont]['input'] = input
        label = C.input_variable(num_output_classes, dynamic_axes=input_dynamic_axes)
        self.model[forecasthorizont]['label'] = label

        # Create the network
        h = input
        with C.layers.default_options(init = C.glorot_uniform()):
            for i in range(params['num_hidden_layers']):
                h = C.layers.Dense(params['hidden_layers_dim'], 
                                    activation = C.relu)(h)
            z = C.layers.Dense(num_output_classes, activation=None)(h)   

        # Define error metric and trainer
        loss = C.cross_entropy_with_softmax(z, label)
        label_error = C.classification_error(z, label)
        lr_per_minibatch = C.learning_rate_schedule(0.125,C.UnitType.minibatch)
        self.model[forecasthorizont]['trainer'] = C.Trainer(z, (loss, label_error), [C.sgd(z.parameters, lr=lr_per_minibatch)])



                    
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
            
        

