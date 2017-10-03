from statsmodels.tsa.arima_model import ARIMA
import numpy as np
from pandas import DataFrame
from .clusterer import Clusterer

class CorrelationClustererModel(object):
    params = {
        # The features which are used as input
        'externalFeatures': ['temperature', 'dewpoint', 'national', 'school'], #, 'time'
        'shifts': list(range(1,5)),#48)), # +[cur*96 for cur in range(7)] + [cur*96*7 for cur in range(4)],

        # Describes the amount of models which is used to do the forecasting.
        # Each additional model is trained for a time, one step further in the
        # future.
        # Setting this to 1 means that only one 15 minute prediction is performed.
        # To create a 24 hour forecasting series the amount_of_models has to be 
        # set to 96.
        'amount_of_models': 1,
    }



class CorrelationClusterer(Clusterer):
    """
    This is disaggregator which clusters all meters by their correlation 
    towards each other. That means, that meters with similar powerflows will
    end up in a group. The kind of correlation is chosen.
    ----------
    Attributes:
    model_class : The model type, which belonging to the forecaster.
    
    """

    # The related model
    model_class = CorrelationClustererModel

    def __init__(self, model = None):
        """
        Constructor of this class which takes an optional model as input.
        If no model is given, it createsa default one.
        """
        super(CorrelationClusterer, self).__init__(model)

        
    def train(self, meters):
        '''
        Does the training of the neural network. Each load chunk is translated into 
        multiple minibatches used for training the network.
        '''

        # Create the model
        trainer = self._model_creation()
        
        #Initialize the parameters for the trainer, we will train in large minibatches in sequential order
        columns = [('power', 'active'), ('school',''), ('national',''), ('temperature',''), ('dewpoint','')]
        for chunk in meters.load(sample_period=900, cols=columns, ignore_missing_columns=True):
            chunk = chunk[chunk[('power','active')].notnull()]
            self._addShiftsToChunkAndReduceToValid(chunk)

            # Prepare input for ANN 
            training_features = chunk.drop(('power','active'),axis=1)
            training_features = np.asarray(training_features, dtype = "float32")
            training_labels = DataFrame(chunk.loc[:,('power','active')])
            training_labels = np.asarray(training_labels, dtype = "float32")
            num_minibatches = training_features.shape[0] // self.model.params['size_minibatches'] #len(training_data.index) // minibatch_size
            tf = np.array_split(training_features,num_minibatches)
            tl = np.array_split(training_labels, num_minibatches)

            # Do the training batch per batch
            for i in range(num_minibatches* self.model.params['num_passes']):
                features = np.ascontiguousarray(tf[i%num_minibatches])
                labels = np.ascontiguousarray(tl[i%num_minibatches])
    
                # Specify the mapping of input variables in the model to actual minibatch data to be trained with
                trainer.train_minibatch({self.model.input : features, self.model.label : labels})
                batchsize, loss, error = self._track_training_progress(trainer, i, self.model.params['training_progress_output_freq'], verbose = True)
                if not (loss == "NA" or error =="NA"):
                    self.model.plotdata["batchsize"].append(batchsize)
                    self.model.plotdata["loss"].append(loss)
                    self.model.plotdata["error"].append(error)


    def predict(self, horizon):
        '''
        This method uses the learned model to predict the future
        '''
        test_features = np.ascontiguousarray(test_data[predictor_names], dtype = "float32")
        test_labels = np.ascontiguousarray(test_data[["next_day","next_day_opposite"]], dtype="float32")

        avg_error = trainer.test_minibatch({input : test_features, label : test_labels})
        print("Average error: {0:2.2f}%".format(avg_error * 100))



    def _addShiftsToChunkAndReduceToValid(self, chunk):
        '''
        This function takes the current chunk of the meter and adapts it sothat
        it can be used for training. That means it extends the dataframe by the 
        missing features we want to learn.
        '''
        
        # Compute the stock being up (1) or down (0) over different day offsets compared to current dat closing price
        chunk
        for i in self.model.params['shifts']:
            chunk[('shifts', str(i))] = chunk[('power','active')].shift(i)    # i: number of look back days
        chunk.drop(chunk.index[chunk[('shifts',str(self.model.params['shifts'][-1]))].isnull()], inplace=True)


   


    def _model_creation(self):
        '''
        Build the ANN
        '''

        # Define input and output which his fixed
        params = self.model.params
        num_output_classes = 1 
        input_dim = len(params['externalFeatures']) + len(params['shifts'])
        input_dynamic_axes = [C.Axis.default_batch_axis()]
        input = C.input_variable(input_dim, dynamic_axes=input_dynamic_axes)
        self.model.input = input
        label = C.input_variable(num_output_classes, dynamic_axes=input_dynamic_axes)
        self.model.label = label

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
        return C.Trainer(z, (loss, label_error), [C.sgd(z.parameters, lr=lr_per_minibatch)])



                    
    def _track_training_progress(self, trainer, mb, frequency, verbose = False):
        '''

        '''
        training_loss = "NA"
        eval_error = "NA"
        if mb%frequency == 0:
            training_loss = trainer.previous_minibatch_loss_average
            eval_error = trainer.previous_minibatch_evaluation_average
            if verbose: 
                print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error*100))
        return mb, training_loss, eval_error

