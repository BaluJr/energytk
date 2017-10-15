from matplotlib import pyplot as plt
import math
import numpy as np
import os
import pandas as pd
import random
import time
from .forecaster  import Forecaster

import cntk as C
from cntk.layers import Convolution, Sequential, Dense, MaxPooling, AveragePooling
from cntk.layers.typing import Tensor, Sequence
from cntk.learners import learning_rate_schedule, momentum_schedule

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve



class CnnForecasterModel():
    params = {
        # The target storage to store the data to
        'forecasting_storage': "C:\\Users\\maxim_000\\Documents\\InformatikStudium_ETH\\Masterarbeit\\6_Data\\Tmp\\forecasting.hdf",
        
        # Input Config
        'input_dim': (1, 28, 28),

        'input_dim_reader': 28 * 28,
       
        'output_dim': 10,

        # Initialize the parameters for the trainer
        'minibatch_size': 64,

        'num_samples_per_sweep': 60000,


        'num_sweeps_to_train_with': 10,

       
        'learning_rate': 0.2,

        'minibatch_size': 64,

        'num_samples_per_sweep': 60000,

        'training_progress_output_freq': 500
     
    }



class CnnForecaster(Forecaster):
    """
    This is a forecaster based on convolutional neural network.
    It analyses the last days.

    ----------
    model_class : The model type, which belonging to the forecaster.
    
    """
    
    # The related model
    model_class = CnnForecasterModel

    
    def create_model(self, x):
        """
        Create the model for time series prediction
        """
        model = C.layers.Sequential([
            Convolution(filter_shape=(5,5),num_filters=8,strides=(2,2),pad=True,activation=C.relu),
            Convolution(filter_shape=(5,5),num_filters=16,strides=(2,2),pad=True,activation=C.relu),
            Dense(shape=10)
            ])
        z = model(x)


    def __init__(self, model = None):
        """
        Constructor of this class which takes an optional model as input.
        If no model is given, it creates a default one.
        """
        super(CnnForecaster, self).__init__(model)


    def train_test(self, train_reader, test_reader, model_func, num_sweeps_to_train_with=10):
        params = self.model.params
            
        # Instantiate the model function; x is the input (feature) variable 
        # We will scale the input image pixels within 0-1 range by dividing all input value by 255.
        model = model_func(input/255)
    
        # Instantiate the loss and error function
        loss, label_error = create_criterion_function(model, label)

        # Instantiate the trainer object to drive the model training
        learning_rate = 0.2
        lr_schedule = learning_rate_schedule(learning_rate, C.UnitType.minibatch)
        mtm_schedule = momentum_schedule(0.9)
        learner = C.sgd(model_func.parameters, lr_schedule)
        trainer = C.Trainer(model_func, (loss, label_error), [learner])
    
        # Initialize the parameters for the trainer

        num_minibatches_to_train = (params['num_samples_per_sweep'] * params['num_sweeps_to_train_with']) / params['minibatch_size']
    
        # Map the data streams to the input and labels.
        input_map={
            label  : train_reader.streams.labels,
            input: train_reader.streams.features
        } 
    
     
        # Start a timer
        start = time.time()

        for i in range(0, int(params['num_minibatches_to_train'])):
            # Read a mini batch from the training data file
            data=train_reader.next_minibatch(minibatch_size, input_map=input_map) 
            trainer.train_minibatch(data)
            print_training_progress(trainer, i, params['training_progress_output_freq'], verbose=1)
     
        # Print training time
        print("Training took {:.1f} sec".format(time.time() - start))
    
        # Test the model
        test_input_map = {
            label : test_reader.streams.labels,
            input  : test_reader.streams.features
        }

        # Test data for trained model
        test_minibatch_size = 512
        num_samples = 10000
        num_minibatches_to_test = num_samples // test_minibatch_size

        test_result = 0.0   

        for i in range(num_minibatches_to_test):
    
            # We are loading test data in batches specified by test_minibatch_size
            # Each data point in the minibatch is a MNIST digit image of 784 dimensions 
            # with one pixel per dimension that we will encode / decode with the 
            # trained model.
            data = test_reader.next_minibatch(test_minibatch_size, input_map=test_input_map)
            eval_error = trainer.test_minibatch(data)
            test_result = test_result + eval_error

        # Average of evaluation errors of all test minibatches
        print("Average test error: {0:.2f}%".format(test_result*100 / num_minibatches_to_test))


    def train(self, meters, verbose = False):
        '''
        Does the training of the neural network. Each load chunk is translated into 
        multiple minibatches used for training the network.
        '''
        params = self.model.params
        
        num_minibatches_to_train = (params['num_samples_per_sweep'] * params['num_sweeps_to_train_with']) / params['minibatch_size']
        input = C.input_variable(params['input_dim'])
        label = C.input_variable(params['output_dim'])

        reader_train = create_reader(train_file, True, input_dim_reader, output_dim)
        reader_test = create_reader(test_file, False, input_dim_reader, output_dim)
        train_test(reader_train, reader_test, z)

        z.output.owner.b.asarray()

        input_map={
            label: reader_train.streams.labels,
            input: reader_train.streams.features
        } 
        aSingleImage = reader_train.next_minibatch(1, input_map=input_map)

        image = aSingleImage[input].asarray().reshape(1,28,28)
        plt.imshow(np.squeeze(image), cmap="gray_r")
        plt.axis('off')

        yy = C.combine([z.outputs[0].owner])
        yy = C.softmax(yy)

        image = [image.reshape(1,28,28)]
        prediction = np.squeeze(yy(image))

        model_pool = C.layers.Sequential([
            Convolution(filter_shape=(5,5),num_filters=8,strides=(1,1),pad=False,activation=C.relu,  name="conv1"),
            MaxPooling((2,2),(2,2), name="pool1", pad=True),
            Convolution(filter_shape=(5,5),num_filters=16,strides=(1,1),pad=False,activation=C.relu,  name="conv2"),
            MaxPooling((4,4),(4,4), name="pool2"),
            Dense(shape=10, name="dense1")
            ])
        z_pool = model_pool(input)

        # Analyse the new model
        z_pool.pool2.shape
        (1*5*5)*8+8 + (1*5*5)*16+16 + (4*4*16)*10+10
        C.logging.log_number_of_parameters(z_pool)

        num_params = (1*5*5)*8+8 + (1*5*5*8)*16+16 + (4*4*16*10)+10

        reader_train = create_reader(train_file, True, params['input_dim_reader'], params['output_dim'])
        reader_test = create_reader(test_file, False, params['input_dim_reader'], params['output_dim'])
        train_test(reader_train, reader_test, z_pool)




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
        pass


        
      