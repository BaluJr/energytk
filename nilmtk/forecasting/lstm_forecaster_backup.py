from matplotlib import pyplot as plt
import math
import numpy as np
import os
import pandas as pd
import random
import time
import cntk as C
from .forecaster import Forecaster

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve



class LstmForecasterModel():
    params = {
        # The target storage to store the data to
        'forecasting_storage': "C:\\Users\\maxim_000\\Documents\\InformatikStudium_ETH\\Masterarbeit\\6_Data\\Tmp\\forecasting.hdf",
        
        # Learning Rate
        'learning_rate': 0.005,

        # How many errors are taken together until training step
        'size_minibatches': 100,
        
        # How often each minibatch is trained
        'num_passes': 100,

        # Whether to use (100 for is_fast, else 2000)
        'epochs': 100,

        # How many inputs used per day  
        'timesteps': 14,
        # 10 Tage auf einmal
        'batch_size': 14 * 10,

        # Maximum value sothat we devide by this
        'normalize': 20000,

        # Dimensionality of LSTM cell
        'h_dims': 14,

        # The hidden dimensionality of the LSTM-NN
        'lstm_dim': 25

    }



class LstmForecaster(Forecaster):
    """This is a forecaster based on a distinct artificial neural network (ANN) 
    for each of the the forecasting distances.
    ----------
    model_class : The model type, which belonging to the forecaster.
    
    """
    
    # The related model
    model_class = LstmForecasterModel

    def do_a_testrun(self):
        params = self.model.params
        np.random.seed(0)
        X, Y = self.generate_solar_data("https://www.cntk.ai/jup/dat/solar.csv", 
                           params['timesteps'], normalize=params['normalize'])
        tst1 = X['train'][0:3]
        tst2 = Y['train'][0:3]
        self.train(X,Y)

    def generate_solar_data(self, input_url, time_steps, normalize=1, val_size=0.1, test_size=0.1):
        """
        generate sequences to feed to rnn based on data frame with solar panel data
        the csv has the format: time ,solar.current, solar.total
         (solar.current is the current output in Watt, solar.total is the total production
          for the day so far in Watt hours)
        """
        # try to find the data file local. If it doesn't exists download it.
        cache_path = os.path.join("data", "iot")
        cache_file = os.path.join(cache_path, "solar.csv")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        if not os.path.exists(cache_file):
            urlretrieve(input_url, cache_file)
            print("downloaded data successfully from ", input_url)
        else:
            print("using cache for ", input_url)    
        df = pd.read_csv(cache_file, index_col="time", parse_dates=['time'], dtype=np.float32)
        
        # normalize data
        df['solar.current'] /= normalize
        df['solar.total'] /= normalize
    
        # add the maximum values per day
        df["date"] = df.index.date
        grouped = df.groupby(df.index.date).max()
        grouped.columns = ["solar.current.max", "solar.total.max", "date"]
        df_merged = pd.merge(df, grouped, right_index=True, on="date")
        df_merged.drop('date', axis = 1, inplace=True)

        # we group by day so we can process a day at a time.
        grouped = df_merged.groupby(df_merged.index.date)
        per_day = []
        for _, group in grouped:
            per_day.append(group)

        # split the dataset into train, validatation and test sets on day boundaries
        val_size = int(len(per_day) * val_size)     # How much used for validation
        test_size = int(len(per_day) * test_size)   # How much used for testing
        next_val = 0
        next_test = 0
        result_x = {"train": [], "val": [], "test": []}
        result_y = {"train": [], "val": [], "test": []}    
        for i, day in enumerate(per_day):
            total = day["solar.total"].values
            if len(total) < 8:
                # Skip with less than 8
                continue
            
            if i >= next_val:
                current_set = "val"
                next_val = i + int(len(per_day) / val_size)
            elif i >= next_test:
                current_set = "test"
                next_test = i + int(len(per_day) / test_size)
            else:
                current_set = "train"

            max_total_for_day = np.array(day["solar.total.max"].values[0])
            
            if len(total) >= 14:
                result_x[current_set].append(total[0:14])
                result_y[current_set].append([max_total_for_day]) 
            else:
                print("Nope")
            #for j in range(2, len(total)):
            #    result_x[current_set].append(total[0:j])            # Per day all profils up to length 14
            #    result_y[current_set].append([max_total_for_day])   # Always the max total
            #    # Maximum use 14 timesteps
            #    if j >= time_steps:
            #        break

        # make result_y a numpy array
        for ds in ["train", "val", "test"]:
            result_y[ds] = np.array(result_y[ds])
        return result_x, result_y

    def next_batch(x, y, ds):
        """
        Get the next batch for training
        """
        def as_batch(data, start, count):
            return data[start:start + count]
        BATCH_SIZE = self.model.params['batch_size']
        for i in range(0, len(x[ds]), BATCH_SIZE):
            yield as_batch(X[ds], i, BATCH_SIZE), as_batch(Y[ds], i, BATCH_SIZE)



    def __init__(self, model = None):
        """
        Constructor of this class which takes an optional model as input.
        If no model is given, it creates a default one.
        """
        super(LstmForecaster, self).__init__(model)


    def train(self, meters, verbose = False):
        '''
        Does the training of the neural network. Each load chunk is translated into 
        multiple minibatches used for training the network.
        '''
        # C.try_set_default_device(C.cpu())
        X = meters
        Y = verbose
        params = self.model.params

        # Hier muss ich spaeter mal mehrere Features rein geben pro step 
        x = C.sequence.input_variable(1)                                # Das fuegt glaube sofort schon eine Dyn Achse ein
        # create the model
        z = self.create_model(x)
        # expected output (label), also the dynamic axes of the model output is specified as the model of the label input 
        l = C.input_variable(1, dynamic_axes=z.dynamic_axes, name="y")  # Das ist die gleiche dyn Achse die aus x in z rein kam

        # Loss and error function 
        loss = C.squared_error(z, l)
        error = C.squared_error(z, l)

        # Use adam optimizer (Das leicht verwirrende: Eingabe, Ausgabe u. Trainer werden separat rumgereicht)
        lr_schedule = C.learning_rate_schedule(params['learning_rate'], C.UnitType.minibatch)
        momentum_time_constant = C.momentum_as_time_constant_schedule(params['batch_size'] / -math.log(0.9)) 
        learner = C.fsadagrad(z.parameters, lr = lr_schedule, momentum = momentum_time_constant)
        trainer = C.Trainer(z, (loss, error), [learner])

        # Training
        loss_summary = []
        start = time.time()
        t1 = time.time()
        # iterate whole dataset epoch times
        for epoch in range(0, params['epochs']):
            # iterate the minibatches
            for x_batch, l_batch in self.next_batch(X, Y, "train"):
                # one time training per minibatch
                trainer.train_minibatch({x: x_batch, l: l_batch})        
            if epoch % (params['epochs'] / 10) == 0:
                training_loss = trainer.previous_minibatch_loss_average
                loss_summary.append(training_loss)
                print("epoch: {}, loss: {:.4f}".format(epoch, training_loss))
                print(time.time()-t1)
        print("Training took {:.1f} sec".format(time.time() - start))

        # Plot the output
        #plt.plot(loss_summary, label='training loss');
        #plt.show()

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

       # predict
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


        
      

    def create_model(self, x):
        """
        Create the model for time series prediction
        """
        with C.layers.default_options(initial_state = 0.1):
            m = C.layers.Recurrence(C.layers.LSTM(self.model.params['h_dims']))(x)
            m = C.sequence.last(m)
            m = C.layers.Dropout(0.2)(m)
            m = C.layers.Dense(1)(m)
            return m

    def next_batch(self, x, y, ds):
        """get the next batch to process"""

        def as_batch(data, start, count):
            part = []
            for i in range(start, start + count):
                part.append(data[i])
            return part
        
        BATCH_SIZE = self.model.params['batch_size']
        for i in range(0, len(x[ds])-BATCH_SIZE, BATCH_SIZE):
            yield as_batch(x[ds], i, BATCH_SIZE), as_batch(y[ds], i, BATCH_SIZE)

    # validate
    def get_mse(self, X,Y,labeltxt, trainer, x, l):
        result = 0.0
        for x1, y1 in self.next_batch(X, Y, labeltxt):
            eval_error = trainer.test_minibatch({x : x1, l : y1})
            result += eval_error
        return result/len(X[labeltxt])
        

