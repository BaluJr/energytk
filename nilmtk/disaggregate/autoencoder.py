# This module tries to achieve disaggregation by using an autoencoder to 
# find the main influences in the powerflow.
# Based on CNTK
#
# Realization on the basis of the 
# - Autoencoder example: For the model
# - The financial example: How to load from pandas 
# - The CNN example: Because the financial example creates redundant data. I only have time series.

# General packages
from __future__ import print_function, division
from collections import OrderedDict, deque
import numpy as np
import pickle
import os
import sys
from itertools import product

# Packages for data handling and machine learning 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn_pandas import DataFrameMapper
import sklearn.preprocessing, sklearn.decomposition, sklearn.linear_model, sklearn.pipeline, sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
import sklearn.metrics

# Packages from nilmtk
from nilmtk.disaggregate import Disaggregator
from nilmtk.feature_detectors.cluster import hart85_means_shift_cluster
from nilmtk.feature_detectors.steady_states import find_steady_states_transients

# Packages from Cntk
import cntk as C

# Fix the seed for repeatability of experiments
SEED = 42
np.random.seed(SEED)




class Autoencoder(Disaggregator):
    """ 
    Super fancy unsupervised neural network approach.

    Attributes
    ----------
    window_width: int 
        The datapoints to encode together
    encoding_dim : int
        The dimensionality of the encoded signal
    model : dict
        Mal noch nichts. 
    chunksize: int
        The chunksize which is not only used for loading but 
        which is also the siz of the series sent to the Cnn
        in one step.
    """

    def __init__(self):
        self.window_width = 5000
        self.encoding_dim = 32
        self.model = None
        self.chunksize = 5000


    #region The Cntk Functions
    def _create_model_naive(self, features):
        ''' This function creates 
        Parameters
        ----------
        features : A single window of self.window_width
        '''
        with C.layers.default_options(init = C.glorot_uniform()):
            # We scale the input pixels to 0-1 range
            encode = C.layers.Dense(self.encoding_dim, activation = C.relu)(features)
            decode = C.layers.Dense(self.window_width, activation = C.sigmoid)(encode)
        return decode
    

    def _create_model_cnn(self, features):
        '''
        This is the more intelligent approach for learning the windows. 
        It avoids the generation of additional data and optimizes training
        
        Parameters
        ----------
        features : Immediatly the whole timeseries.
        '''
        with C.layers.default_options(init = C.glorot_uniform()):
            # We scale the input pixels to 0-1 range
            encode = C.layers.Convolution1D((self.window_width,), self.encoding_dim, reduction_rank =0, activation = C.relu, name="encoding")(features)
            decode = C.layers.Convolution1D(1, self.window_width, activation = C.sigmoid, name="decoding")(encode)
        return decode

    #endregion


    
    def train(self, metergroup, cols=[('power', 'active')], trainingsmode = 'cnn',  **kwargs):
        """
        Training the autoencoder to detect the best sparse representation for the 
        powerflow. This sparse representation is then afterwards used to cluster.
        The hope is, that the the neural network is more flexible in defining 
        interesting signatures.

        Parameters
        ----------
        metergroup : a nilmtk.MeterGroup object
        cols: nilmtk.Measurement, should be one of the following
            [('power','active')]
            [('power','apparent')]
            [('power','reactive')]
            [('power','active'), ('power', 'reactive')]
        trainingsmode: 'cnn' or 'naive'
        """
            
        if self.model:
            raise RuntimeError(
                "This implementation of Combinatorial Optimisation"
                " does not support multiple calls to `train`.")

        # Create the cntk model dependent of the given params
        if trainingsmode == 'cnn':
            input = C.input_variable(self.chunksize)
            model = self._create_model_cnn(input)
        else:
            input = C.input_variable(self.window_width)
            model = self._create_model_naive(input)

        # Train 
        for i, meter in enumerate(metergroup.meters):
            print("Training model for meter '{}'".format(meter))
            power_series = meter.load(chunksize=self.chunksize, **kwargs)
            chunk = next(power_series)  
            self.train_on_chunk(chunk, meter)

            try:
                next(power_series)
            except StopIteration:
                pass

        print("Done training!")

    
    def train_on_chunk(self, chunk, trainingsmode):
        """Signature is fine for site meter dataframes (unsupervised
        learning). Would need to be called for each appliance meter
        along with appliance identifier for supervised learning.
        Required to be overridden to provide out-of-core
        disaggregation.

        Parameters
        ----------
        chunk : pd.DataFrame where each column represents a
            disaggregated appliance
        meter : ElecMeter for this chunk
        """
        #chunk["diff"] = chunk.diff(1).fillna(0)
        
        line = np.asarray(chunk[("power","active")], dtype = "float32")
        C.Trainer(

    def disaggregate(self, mains, output_datastore, **load_kwargs):
        """
        Disaggregate mains according to the model learnt previously.

        Parameters
        ----------
        mains : nilmtk.ElecMeter or nilmtk.MeterGroup => In facts the 3 phases
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm. => Wird in einem 2. Step benutzt 
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        """
        pass
    

    def disaggregate_chunk(self, chunk, prev, transients):
        """
        Parameters
        ----------
        chunk : pd.DataFrame
            mains power
        prev
        transients : returned by find_steady_state_transients

        Returns
        -------
        states : pd.DataFrame
            with same index as `chunk`.
        """
        raise NotImplementedError("Muss ich ggf noch mal von den anderen Klassen herkopieren.")


    def export_model(self, filename):
        raise NotImplementedError("Muss ich ggf noch mal von den anderen Klassen herkopieren.")


    def import_model(self, filename):
        raise NotImplementedError("Muss ich ggf noch mal von den anderen Klassen herkopieren.")

    #endregion
