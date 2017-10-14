# General packages
import multiprocessing

# Packages for data handling and machine learning 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn_pandas import DataFrameMapper
import sklearn.preprocessing, sklearn.decomposition, sklearn.linear_model, sklearn.pipeline, sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
import sklearn.metrics
from .clusterer import Clusterer
from nilmtk import DataSet, ExternDataSet


import pickle as pckl

class ExtCorrelationClustererModel(object):
    params = {
        # The external, which is used to create the vectors for each element, DataSet
        'ext_data_dataset': "C:\\Users\\maxim_000\\Documents\\InformatikStudium_ETH\\Masterarbeit\\6_Data\\Tmp\\ExtData.hdf",
        
        # The features which are used during the forecasting
        # ('cloudcover', ''), ('humidity', 'relative'), ('temperature', ''), ('humidity', 'absolute'), ('dewpoint', ''), ('windspeed', ''), ('precip', '')
        # ('national', ''), ('school', '')
        'externalFeatures': [('temperature', ''), ('dewpoint', ''), ('precip', ''), ('national', ''), ('school', '')], #, 'time'
        
        # How the daytime is regarded
        'hourFeatures': ['H00-06', "H06-09", "H09-12", "H12-15", "H15-18", "H18-21", "H21-24"],

        # How the weekdays are paired
        'weekdayFeatures': ["W1-5","W6","W7"],

        # Momentan wird keine Autokorrelation benutzt
        'shifts': [], #list(range(1,5)) + [cur*96 for cur in range(7)] + [cur*96*7 for cur in range(4)],

        # Self correlation resolution (The resolution in which the self resolution is checked)
        'self_corr_freq': 60*15,
        
        # The type of correlation used to build up the vectors
        'method': 'pearson' #'pearson', 'kendall', 'spearman'
    }
        
    config = {
        'size_input_queue': 0,
        'verbose': True
    }

    # This is the resulting dataframe with the relations between all meters to all external features
    correlations = None

    # The centroids after doing the clustering
    centroids = None
    
    # The assignments after doing the clustering
    assignments = None



class ExtCorrelationClusterer(Clusterer):
    """
    This clusterer groups the meters by their correlation towards external data.
    The external data which is used is given in the model as parameter.
    After training, this model contains a vector for each meter.
    Take care, that every element might have separate data to be benchmarked against.

    Ich gehe von niedrigerer Aufloesung aus und dass es in den Speicher passt.
    """
    model_class = ExtCorrelationClustererModel

    def __init__(self, model = None):
        """
        Constructor of this class which takes an optional model as input.
        If no model is given, it createsa default one.
        """
        super(ExtCorrelationClusterer, self).__init__(model)

        
    def correlate(self,input, weekdays, hours):
        '''
        This is the simple function which is executed in parallel to 
        accellerate the process.
        Each process only gets the location of its target timeline as 
        an input. It then only takes the data from the global element.
        '''
        if isinstance(input, int):
            shift = input
            return current_loadseries.autocorr(shift)
        else:
            # Add the external data
            dim = input
            corrdataframe = group_data[dim]
            
            # Add the time related features
            for weekday in weekdays:
                days_of_group = range(int(weekday[1]),int(weekday[3]))
                corrdataframe[weekday] = current_loadseries.index.weekday.apply(lambda e: e in days_of_group)
            for hour in hours:
                days_of_group = range(int(hour[1:3]),int(weekday[5:6]))
                corrdataframe[weekday] = current_loadseries.index.weekday.apply(lambda e: e in days_of_group)

            # Calculate the correlations
            correlations = corrdataframe.apply(lambda col: col.corr(current_loadseries.iloc[:,0], method=self.model['method']), axis=0)
            return correlations

        


    def cluster(self, meters, targetFile, n_jobs = -1):
        '''
        Do it in a parallelized way to accellerate it. 
        Take care that only households below a maximum consumption of 32kw are 
        supported as we use 16bit integers.
        To accellerate the process the calculation is performed in the following 
        way: One thread is loading. One is doing the correlation. One is storing.
        
        Parameters:
        meters: The meters which shall be clustered
        '''

        # We need global variables sothat they can be accessed in the subprocesses
        global current_loadseries
        global group_data
        clusterer_timeseries = []
        
        # Declare amount of processes
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        # Prepare result frame
        dims = self.model.params['externalFeatures'] + self.model.params['shifts']
        corrs = self.model.correlations = pd.DataFrame(columns=dims + self.model.params['weekdayFeatures'] + self.model.params['hourFeatures'])

        # Load the external data specified in the params
        extDataSet = ExternDataSet(self.model.params['ext_data_dataset'])
        periodsExtData = [int(dev['sample_period']) for dev in extDataSet.metadata['meter_devices'].values()]
        min_sampling_period = min(periodsExtData + [self.model.params['self_corr_freq']]) * 2

        # Group the meters by there zip
        try:
            corrs = pckl.load(open("coors.pckl",'rb'))
        except:
            metergroups = meters.groupby('zip', use_appliance_metadata = False)
            processed_meters = 0
            for zip in metergroups:
                group = metergroups[zip]
                # Load the groupspecific data (weather)
                group_data = extDataSet.get_data_for_group(zip, group.get_timeframe(), min_sampling_period, self.model.params['externalFeatures'])
                # Then go through all meters
                for processed_meters, meter in enumerate(group.meters):

                    # meters load (Und wieder auf kontinuierliche Zeitreihe bringen)
                    current_loadseries = meter.power_series_all_data(dtype='float16')#, load_kwargs={'sample_period':min_sampling_period})
                    current_loadseries = current_loadseries.resample('2s', how='mean').interpolate().resample('2min', how='mean')
                    
                    #current_loadseries = current_loadseries.tz_localize('UTC')
               
                    #pool = multiprocessing.Pool(processes=n_jobs)
                    #corr_vector = pool.map(correlate, dims)

                    corr_vector = []
                    corr_vector.append(self.correlate(difs + self.model.params['weekdayFeatures'] + self.model.params['hourFeatures']))
                    corrs.loc[meter.identifier,:] = corr_vector

                    if self.model.config['verbose']:
                        print('Correlation set up for {0} - {1}/{2}'.format(meter,processed_meters,len(group.meters)))
            pckl.dump(corrs, open("coors.pckl",'wb'))

        # Do the clustering after the created vectors
        centroids, assignments  = self._cluster_vectors(corrs)
        corrs['cluster'] = assignments
        self.model.assignments = assignments  
        self.model.centroids = centroids

        # Return the clustering result as groups of metering ids (than one can easily select)
        to_list = lambda x: list(x)
        tst = corrs.reset_index().groupby('cluster').agg({'index':to_list})
        tst.to_csv(targetFile) # return tst['index'] 


        
    def _cluster_vectors(self, correlation_dataframe, max_num_clusters=3, method='kmeans'):
        ''' Applies clustering on the previously extracted vectors. 

        Parameters
        ----------
        events : pd.DataFrame with the dimensions as columns
        max_num_clusters : int
        method: string Possible approaches are "kmeans" and "ward"
        Returns
        -------
        centroids : ndarray of int32s
            The correlation values which belong together
            
        labels: ndarray of int32s
            The assignment of each vector to the clusters
        '''

        correlation_dataframe = correlation_dataframe.fillna(0)
           
        # Preprocess dataframe
        mappings = [(dim, None) for dim in correlation_dataframe.columns]


        mapper = DataFrameMapper(mappings) 
        clusteringInput = mapper.fit_transform(correlation_dataframe.copy())
    
        # Do the clustering
        num_clusters = -1
        silhouette = -1
        k_means_labels = {}
        k_means_cluster_centers = {}
        k_means_labels_unique = {}

        # Special case:
        if len(correlation_dataframe) == 1: 
            return np.array([events.iloc[0,:]]), np.array([0])

        # If exact cluster number not specified, use cluster validity measures to find optimal number
        for n_clusters in range(2, max_num_clusters):
            try:
                # Do a clustering for each amount of clusters
                labels, centers = self._apply_clustering_n_clusters(clusteringInput, n_clusters, method)
                k_means_labels[n_clusters] = labels
                k_means_cluster_centers[n_clusters] = centers
                k_means_labels_unique[n_clusters] = np.unique(labels)

                # Then score each of it and take the best one
                try:
                    sh_n = sklearn.metrics.silhouette_score(
                        events, k_means_labels[n_clusters], metric='euclidean')
                    if sh_n > silhouette:
                        silhouette = sh_n
                        num_clusters = n_clusters
                except Exception as instance:
                    num_clusters = n_clusters

            except Exception as e:
                if num_clusters > -1:
                    return k_means_cluster_centers[num_clusters]
                else:
                    return np.array([0])

        return k_means_cluster_centers[num_clusters].flatten(), k_means_labels[num_clusters]

    

    def _apply_clustering_n_clusters(self, X, n_clusters, method='kmeans'):
        """
        :param X: ndarray
        :param n_clusters: exact number of clusters to use
        :param method: string kmeans or ward
        :return:
        """
        if method == 'kmeans':
            k_means = KMeans(init='k-means++', n_clusters=n_clusters)
            k_means.fit(X)
            return k_means.labels_, k_means.cluster_centers_








        ## Do the calculations in parallel (one extdata after the other)
        #tst = pd.DataFrame(np.random.random([10000000,3]))
        #meters.load(load_kwargs={type:'float16'})

        #l = multiprocessing.Lock()
        #counter = 0
        #def init(l, c):
        #    global lock
        #    global counter
        #    lock = l
        #    counter = c

        #pool = multiprocessing.Pool(initializer=init, initargs=(l,c), processes=n_jobs)
        #pool.map(processFile, meters.all_meters())