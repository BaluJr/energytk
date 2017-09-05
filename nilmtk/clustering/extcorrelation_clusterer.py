# General packages
import multiprocessing

# Packages for data handling and machine learning 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn_pandas import DataFrameMapper
import sklearn.preprocessing, sklearn.decomposition, sklearn.linear_model, sklearn.pipeline, sklearn.metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
import sklearn.metrics



class ExtCorrelationClustererModel(object):
    params = {
        # The external, which is used to create the vectors for each element, DataSet
        'ext_data_dataset': "E:/ExternalData/ExtData.hdf",
        
        # The features which are used during the forecasting
        'externalFeatures': ['temperature', 'dewpoint', 'national', 'school'], #, 'time'
        'shifts': list(range(1,5)),#48)), # +[cur*96 for cur in range(7)] + [cur*96*7 for cur in range(4)],

        # Self correlation resolution (The resolution in which the self resolution is checked)
        'self_corr_freq': 60,
    },
        
    config = {
        'size_input_queue': 0,
        'verbose': True
    },

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
    """
    model_class = ArimaForecasterModel

    def __init__(self, model = None):
        """
        Constructor of this class which takes an optional model as input.
        If no model is given, it createsa default one.
        """
        super(ExtCorrelationClusterer, self).__init__(model)



    def cluster(self, meters, n_jobs = -1):
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
        global ext_data
        global current_loadseries
        global group_data
        clusterer_timeseries = []
        
        # Declare amount of processes
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        # Prepare result container
        dims = self.model.params['externalFeatures'] + self.model.params['shifts']
        self.model.correlations = pd.DataFrame(columns=dims)
        corrs = self.model.correlations

        # Load the external data specified in the params (low res -> all in memory)
        extDataSet = DataSet(self.model.params['ext_data_dataset'])
        clusterer_timeseries['const']: extDataSet.load('', dtype='float32')
        highest_freq = max([dev.metadata.sample_rate for dev in extData.devices()] + self.model.params['self_corr_freq'])

        # Group the meters by there zip
        metercount = len(meters)
        metergroups = meters.groupby('zip')
        processed_meters = 0
        for group in metergroups:

            # Load the groupspecific data (weather)
            group_data = extData.getFor(group, self.model.params['externalFeatures']) #Irgendwie laden

            # Then go through all meters
            for meter in group:
                processed_meters += 1

                # meters load
                current_loadseries = meter.load(dtype='float16', load_kwargs={'freq':highest_freq})
               
                pool = multiprocessing.Pool(processes=n_jobs)
                corr_vector = pool.map(corr, dims)
                corrs.loc[meter.identifier(),:] = corr_vector

                if self.model.config['verbose']:
                    print('Correlation set up for %s - %i/%i'.format(meter,processed_meters,metercount))

        # Do the clustering after the created vectors
        centroids, assignments  = self.model._cluster_vectors()
        corrs.loc['cluster'] = assignments
        self.model.assignments = assignments  
        self.model.centroids = centroids


        # Return the clustering result as groups of metering ids (than one can easily select)
        to_list = lambda x: list(x)
        tst = dataToRequest.groupby('cluster').agg({'index':to_list})
        return tst # Hier sollten Listen der Identifier entstanden sein





    def correlate(self,input):
        '''
        This is the simple function which is executed in parallel to 
        accellerate the process.
        Each process only gets the location of its target timeline as 
        an input. It then only takes the data from the global element.
        '''
        if 'shift' in input:
            freq = tst
        return current_loadseries.autocorr(shift)


        
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
              
        # Preprocess dataframe
        mapper = DataFrameMapper([('active transition', None)]) 
        clusteringInput = mapper.fit_transform(events.copy())
    
        # Do the clustering
        num_clusters = -1
        silhouette = -1
        k_means_labels = {}
        k_means_cluster_centers = {}
        k_means_labels_unique = {}

        # Special case:
        if len(events) == 1: 
            return np.array([events.iloc[0]["active transition"]]), np.array([0])

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
                except Exception as inst:
                    num_clusters = n_clusters

            except Exception:
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




    def forecast(self):
        





def RandomTimeSeries(start, freq, end=None, periods=-1):
    if periods != -1:
        series = pd.date_range(start=start, periods=period, freq=freq)
    else:
        series = pd.date_range(start=start, end=end, freq=freq)
    return pd.Series(np.random.randn(len(series)), index=series)





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