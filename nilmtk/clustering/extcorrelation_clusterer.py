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
from nilmtk import DataSet, ExternDataSet, ElecMeter, MeterGroup
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle as pckl

class ExtCorrelationClustererModel(object):
    ''' The model belonging to the external correlation clusterer

    Attributes
    ----------
    correlations: pd.DataFrame
        This is the resulting dataframe with the relations between all meters to all external features
    centroids: [np.ndarray]
        The centroids after doing the clustering
    assignments: [int,...]
        The assignments after doing the clustering
    '''

    params = {
        # The external, which is used to create the vectors for each element, DataSet
        'ext_data_dataset': "C:\\Users\\maxim_000\\Documents\\InformatikStudium_ETH\\Masterarbeit\\6_Data\\Tmp\\ExtData.hdf",
        
        # The features which are used during the forecasting
        # ('cloudcover', ''), ('humidity', 'relative'), ('temperature', ''), ('humidity', 'absolute'), ('dewpoint', ''), ('windspeed', ''), ('precip', '')
        # ('national', ''), ('school', '')
        'externalFeatures': [('temperature', ''), ('dewpoint', ''), ('precip', ''), ('national', ''), ('school', '')], #, 'time'
        

        # How the daytime is regarded
        'hourFeatures': ['00-06', "06-09", "09-12", "12-15", "15-18", "18-21", "21-24"],

        # How the weekdays are paired
        'weekdayFeatures': ["0-5","5-6","6-7"],

        # Momentan wird keine Autokorrelation benutzt
        'shifts': [], #list(range(1,8)) + [cur*96 for cur in range(7)] + [cur*96*7 for cur in range(4)],

        # Self correlation resolution (The resolution in which the self resolution is checked)
        'self_corr_freq': 60*15,
        
        # The type of correlation used to build up the vectors
        'method': 'pearson', #'pearson', 'kendall', 'spearman'

        # The sample period for the load profiles
        'sample_period': 300,

        # The section in which to perform the correlation, Of type timeframe
        'section': None,

        # How the meters are grouped and the correlation calculated
        'grouping': 'single'   #'single', 'all', 'grouped'
    }
        
    config = {
        'size_input_queue': 0
    }

    def __init__(self, args = {}):
        '''
        The paramters are set to the optional paramters handed in by the constructor.
        
        Parameters
        ----------
        args: dic
            Parameters which are optionally set
        '''
        for arg in args:
            self.params[arg] = args[arg]



class ExtCorrelationClusterer(Clusterer):
    """
    This clusterer groups the meters by their correlation towards external data.
    The external data which is used is given in the model as parameter.
    After training, this model contains a vector for each meter.
    Take care, that every element might have separate data to be benchmarked against.

    Everything is done inmemory to make it fit into the memory.
    """
    model_class = ExtCorrelationClustererModel

    def __init__(self, model = None):
        """
        Constructor of this class which takes an optional model as input.
        If no model is given, it creates a default one.

        Paramters
        ---------
        model: Model of type model_class
            The model which shall be used.
        """
        super(ExtCorrelationClusterer, self).__init__(model)


    def set_up_corrdataframe(self, dim, weekdays, dayhours):
        '''
        This is the simple function which is executed in parallel to 
        accellerate the process.
        Each process only gets the location of its target timeline as 
        an input. It then only takes the data from the global element.
        '''
        # Fastest 0.34 for 20
        def tst3(current_loadseries, days_of_group):
            days = current_loadseries.index.weekday.values
            result = np.ones(len(current_loadseries)).astype(bool)
            for day in days_of_group:
                result |= (days == day)
            return pd.Series(result)
        # bit slower 0.49 for 20
        def tst(current_loadseries, days_of_group):
            t = np.in1d(current_loadseries[:-1].index.weekday.values, days_of_group)
        # Slow 1.47 for 20
        def tst2(current_loadseries, days_of_group):
            t = current_loadseries[:-1].index.weekday.apply(lambda e, dog=days_of_group: e in dog)
            
        # Add the time related features
        global corrdataframe
        corrdataframe = group_data[dim]
        days = current_loadseries.index.weekday.values
        for weekday in weekdays:
            idx = ('weekday', weekday)
            days_of_group = set(range(int(weekday[0]),int(weekday[2])))
            corrdataframe[idx] = False
            for day in days_of_group:
                corrdataframe[idx] |= (days == day)
        hours = current_loadseries.index.hour.values
        for dayhour in dayhours:
            idx = ('hour', dayhour)
            hours_of_group = set(range(int(dayhour[:2]),int(dayhour[3:])))
            corrdataframe[idx] = False
            for hour in hours_of_group:
                corrdataframe[idx] |= (hours == hour)

        
    def correlate(self):
        '''
        This is the simple function which is executed in parallel to 
        accellerate the process.
        Each process only gets the location of its target timeline as 
        an input. It then only takes the data from the global element.

        Returns
        -------
        correlations: pd.DataFrame
            The dataframe of the model.
        '''

        correlations = corrdataframe.apply(lambda col, method = self.model.params['method']: col.corr(current_loadseries, method=method), axis=0)
        return correlations.fillna(0)




    def calculate_correlations(self, meters, extDataSet, n_jobs = -1, tmp_folder = None, verbose = False):
        ''' Function only setting up the correlations without clustering.
        Sometimes it is also required to just get the correlations.

        Parameters
        ----------
        meters: nilmtk.MeterGroup
            The meters to cluster, from which the demand is loaded.
        extDataSet: nilmtk.DataSet
            The External Dataset containing the fitting data.
        n_jobs: int
            ! Not used at the moment !
            Defines the amount of processes. (-1 = amount of cpu cores)
        verbose: bool
            Whether to print additional output

        Returns
        -------
        correlations: pd.DataFrame
            DataFrame with a column per external feature 
            and a row for each meter.
        '''
        #try:
        #    return pckl.load(open(tmp_folder + "_" + self.model.params['grouping'],'rb'))
        #except:
            # We need global variables if we want to use multiprocessing
        global current_loadseries
        global group_data
        global corrdataframe
        clusterer_timeseries = []
        
        # Declare amount of processes
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        # Prepare result frame
        dims = self.model.params['externalFeatures'] + self.model.params['shifts']
        weekdayFeatures = [('weekday', cur) for cur in self.model.params['weekdayFeatures']]
        hourFeatures = [('hour', cur) for cur in self.model.params['hourFeatures']]
        corrs = self.model.correlations = pd.DataFrame(columns=dims +  weekdayFeatures + hourFeatures)

        # Load the external data specified in the params
        periodsExtData = [int(dev['sample_period']) for dev in extDataSet.metadata['meter_devices'].values()]
        min_sampling_period = min(periodsExtData + [self.model.params['self_corr_freq']]) * 2

        # Group the meters by the designated strategy
        try:
            if self.model.params['grouping'] == 'single':
                metergroups = meters.groupby('zip', use_appliance_metadata = False)
                zips = list(metergroups.keys())
                metergroups = metergroups.values()              
            elif self.model.params['grouping'] == 'all':
                metergroups =  [MeterGroup([meters])]
                zips = [meters.meters[0].building_metadata['zip']]        
            elif self.model.params['grouping'] == 'cluster':
                metergroups =  [meters]
                zips = [meters.meters[0].meters[0].building_metadata['zip']]       

        except:
            zips = [meters.meters[0][1].building_metadata['zip']] * len(meters)

        processed_meters = 0
        for i, group in enumerate(metergroups):
            zip = zips[i]                        
            
            # Load the groupspecific data (weather)
            group_data = extDataSet.get_data_for_group(zip, self.model.params['section'], 300, self.model.params['externalFeatures']) # min_sampling_period
                
            # Then go through all meters
            for processed_meters, meter in enumerate(group.meters):

                # meters load (Und wieder auf kontinuierliche Zeitreihe bringen)
                current_loadseries = meter.power_series_all_data(dtype='float16', sample_period = self.model.params['sample_period'], sections = self.model.params['section'], tmp_folder = tmp_folder, verbose=verbose)#, load_kwargs={'sample_period':min_sampling_period})
                    
                if processed_meters == 0:
                    self.set_up_corrdataframe(dims,self.model.params['weekdayFeatures'], self.model.params['hourFeatures'])
                
                # Multiprocessing currently deactivated
                #if processed_meters != 0:
                #    newSeries = pd.Series(index = current_loadseries.asfreq('4s').index)
                #    resampledload = current_loadseries.combine_first(newSeries)
                #    resampledload = resampledload.interpolate()
                #    current_loadseries = resampledload.resample('2min', how='mean')               
                #pool = multiprocessing.Pool(processes=n_jobs)
                #corr_vector = pool.map(correlate, dims)

                corr_vector = []
                corr_vector = self.correlate()
                corrs.loc[meter.identifier,:] = corr_vector

                if verbose:
                    print('Correlation set up for {0} - {1}/{2}'.format(meter,processed_meters,len(group.meters)))
            
        #pckl.dump(corrs, open(tmp_folder + "_" + self.model.params['grouping'],'wb'))
        return corrs

        
    

    def cluster(self, meters, extDataSet, target_file, return_correlations = False, tmp_folder = None,  n_jobs = -1, verbose = False):
        ''' The main clustering function.

        Do it in a parallelized way to accellerate it. 
        Take care that only households bmelow a maximum consumption of 32kw are 
        supported as we use 16bit integers.
        To accellerate the process the calculation is performed in the following 
        way: One thread is loading. One is doing the correlation. One is storing.
        
        Parameters
        ----------
        meters: nilmtk.MeterGroup
            The meters to cluster, from which the demand is loaded.
        extDataSet: nilmtk.DataSet
            The External Dataset containing the fitting data.
        target_file: string
            Path to the file where the cluster results shall be stored.
        return_correlations: pd.DataFrame
            Also returns the correlations of the appliances towards the 
            external features.
        n_jobs: int
            ! Not used at the moment !
            Defines the amount of processes. (-1 = amount of cpu cores)
        verbose: bool
            Whether to print additional output

        Returns
        -------
        clusterings: pd.DataFrame
            A column for each cluster which only has one field, the list 
            of all meters.
        correlations: pd.DataFrame
            Only when return_correlations==True
            DataFrame with a column per external feature 
            and a row for each meter.
        '''
        
        # First calculate the correlations
        corrs = self.calculate_correlations(meters, extDataSet, tmp_folder = tmp_folder, verbose = verbose)

        # Do the clustering after the created vectors
        centroids, assignments  = self._cluster_vectors(corrs, 5)
        corrs['cluster'] = assignments
        self.model.assignments = assignments  
        self.model.centroids = centroids

        # Return the clustering result as groups of metering ids (than one can easily select)
        to_list = lambda x: list(x)
        result = corrs.reset_index().groupby('cluster').agg({'index':to_list})
        result.rename(columns={'index': 'elecmeters'})
        result.to_csv(target_file) # return result['index'] 
        
        if return_correlations:
            return result, corrs
        else:
            return result


        
    def _cluster_vectors(self, correlation_dataframe, max_num_clusters=3, method='kmeans'):
        ''' Applies clustering on the previously extracted vectors. 

        Parameters
        ----------
        correlation_dataframe: pd.DataFrame 
            The dataframe to cluster
        max_num_clusters : int
            Amount of clusters for which k-means is performed and of 
            which the best (BIC) is taken.
        method: string ("kmeans" and "ward")
            Used approach to do the forecasting.
        
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
        scaler = StandardScaler()
        clusteringInput = scaler.fit_transform(clusteringInput)

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
                        clusteringInput, k_means_labels[n_clusters], metric='euclidean')
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
        centroids = scaler.inverse_transform(k_means_cluster_centers[num_clusters])
        return centroids, k_means_labels[num_clusters]

    

    def _apply_clustering_n_clusters(self, X, n_clusters, method='kmeans'):
        ''' Do the clustering.
        Currently only kmeans is supported. Can be easily replaced.
        
        Parameters
        ----------
        X: ndarray
            Array to cluster
        n_clusters: int
            Exact number of clusters to use
        method: string 
            Currenty only kmeans is supported

        Returns
        -------
        centroids : n_cluster * (ndarray of int32s)
            The correlation values which belong together    
        labels: n_cluster * (ndarray of int32s)
            The assignment of each vector to the clusters
        '''
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