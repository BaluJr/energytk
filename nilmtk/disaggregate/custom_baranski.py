#To Do Baranski:
# -Build the Events 
# -- Muss ich noch ausbauen, dass nicht nur die States sondern auch die Transitions mit zusatzinfos gespeichert werden (Laenge, Step)
# -- Dafuer baue ich eine eigene Unterfunktion
# -Print the Events im 3d Raum 
# -Clustering 
# --Hier nehme ich KMeans, muss eben aus der Dokumentation raus lesen wie hoeherdimensionale Daten ausgelesen werden
# --Auch GMM mal probieren. -> Das austesten wie viele Schritte es gibt wird interessant 
#   Ggf kann man dann noch eine Restgruppe festlegen und das muessen dann die ueberlagerten sein.
# --Ggf ist Ward Clustering eine wesentlich bessere Idee (Insbesondere auch Moeglichkeit Connectivity Matrix zu setzen)
#
# -Print Events mit Clustering

# -Die FHMMs bilden 
# -- abschreiben aus Matlab

# -Die Ablaeufe bilden
# -- abschreiben aus Matlab


# Das Events raus rechnen baue ich am Besten in den Training Step. Wie Preprocessing. 
# => Obwohl das natuerlich nur einen einzelnen Datensatz Sinn macht

# Kommentare zu Vgl mit sklearn:
# 1. kwargs ist dort eher verboten (siehe BaseEstimator Doku)
# 2. Es gibt eine Score Funktion 
#    Die ist in dem Beispiel in NilmTK nur von aussen ergaenzt ueber diese Predict Hilfsfunktion
#    In sklearn in Mixins beschrieben. Aber in Doku falsch wuerde ich sagen, dass ALLE estimators score() haben.



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
from nilmtk.disaggregate import UnsupervisedDisaggregator
from nilmtk.feature_detectors.cluster import hart85_means_shift_cluster
from nilmtk.feature_detectors.steady_states import find_steady_states_transients

# Fix the seed for repeatability of experiments
SEED = 42
np.random.seed(SEED)




class CustomBaranski(UnsupervisedDisaggregator):
    """ Customized Baranski Algorithm.
    Allows multiple states per machine and additionally supports 
    multiple states.

    Erweiterungen die in NILM Eval unterschlagen wurden:
    1. Die hochfrequenten events werden zusammen genommen (Das stand ja auch schon im Paper Seite 2)
    2. Die Erweiterung ist dann eben, dass man komplexe Events mit beruecksichtigt. Siehe das Nachfolgepaper.
    
    Probleme denen ich mich direkt stellen kann, die in Matlab auch nicht geloest wurden:
    1. Ueber die Zeit koennen sich die Anlagen veraendern -> Wenn ich alles betrachte ist mir das ja im Prinzip egal. 
    -> Das ist mir ja im Prinzip egal!
    2. Grosse Datenmengen: Es ist mit Problemen ueber mehrere Jahre 3 phasig zu rechnen.
    -> Der Featurespace ist DEUTLICH kleiner
    -> Ansaetze wiederholende Pattern zu erkennen
    => Komplexitaet Suchalgorithmus
    Datenstruktur: Alle Events in eine Reihe schreiben, 

    Attributes
    ----------
    model : dict
        Mal noch nichts. 
    """

    #region All disaggregator functions which are not used at the moment 

    def export_model(self, filename):
        raise NotImplementedError("Muss ich ggf noch mal von den anderen Klassen herkopieren.")


    def import_model(self, filename):
        raise NotImplementedError("Muss ich ggf noch mal von den anderen Klassen herkopieren.")

    #endregion


   
    #region Used disaggregator functions

    def __init__(self):
        self.MODEL_NAME = "BARANSKI"
        self.cols = [("power", "active")]
        self.noise_level = 70
        self.state_threshold = 15
        self.max_num_clusters = 12 # from matlab project

    
    def train(self, metergroup, **load_kwargs):
        """ Gets a site meter and trains the model based on it. 
        Goes chunkwise through the dataset and returns the events.
        In the end does a clustering for identifying the events.
        For signature description see basic class: It should get a sitemeter for unsupervised learning.

        Parameters
        ----------
        metergroup : a nilmtk.MeterGroup object
        For custom baranski (is unsupervised), this is a single site meter.
        """
        
        # Go through all parts and extract events
        events = []
        # 1. Get Events (Das ist ja schon vorhanden) -> Das sollte ich ausbauen -> GetSignatures
        # -> man separiert in die verschiedenen moeglichen Signaturen
        # --> Einen Separator als Oberklasse, Dann mehrere Separatoren fuer die einzelnen Typen an Effekt 
        #       -Rising Spike: 
        #       -Rising Spike:  
        #       -Pulse
        #       -Fluctuation
        #       -Quick Vibrate
        #       -Gradual Falling
        #       -Flatt
        # --> Man arbeitet mit Verdeckung: Event, verdeckt wenn RaisingSpike, FallingSpike, verdeckt wenn Pulse
        #
        # --> Jede Signatur hat eigene spezielle Eigenschaften
        # --> Einige sollten eine Wildcard beinhalten
        # Ich will hier ein 3d Pandas aufbauen
        #events = self._load_if_available()
        #if not events is None:
        #    self.events = events
        #    return

        events = pd.DataFrame()
        for i, elec in enumerate(metergroup.all_meters()):
            print("Find Events for " + str(elec.metadata))
            transitions = find_steady_states_transients(
                elec, cols=self.cols, state_threshold=self.state_threshold,
                noise_level=self.noise_level, **load_kwargs)[1]
            # Mark as on- or off-event
            transitions['type'] = transitions >= 0
            transitions['meter'] = elec
            events = events.append(transitions)

        events.index.rename('time', inplace=True)
        events.set_index(['type', 'meter'], append=True, inplace=True)
        events = events.reorder_levels([2,1,0])
        events.sort_index(inplace=True)
        # Hier vielleicht noch die Kombinationen finden
        self.events = events
        
        #self._save(events)


  
        # 2. Cluster the events using different cluster methodologies (Zuweisung passiert automatisch)
        # Ah es gibt doch ein predict: Und zwar elemente Clustern zuweisen        
        clusters = None #self.  _load_if_available(what='cluster')
        if clusters is None:
            for curGroup, groupEvents in events.groupby(['meter','type']):
                centroids, assignments = self._cluster_events(groupEvents, max_num_clusters=self.max_num_clusters, method='kmeans')
                events.loc[curGroup,'cluster'] = assignments  
            #self._save(events, 'cluster')
        else:
            pass
            #events = clusters

        self.model = events
        


    
    def train_on_chunk(self, chunk):
        """ 
        This function is actually not needed as the chunkwise processing is included inside the find_steady_states_transients function.
        This function goes through the power line and already identifies the events.
        For signature description see basic class: Only gets the chunk from the sitemeter, as it is unsupervised.

        Parameters
        ----------
        chunk : pd.DataFrame where each column represents a
            disaggregated appliance
        meter : ElecMeter for this chunk
        """
        pass

    def disaggregate(self, mains, output_datastore, **load_kwargs):
        """Disaggregate mains according to the model learnt previously.
        At the moment not used as we use the predict function in the main 
        script.

        Parameters
        ----------
        mains : nilmtk.ElecMeter or nilmtk.MeterGroup => In facts the 3 phases
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm. => Wird in einem 2. Step benutzt 
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        """
        
        # 3. Generate Finite State Machines (Test whether clustering fits)
        # 4. Bildender Sequenzen fuer jede State machine
        for meter, content in clusters.groupby(level=0):
            length = []
            for type, innerContent in content.groupby(level=1):
                length.append(len(innerContent.reset_index().cluster.unique()))
            if len(set(length)) > 1 :
                raise Exeption("different amount of clusters")
        
        clusters['appliance'] = pd.Series()
        clusterCenters = clusters.groupby(['meter','type','cluster']).mean()
        for meter, content in clusterCenters.groupby(level=0):
            ups = content.loc[meter,True]['active transition']
            downs = content.loc[meter,False]['active transition']
            appliancesUp, appliancesDown = self._findPairs(ups,downs)
            for i in range(len(ups)):
                clusters.loc[(meter, True, slice(None), i),['appliance']] = appliancesUp[i]
                clusters.loc[(meter, False, slice(None), i),['appliance']] = appliancesDown[i] 


        # 5. Einzelnd plotten, dh pro Phase (Ein plot fro appliance
        for instance, instanceEvents in events.groupby(level=0):
            instanceEvents.set_index('appliance', append="True", inplace=True)
            tmp = instanceEvents.reset_index([0,1,3], drop=True).unstack('appliance').sort_index()
            tmp2 = tmp.cumsum().fillna(method="ffill").fillna(method="bfill")
            tmp2.plot(subplots=True, figsize=(6, 6));
            plt.show()
            allEventsTimeline = pd.concat(events[instanceNumber]).cumsum()
            disaggredatedTimelines = [tmp.cumsum() for tmp in disaggregated[instanceNumber]]
            self._plot_with_subplots(mains[instanceNumber+1], allEventsTimeline, disaggregated)

        # Initially all appliances/meters are in unknown state (denoted by -1)
        prev = OrderedDict()
        learnt_meters = self.centroids.index.values
        for meter in learnt_meters:
            prev[meter] = -1

        timeframes = []
        # Now iterating over mains data and disaggregating chunk by chunk
        for chunk in mains.power_series(**load_kwargs):
            # Record metadata
            timeframes.append(chunk.timeframe)
            measurement = chunk.name
            power_df = self.disaggregate_chunk(
                chunk, prev, transients)

            cols = pd.MultiIndex.from_tuples([chunk.name])

            for meter in learnt_meters:
                data_is_available = True
                df = power_df[[meter]]
                df.columns = cols
                key = '{}/elec/meter{:d}'.format(building_path, meter + 2)
                output_datastore.append(key, df)

            output_datastore.append(key=mains_data_location,
                                    value=pd.DataFrame(chunk, columns=cols))

        if data_is_available:
            self._save_metadata_for_disaggregation(
                output_datastore=output_datastore,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                supervised=False,
                num_meters=len(self.centroids)
            )
    
    

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



    #endregion

    #region machine learning functionality help functions
    
    def _cluster_events(self, events, max_num_clusters=3, exact_num_clusters=None, method='kmeans'):
        ''' Applies clustering on the previously extracted events. 
        The _transform_data function can be removed as we are immediatly passing in the 
        pandas dataframe.

        Parameters
        ----------
        events : pd.DataFrame with the columns "PowerDelta, Duration, MaxSlope"
        max_num_clusters : int
        exact_num_clusters: int
        method: string Possible approaches are "kmeans" and "ward"
        Returns
        -------
        centroids : ndarray of int32s
            Power in different states of an appliance, sorted
            
        labels: ndarray of int32s
            The assignment of each event to the events
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

        # If the exact number of clusters are specified, then use that
        if exact_num_clusters is not None:
            labels, centers = _apply_clustering_n_clusters(clusteringInput, exact_num_clusters, method)
            return centers.flatten()

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



        # Postprocess and return clusters (weiss noch nicht ob das notwendig ist)
        centroids = np.append(centroids, 0)  # add 'off' state
        centroids = np.round(centroids).astype(np.int32)
        centroids = np.unique(centroids)  # np.unique also sorts
        return centroids
    

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



    def _cluster_on_offs(self, events, max_num_clusters=3, exact_num_clusters=None):
        ''' This function clusters together the on and off events to find applicances 
            that belong together.
        '''
        
        allEvents = list(events['pos'][0]) + [abs(k) for k in events['neg'][0]]* 3

        # First create the connectivity matrix
        positives = len(events['pos'])
        negatives = len(events['neg'])
        A = np.zeros((positives,positives))
        B = np.ones((positives,negatives))
        C = np.ones((negatives,positives))
        D = np.zeros((negatives,negatives))
        connectivityMatrix = np.bmat([[A, B], [C, D]])

        # Do the clustering
        num_clusters = -1
        silhouette = -1
        k_means_labels = {}
        k_means_cluster_centers = {}
        k_means_labels_unique = {}

        # Then do the hierarchical clustering
        for n_clusters in range(2, np.min([max_num_clusters,positives+1,negatives+1])):
            try:
                # Do a clustering for each amount of clusters
                
                k_means = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivityMatrix)
                k_means.fit(allEvents)
                labels = k_means.labels_
                centers = k_means.cluster_centers_
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
        
            except Exception as inst:
                if num_clusters > -1:
                    return k_means_cluster_centers[num_clusters]
                else:
                    return np.array([0])

        return k_means_labels[num_clusters]


    def _pandas_sklearn_test(self):
        data = pd.DataFrame({'pet':      ['cat', 'dog', 'dog', 'fish', 'cat', 'dog', 'cat', 'fish'], 
                             'children': [4., 6, 3, 3, 2, 3, 5, 4],
                             'salary':   [90, 24, 44, 27, 32, 59, 36, 27]})
        mapper = DataFrameMapper([('pet',None),('children', None)], input_df = True) 
             #('pet', sklearn.preprocessing.LabelBinarizer()),
             #(['children'], sklearn.preprocessing.StandardScaler())
        mapper.fit_transform(data.copy())
        print(str(mapper.transformed_names_))

    #endregion

    def _speedtest():
        s = pd.Series(np.random.randint(0,100,size=(1000000)))
        d = s.diff()
        target = d[d.abs() > 30]
        
    def _save(self, events, what='events'):
        #file = open("tmpEvents.pckle", "w+")
        #pickle.dump(events,file)
        #file.close()
        if what=='events':
            events.to_csv("tmpEvents.csv")
        else:
            events.to_csv("tmpClusters.csv")



    def _load_if_available(self, what='events'):
        #if os.path.isfile("tmpEvents.pckle"):
        #    file = open("tmpEvents.pckle", "r")
        #    tmp = pickle.load(file)
        #    file.close()
        #    return tmp
        target = ""
        if what == 'events':
            target = "tmpEvents.csv"
            columns = ['meter', 'type', 'time']
        else:
            target = "tmpClusters.csv"
            columns = ['meter', 'type', 'time', 'cluster']

        if os.path.isfile(target):
            return pd.read_csv(target, index_col=columns).sort_index()
        else:
            return None


    def _findPairs(self, a1, a2):
        up = range(len(a1))
        down = [None] * len(a2)
        for appliance in up:
            diff = sys.maxint
            minimumParter = None
            for partner in range(len(a2)):
                if not down[partner] is None:
                    continue
                curDiff = np.fabs(np.add(a1[appliance], a2[partner])) # Elementwise absolute differences
                if curDiff < diff:
                    diff = curDiff
                    minimumPartner = partner
            down[minimumPartner] = appliance
        return up, down