from __future__ import print_function, division
from collections import OrderedDict, deque
import time
from datetime import datetime
from nilmtk.feature_detectors.cluster import hart85_means_shift_cluster
from nilmtk.feature_detectors.steady_states import find_steady_states_transients
from nilmtk.disaggregate.accelerators import find_steady_states_fast, find_transients_baranskistyle_fast, find_transients_fast, pair_fast, find_sections
from nilmtk.timeframe import merge_timeframes, TimeFrame
from nilmtk.disaggregate import SupervisedDisaggregator, UnsupervisedDisaggregatorModel
import numpy as np
import pandas as pd
from functools import reduce, partial
from multiprocessing import Pool
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift
from collections import Counter
from sklearn_pandas import DataFrameMapper  
import pickle as pckl
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# Fix the seed for repeatability of experiments
SEED = 42
np.random.seed(SEED)


class EventbasedCombinationDisaggregatorModel(UnsupervisedDisaggregatorModel):        
    '''
    Attributes:
    steady_states: Per phase/sitemeter a list with the steady states (start, end, powerlevel)
    transients:    Per phase/sitemeter a list with the transients (start, delta, end, signature)
    centroids:    meta information centroids from clustering 
    [appliances].events: Per appliance/centroid the dataframe with load for each separated appliance
    '''
    params = {
        # The noise during measuring
        'noise_level': 10, 
        
        # the necessary delta to detect it as a load step
        'state_threshold': 15,

        # variance in power draw allowed for pairing a match
        'min_tolerance': 20, 
        
        # Minimum amount of steps to define it as a steady state
        'min_n_samples':2,
        
        # cols: nilmtk.Measurement, should be one of the following
        #  [('power','active')]
        #  [('power','apparent')]
        #  [('power','reactive')]
        #  [('power','active'), ('power', 'reactive')]
        'cols': [('power','active')],


        # If transition is greater than large_transition,
        # then use percent of large_transition
        'percent_tolerance': 0.035,
        
        # Lower boundary for a large transition
        'large_transition': 1000,
        
        # Maximal amount of time, an appliance can be on
        'max_on_length' : pd.Timedelta(hours = 2),


        # Take more switch events to enable more complex state machines.
        'max_baranski_clusters': 7,

        # Whether the Hart Algorithms is activated
        'hart_activated' : True,
        
        # Whether the Baranski Algorithms is activated
        'baranski_activated': True,

        # Amount of clusters built for Baranski
        'max_num_clusters': 12,

        # Weights for the optimiyer in baranski
        'gamma1': 0.2,
        'gamma2': 0.2,
        'gamma3': 0.2,

        # How often an event has to appear sothat it is counted as appliance
        'min_appearance': 100,

        # The sample period, in seconds, used for both the
        # mains and the disaggregated appliance estimates.
        'sample_period':120,
        
    }
    
    def extendModels(otherModel):
        '''
        For unsupervised learning, the extend function for the models 
        is even more important as for the supervised/transfer case,
        because each data to disaggregate can be seen as new 
        training data.        
        '''
        pass




def add_segments_improved(params):
#for i in range(len(metergroup)):
    transients, states, threshold, noise_level = params

    # For testing
    #tmp = self.model.steady_states[0][pd.Timestamp('2015-04-17 07:05:04.675000+00:00'):pd.Timestamp('2015-04-17 07:30:04.675000+00:00')]
    #indices = np.array(tmp.index)
    #values = np.array(tmp.iloc[:,0])

    indices = np.array(states.index)
    values = np.array(states.iloc[:,0])
    sections = find_sections((indices, values, noise_level))
    transients['segment'] = pd.Series(data = sections, index = transients.index).shift().bfill()
    transients = transients.join(transients.groupby("segment").size().rename('segmentsize'), on='segment')

    # Don't ask me, why error occured when I do this before the count
    transients['ends'] = transients['ends'].dt.tz_localize('utc')
    transients = transients.reset_index()

    # Additional features, which come in handy
    transients['spike'] =  transients['signature'].apply(lambda sig: np.cumsum(sig).max())
    transients['spike'] = transients['spike'].clip(0)
    #transients['spike'] +=  transients['signature'].apply(lambda sig: np.clip([np.cumsum(sig).min(), 0]))
    transients['length'] = transients['ends'] - transients['starts']

    return (transients, states)


def add_segments(params):
    transients, states, threshold, noise = params

    baseload_separators = (states<=(states.min() + threshold))
    transients['segment'] = baseload_separators.cumsum().shift(1).fillna(0).astype(int)
    transients = transients.join(transients.groupby("segment").size().rename('segmentsize'), on='segment')
    #tst.groupby(['ends_other']).count()['segment'].cumsum().plot()                 # Events per Segmentsize
    #tst.groupby(['segment']).mean().groupby("ends_other").count().cumsum().plot()  # Segments per SegmentSize
    #tst[tst['ends_other']==832]['active transition'].cumsum().plot()               # Print the Segment
                        
    # Don't ask me, why error occured when I do this before the count
    transients['ends'] = transients['ends'].dt.tz_localize('utc')
    transients = transients.reset_index()

    # Additional features, which come in handy
    transients['spike'] =  transients['signature'].apply(lambda sig: np.cumsum(sig).max())
    transients['spike'] = transients['spike'].clip(0)
    #transients['spike'] +=  transients['signature'].apply(lambda sig: np.clip([np.cumsum(sig).min(), 0]))
    transients['length'] = transients['ends'] - transients['starts']

    return (transients, states)


class EventbasedCombination(SupervisedDisaggregator):
    """ This is the final used disaggregator, which bases on the the 
    combination of the available event based disaggregators. In our case
    Baranski and Hardt.

    It works a little bit different concerning how results are stored. As 
    it only disaggregates state machines, it only stores the flanks and all
    other parts have to be reconstructed by using ffill.
    That reduces load times significantly. 
    (This approach makes accumulation in next step also much faster,
    sothat only the final results have to be in memory.)
    """

    """ Necessary attributes of any approached meter """
    Requirements = {
        'max_sample_period': 10,
        'physical_quantities': [['power','active']]
    }

    """ The related model """
    model_class = EventbasedCombinationDisaggregatorModel


    def __init__(self, model = None):
        if model == None:
            model = self.model_class();
        self.model = model;
        super(EventbasedCombination, self).__init__()

    def train(self, metergroup, output_datastore, **kwargs):
        """
        Trains and immediatly disaggregates

        Parameters
        ----------
        metergroup : a nilmtk.MeterGroup object
        
        output_datastore: a nilmtk.Datastore where the disaggregated meters are placed
        """
    
        # Prepare
        kwargs = self._pre_disaggregation_checks(metergroup, kwargs)
        kwargs.setdefault('sections', metergroup.good_sections().merge_shorter_gaps_than('10min'))#.drop_all_but(3))
        pool = None
        metergroup = metergroup.sitemeters() # Only the main elements are interesting
        model = self.model        
        model.steady_states = []
        model.transients = []
        model.centroids = [{}] * len(metergroup)
        model.clusterer = [{}] * len(metergroup)
        model.clusterlimits = [{}] * len(metergroup)
        model.appliances = [[]] * len(metergroup)
        model.overall_powerflow = overall_powerflow = [] 

                
        # 1. Load the events from the poweflow data
        print('Extract events')
        loader = []
        steady_states_list = []
        transients_list = []   
        try:
            self.model = model = pckl.load(open('E:disaggregation_events/' + str(metergroup.identifier) + '.pckl', 'rb'))
        except:
            pool = Pool(processes=4)
            t1 = time.time()
            for i in range(1):#len(metergroup)):
                overall_powerflow.append(None)
                steady_states_list.append([])
                transients_list.append([])
                loader.append(metergroup.meters[i].load(cols=self.model.params['cols'], chunksize = 31000000, **kwargs))
            try:
                while(True):
                    input_params = []
                    for i in range(1):#len(metergroup)):
                        power_dataframe = next(loader[i]).dropna()
                        if overall_powerflow[i] is None:
                            overall_powerflow[i] = power_dataframe.resample('5min').agg('mean')  
                        else:
                            overall_powerflow[i] = overall_powerflow[i].append(power_dataframe.resample('5min', how='mean'))
                        indices = np.array(power_dataframe.index)
                        values = np.array(power_dataframe.iloc[:,0])
                        input_params.append((indices, values, model.params['min_n_samples'], model.params['state_threshold'], model.params['noise_level']))
                    #states_and_transients, tst_original_fast, tst_original  = [], [], []
                    #for i in range(2,len(metergroup)):
                    #    states, transients = find_transients_fast(input_params[i])
                    #    steady_states_list[i].append(states)
                    #    transients_list[i].append(transients)
                    states_and_transients = pool.map(find_transients_fast, input_params)
                    for i in range(1):#len(metergroup)):
                        steady_states_list[i].append(states_and_transients[i][0])
                        transients_list[i].append(states_and_transients[i][1])

            except StopIteration:
                pass
            # set model (timezone is lost within c programming)
            for i in range(1):#len(metergroup)):
                model.steady_states.append(pd.concat(steady_states_list[i]).tz_localize('utc'))
                model.transients.append(pd.concat(transients_list[i]).tz_localize('utc')) #pd.merge(self.steady_states[0], self.steady_states[1], how='inner', suffixes=['_1', '_2'], indicator=True, left_index =True, right_index =True)
                model.transients[-1].index.rename("starts", inplace = True)
            t2 = time.time()
            print(t2-t1)
    
            # Create a fourth powerflow with events, common to all powerflows
            pckl.dump(model, open('E:disaggregation_events/' + str(metergroup.identifier) + '.pckl', 'wb'))
        # model.transients.append(self.separate_simultaneous_events(self.model.transients))
        

        # 2. Separate segments between base load        
        pool = Pool(processes=4) if pool is None else pool      
        t1 = time.time()
        input_params = []
        results = []
        for i in range(len(model.transients)):
            input_params.append((self.model.transients[i], self.model.steady_states[i], self.model.params['state_threshold'], self.model.params['noise_level']))            
            results.append(add_segments_improved(input_params[-1]))
        #results = pool.map(add_segments, input_params)
        for i,cur in enumerate(results):
            self.model.transients[i] = cur[0]
            self.model.steady_states[i] = cur[1]
        print(time.time() - t1)


        # 3. Create all events which per definition have to belong together (tuned Hart)
        t2 = time.time()
        for i in range(len(model.transients)):
            transients = model.transients[i]

            # Separate the two-flag events  # Sinnvolles Preprocessing
            twoevents = transients[transients['segmentsize'] == 2]
            f = {'starts':min, 'ends':max, 'active transition': [min, max], 'spike': max, 'length': sum} #lambda c: c.abs().sum()/2
            #f = {'active transition': lambda c: c.abs().sum()/2, 'spike': max}
            twoevents = twoevents.groupby('segment').agg(f)
            twoevents[('length','sum')] = twoevents[('length','sum')].apply(lambda e: e.seconds)
            twoevents[('duration','max')]=(twoevents[('ends','max')]-twoevents[('starts','min')]).apply(lambda e: e.seconds)
            #ax = plt.axes(projection='3d'); ax.scatter(tst[('active transition', 'avg')].values,tst[('duration', 'max')].values, tst[('spike', 'max')].values, s=0.1)
            #events.plot.scatter(('active transition', 'avg'),('duration', 'log'), c=(events['color']), s=1, colormap='gist_rainbow')
            #events.plot.scatter(('active transition', 'avg'),('duration', 'max'), c=events['color'], s=1)
            twoevents[('active transition', 'avg')] = (twoevents[('active transition','max')].abs() + twoevents[('active transition','min')].abs())/2
            twoevents = twoevents.drop([('ends','max'),('starts','min')], axis=1)
            centroids, labels, twoclusterer = self._cluster_events(twoevents, max_num_clusters = 20, all_columns=True) #
            centroids.columns = [' '.join(col).strip() + '_center' for col in centroids.columns.values] #[col + '_center' for col in centroids.columns]
            model.centroids[i]['2'] = centroids
            model.clusterer[i]['2'] = twoclusterer
            twoevents['appliance'] = labels
            transients = transients.join(twoevents[['appliance']], on="segment")
            # Calc limits
            tmp = twoevents.join(model.centroids[i]['2'], on='appliance', rsuffix='_center').dropna()
            tmp['active transition_maxdelta'] = (tmp['active transition'] - tmp['active transition_center']).abs()
            tmp['spike_maxdelta'] = (tmp['spike'] - tmp['spike_center']).abs()
            tmp = tmp.groupby('appliance').max()[['active transition_maxdelta', 'spike_maxdelta']]
            model.clusterlimits[i]['2'] = tmp[['active transition_maxdelta', 'spike_maxdelta']]
            print("Two events: " + str(time.time() - t2))

            ## Separate the three-flag events
            threeevents = transients[transients['segmentsize'] == 3]
            func = lambda x: x['active transition'].values
            threeevents = threeevents.groupby('segment').apply(func)
            threeevents = pd.DataFrame(index = threeevents.index, data = list(threeevents.values), columns = ['fst', 'sec', 'trd'])
            threeevents_typed = [threeevents[threeevents['sec'] >= 0], threeevents[threeevents['sec'] < 0]]
            for type in range(2):
                centroids, labels, clusterer= self._cluster_segments(threeevents_typed[type])
                centroids.columns = [col + '_center' for col in centroids.columns]
                if type == 0:
                    model.centroids[i]['3'] = centroids
                    model.clusterer[i]['3'] = clusterer
                else:
                    labels[labels != -1] = labels[labels != -1] + len(model.centroids[i]['3'])
                    model.centroids[i]['3'] = model.centroids[i]['3'].append(centroids).reset_index(drop=True)
                    model.clusterer[i]['3'].append(clusterer)
                threeevents_typed[type]['appliance'] = labels
            threeevents = pd.concat(threeevents_typed)
            transients = transients.join(threeevents[['appliance']], on="segment", rsuffix="three")
            transients = transients.combine_first(transients[['appliancethree']].rename(columns={'appliancethree':'appliance'}))
            transients.drop(["appliancethree"],axis=1, inplace = True)
            # Calc limits
            tmp = threeevents.join(model.centroids[i]['3'], on='appliance', rsuffix='_center').dropna()
            tmp['fst_maxdelta'] = (tmp['fst'] - tmp['fst_center']).abs()
            tmp['sec_maxdelta'] = (tmp['sec'] - tmp['sec_center']).abs()
            tmp['trd_maxdelta'] = (tmp['trd'] - tmp['trd_center']).abs()
            model.clusterlimits[i]['3'] = tmp.groupby('appliance').max()[['fst_maxdelta', 'sec_maxdelta', 'trd_maxdelta']]
            print("Three events: " + str(time.time() - t2))

            # Separate the four-flag events
            allfourevents = transients[transients['segmentsize']==4]
            allfoureventsgrouped = allfourevents.groupby('segment')
            fourevents = allfoureventsgrouped.apply(func)
            fourevents = pd.DataFrame(index=fourevents.index, data=list(fourevents.values), columns = ['fst', 'sec', 'trd', 'fth'])

            # First look, whether just overlapping or sequential twoevents
            overlapping = (fourevents['fst']>0) & (fourevents['sec']>0) & (fourevents['trd']<0) & (fourevents['fth']<0)  
            d1 = (fourevents['fst'] + fourevents['trd']).abs() + (fourevents['sec'] + fourevents['fth']).abs()
            d2 = (fourevents['fst'] + fourevents['fth']).abs() + (fourevents['sec'] + fourevents['trd']).abs()
            fourevents['overlap1'] = overlapping & (d1 < d2)
            fourevents['overlap2'] = overlapping & (d2 < d1)
            fourevents['sequential'] = ((fourevents['fst'] + fourevents['sec']) < self.model.params['state_threshold'])
            f = {'active transition': lambda c: c.abs().sum()/2, 'spike': max}
            allfourevents = allfourevents.join(fourevents[['overlap1','overlap2','sequential']], on='segment')

            overlap1 = allfourevents[allfourevents['overlap1']].drop('appliance', axis=1)
            overlap1['grp'] = np.array([0,1,0,1]*(len(overlap1)//4)) + np.outer(range(len(overlap1)//4),np.array([2,2,2,2])).flatten()
            possible_twoevents = overlap1.groupby(overlap1['grp']).agg(f)
            possible_twoevents['appliance']= self._check_for_membership(possible_twoevents, model.clusterer[i]['2'], model.centroids[i]['2'], model.clusterlimits[i]['2'])
            overlap1 = overlap1.join(possible_twoevents[['appliance']], on='grp')
            transients = transients.join(overlap1[['appliance']], on="segment", rsuffix="_overlap1")
            transients = transients.combine_first(transients[['appliance_overlap1']].rename(columns={'appliance_overlap1':'appliance'}))
            transients.drop(["appliance_overlap1"], axis=1, inplace = True)

            overlap2 = allfourevents[allfourevents['overlap2']].drop('appliance', axis=1)
            overlap2['grp'] = np.array([0,1,1,0]*(len(overlap2)//4)) + np.outer(range(len(overlap2)//4),np.array([2,2,2,2])).flatten()
            possible_twoevents = overlap2.groupby(overlap2['grp']).agg(f)
            possible_twoevents['appliance'] = self._check_for_membership(possible_twoevents, model.clusterer[i]['2'], model.centroids[i]['2'], model.clusterlimits[i]['2'])
            overlap2 = overlap2.join(possible_twoevents[['appliance']], on='grp')
            transients = transients.join(overlap2[['appliance']], on="segment", rsuffix="_overlap2")
            transients = transients.combine_first(transients[['appliance_overlap2']].rename(columns={'appliance_overlap2':'appliance'}))
            transients.drop(["appliance_overlap2"], axis=1, inplace = True)

            sequential = allfourevents[allfourevents['sequential']].drop('appliance', axis=1)
            sequential['grp'] = np.array([0,0,1,1]*(len(sequential)//4)) + np.outer(range(len(sequential)//4),np.array([2,2,2,2])).flatten()
            possible_twoevents = sequential.groupby(sequential['grp']).agg(f)
            possible_twoevents['appliance']= self._check_for_membership(possible_twoevents, model.clusterer[i]['2'], model.centroids[i]['2'], model.clusterlimits[i]['2'])
            sequential = sequential.join(possible_twoevents[['appliance']], on='grp')
            transients = transients.join(sequential[['appliance']], on="segment", rsuffix="_overlap2")
            transients = transients.combine_first(transients[['appliance_overlap2']].rename(columns={'appliance_overlap2':'appliance'}))
            transients.drop(["appliance_overlap2"], axis=1, inplace = True)

            # All other events are fourevents
            fourevents_types = ((fourevents['sec'] > 0).astype(int) + (fourevents['trd'] > 0).astype(int)*2,0)[0]
            fourevents_typed = [fourevents[fourevents_types == curtype] for curtype in range(4)]
            for type in range(4):
                centroids, labels, clusterer = self._cluster_segments(fourevents_typed[type][['fst','sec','trd','fth']])
                if type == 0:
                    model.centroids[i]['4'] = centroids
                    model.clusterer[i]['4'] = clusterer
                else:
                    labels[labels != -1] = labels[labels != -1] + len(model.centroids[i]['4'])
                    model.centroids[i]['4'] = model.centroids[i]['4'].append(centroids).reset_index(drop=True)
                    model.clusterer[i]['4'].append(clusterer)
                fourevents_typed[type]['appliance'] = labels
            fourevents = pd.concat(fourevents_typed)
            transients = transients.join(fourevents[['appliance']], on="segment", rsuffix="four")
            transients = transients.combine_first(transients[['appliancefour']].rename(columns={'appliancefour':'appliance'}))
            transients.drop(["appliancefour"],axis=1, inplace = True)
            # Calc limits
            tmp = fourevents.join(model.centroids[i]['4'], on='appliance', rsuffix='_center').dropna()
            tmp['fst_maxdelta'] = (tmp['fst'] - tmp['fst_center']).abs()
            tmp['sec_maxdelta'] = (tmp['sec'] - tmp['sec_center']).abs()
            tmp['trd_maxdelta'] = (tmp['trd'] - tmp['trd_center']).abs()
            tmp['fth_maxdelta'] = (tmp['fth'] - tmp['fth_center']).abs()
            model.clusterlimits[i]['4'] = tmp.groupby('appliance').max()[['fst_maxdelta', 'sec_maxdelta', 'trd_maxdelta', 'fth_maxdelta']]
            
            model.transients[i] = transients 
        print("All short events: " + str(time.time() - t2))

        # Larger segments handled Baranski Style
        # 1. Dh. ich clustere die Events der eindeutig identifiezierten Appliances von oben nach den genauen Specs
        # 2. Dann gehe ich durch jede Segmentgroesse
        #   3. Ich bestimme die Eigenschaften jeden Events u. clustere die Events fuer jede Section
        #   4. Ich merke mir fuer jede Sektion welche Ergebnisse wie haeufig vorkommen + Verteilung der Events
        # 4. Ich clustere hier nach über die Sktionen hinweg.
        #http://scikit-learn.org/stable/auto_examples/cluster/plot_ward_structured_vs_unstructured.html#sphx-glr-auto-examples-cluster-plot-ward-structured-vs-unstructured-py
        # 4. Die verbleibenden Events prüfe ich dann auf fit mit meinen 2-4 State appliances.
        # Man koennte noch beruechsichtigen: Delta zu vorherigem Event als Feature
        for i in range(len(model.transients)):
            transients = model.transients[i]

            segmentsizes = transients['segmentsize'].unique()
            segmentevents = {}
            # Ggf aufsteigend durchgehen: for segmentsize in segmentsizes[np.where(t > 4)[0]:]:
            cur_events = transients[transients['segmentsize']>4]
            cluster_ext = None
            for segno, segment in cur_events.groupby('segment'):
                cluster, labels, clusterer = self._cluster_events(segment[['active transition','spike']], max_num_clusters=self.model.params['max_baranski_clusters'], all_columns = True)
                segment['labels'] = labels
                # For each cluster determine attributes of how the events appear
                labeltimes = segment.groupby('labels').agg({'starts':min, 'ends':max})
                labeltimes = (labeltimes['ends'] - labeltimes['starts']).apply(lambda e: e.seconds).rename('td')
                _, cnt = np.unique(labels, return_counts=True)
                new = pd.concat([cluster,pd.Series(cnt, name = 'cnt'), labeltimes], axis=1)
                cluster_ext = new if cluster_ext is None else cluster_ext.append(new)
        
            # Now cluster the clusters
            cluster, labels, clusterer = self._cluster_events(cluster_ext, max_num_clusters=self.model.params['max_baranski_clusters'], all_columns = True)
            cluster_ext['labels'] = labels
            segment.join(cluster_ext['labels'], on="labels", rsuffix='_cluster')

            # Now check between segments whether same 
            f = lambda a: set(a)
            segmentsperappliance = groupby('appliance').agg({'segment':f})
            # Cluster the segmentevents
            cluster, label, clusterer = self._cluster_events(segmentevents)
            # Die Label anhaengen

            # Nach Segment 
        
            # self._check
            # Schauen ob es faelle gibt, wo gleiche Cluster in gleichen Sektionen auftauchen
            # Zusammenfassen

            # Die anderen Einzelnd

            # Bei denen die in gar kein großes Cluster passen, schauen ob es zu den 4er Events passt 


        # Fertig create the appliances (Pay attention, id per size)
        for i in range(len(metergroup)):
            transients = model.transients[i]
            for (size, appliance), group in transients.groupby(['segmentsize','appliance']):
                if appliance == -1 or size == 1 or (len(group) / size) < self.model.params['min_appearance']:
                    continue
                power = group.set_index('starts')[['active transition']]

                # Correction of errors
                error = power.groupby(np.outer(range(len(group)//size), np.ones(size)).flatten().astype(int)).sum()
                power.update(power.iloc[::size,0] - error.values.flatten())
                model.appliances[i].append(power.cumsum())
                
                
                #power = power.cumsum()
                #all_events = all_events.append(pd.DataFrame(0, columns= col, index=all_events[all_events < 0].dropna().index + pd.Timedelta('0.5sec')))
                #all_events = all_events.append(pd.DataFrame(0, columns= col, index=all_events[all_events > 0].dropna().index - pd.Timedelta('0.5sec')))
                #col = overall_powerflow[0].columns
                #off_events = group[['T2 Time', 'T2 Active']].set_index('T2 Time')
                #off_events.index.names = ['ts']
                #off_events.columns = overall_powerflow[0].columns
                #group = group[['T1 Time', 'T1 Active']].set_index('T1 Time')
                #group.index.names = ['ts']
                #group.columns = col
                #all_events = pd.concat([group, off_events], axis = 0).sort_index()
                ## Add zeros before uprising and after falling flags to enable interpolating
                #all_events = all_events.append(pd.DataFrame(0, columns= col, index=all_events[all_events < 0].dropna().index + pd.Timedelta('0.5sec')))
                #all_events = all_events.append(pd.DataFrame(0, columns= col, index=all_events[all_events > 0].dropna().index - pd.Timedelta('0.5sec')))
                ## Add zero in the end and beginning
                #all_events.loc[overall_powerflow[0].index[0]] = 0
                #all_events.loc[overall_powerflow[0].index[-1]] = 0
                #all_events.sort_index(inplace = True)
                #model.appliances[i].append(all_events.abs().astype(np.float32))

        
        # 2. Der Schritt ist noch eine Ueberlegung wert
        #if (self.model.params['hart_activated']):
        #    # Do the pairing of events
        #    print('Pair')
        #    input_params = []
        #    for cur in model.transients:
        #        input_params.append((cur, model.params['min_tolerance'], model.params['percent_tolerance'], model.params['large_transition']))#, model.params['max_on_length']))
        #    model.pair_df = pool.map(pair_fast, input_params)
        #    for i in range(len(model.pair_df)):
        #        model.pair_df[i]['T1 Time'] = model.pair_df[i]['T1 Time'].astype("datetime64").dt.tz_localize('utc')
        #        model.pair_df[i]['T2 Time'] = model.pair_df[i]['T2 Time'].astype("datetime64").dt.tz_localize('utc')
        
        # 3. Process events the baranski way
        #if (self.model.params['baranski_activated']):
        #    baranski_labels = []
        #    for i in range(len(model.transients)):
        #        centroids, labels = self._cluster_events(model.transients[i], max_num_clusters=self.model.max_baranski_clusters, method='kmeans')
        #        model.transients[i].append(labels)
                
        #        # Cluster events
        #        model.transients['type'] = transitions >= 0
        #        for curGroup, groupEvents in events.groupby(['meter','type']):
        #            centroids, assignments = self._cluster_events(groupEvents, max_num_clusters=self.max_num_clusters, method='kmeans')
        #            events.loc[curGroup,'cluster'] = assignments 

        #        # Build sequences for each state machine
        #        for meter, content in clusters.groupby(level=0):
        #            length = []
        #            for type, innerContent in content.groupby(level=1):
        #                length.append(len(innerContent.reset_index().cluster.unique()))
        #            if len(set(length)) > 1 :
        #                raise Exeption("different amount of clusters")
        

        # Create the powerflow remaining after disaggregation (only 5 min resolution)
        print('Create rest powerflow')
        for i in range(len(model.transients)):
            for j, appliance in enumerate(model.appliances[i]):
                # Resample the appliances to 5min (weighted mean)
                #newDF = pd.DataFrame(index = overall_powerflow[0].index, columns = overall_powerflow[0].columns)
                #model.appliances[i][j] = appliance.asfreq('10s', method='ffill').resample('5min', how='mean')
                def myresample(d):
                    weights = np.append(np.diff(d.index),np.timedelta64(4,'s') - (d.index[-1] - d.index[0]))
                    weights = weights.astype('timedelta64[ms]').astype(int)
                    return np.average(d, weights=weights)
                new_idx = pd.DatetimeIndex(start = appliance[:1].resample('5min').index[0], end = appliance[-2:].resample('5min').index[-1], freq = '5min')
                newSeries = pd.Series(index = new_idx)
                resampledload = appliance.append(newSeries).sort_index()
                resampledload = resampledload.ffill().bfill()
                model.appliances[i][j] = resampledload.resample('5min', how=myresample)

                # Subract from overall powerflow
                new_overall = overall_powerflow[i].loc[model.appliances[i][j].index] - model.appliances[i][j].values
                overall_powerflow[i].update(new_overall)

        # Store the results
        print('Store')
        for phase in range(len(model.transients)):
            building_path = '/building{}'.format(metergroup.building() * 10 + phase)
            for i, appliance in enumerate(self.model.appliances[phase]):
                key = '{}/elec/meter{:d}'.format(building_path, i + 2) # Weil 0 nicht gibt und Meter1 das undiaggregierte ist und 
                output_datastore.append(key, appliance) 
            output_datastore.append('{}/elec/meter{:d}'.format(building_path, 1), overall_powerflow[phase if phase < 3 else phase-1])

        # Store the metadata (Add one for the rest load profile)
        num_meters = [len(cur) + 1 for cur in self.model.appliances] 
        self._save_metadata_for_disaggregation(
            output_datastore=output_datastore,
            sample_period = kwargs['sample_period'] if 'sample_period' in kwargs else 2,
            measurement=overall_powerflow[0].columns,
            timeframes=list(kwargs['sections']),
            building=metergroup.building(),
            supervised=False,
            num_meters=num_meters,
            original_building_meta=metergroup.meters[0].building_metadata
        )

  

    def separate_simultaneous_events(self, transient_list):
        '''
        This function finds the common transients in the list of transients.
        The equal ones are removed from the timelines and then returned as 
        a dedicated timeline.
        '''
        # When only one powerflow no simultaneous events
        if len(transient_list) <= 1:
            return

        # Create sets and assign values
        simultaneous = reduce(lambda left, right: pd.merge(left, right, how='inner', left_index =True, right_index =True), transient_list)
        simultaneous = simultaneous[abs(simultaneous) > 1000].dropna()

        # Remove the simultaneous events from the previous timeflows
        for transients in transient_list:
            transients.drop(simultaneous.index, inplace = True)
        return pd.DataFrame(simultaneous.sum(axis=1), columns=['active transition'])

 

    def _save_metadata_for_disaggregation(self, output_datastore,
                                          sample_period, measurement,
                                          timeframes, building,
                                          meters=None, num_meters=None,
                                          supervised=True, original_building_meta = None):
        """Add metadata for disaggregated appliance estimates to datastore.

        This method returns nothing.  It sets the metadata
        in `output_datastore`.

        Note that `self.MODEL_NAME` needs to be set to a string before
        calling this method.  For example, we use `self.MODEL_NAME = 'CO'`
        for Combinatorial Optimisation.

        Parameters
        ----------
        output_datastore : nilmtk.DataStore subclass object
            The datastore to write metadata into.
        sample_period : int
            The sample period, in seconds, used for both the
            mains and the disaggregated appliance estimates.
        measurement : 2-tuple of strings
            In the form (<physical_quantity>, <type>) e.g.
            ("power", "active")
        timeframes : list of nilmtk.TimeFrames or nilmtk.TimeFrameGroup
            The TimeFrames over which this data is valid for.
        building : int
            The building instance number (starting from 1)
        supervised : bool, defaults to True
            Is this a supervised NILM algorithm?
        meters : list of nilmtk.ElecMeters, optional
            Required if `supervised=True`
        num_meters : [int]
            Required if `supervised=False`, Gives for each phase amount of meters
        """

        # TODO: `preprocessing_applied` for all meters
        # TODO: submeter measurement should probably be the mains
        #       measurement we used to train on, not the mains measurement.

        # DataSet and MeterDevice metadata:
        meter_devices = {
            'disaggregate' : {
                'model': str(EventbasedCombinationDisaggregatorModel), #self.model.MODEL_NAME,
                'sample_period': 0,         # This makes it possible to use the special load functionality later
                'max_sample_period': sample_period,
                'measurements': [{
                    'physical_quantity': measurement.levels[0][0],
                    'type': measurement.levels[1][0]
                }]
            },
            'rest': {
                'model': 'rest',
                'sample_period': sample_period,
                'max_sample_period': sample_period,
                'measurements': [{
                    'physical_quantity': measurement.levels[0][0],
                    'type': measurement.levels[1][0]
                }]
            }
        }

        # HIERM USS ICH ICH EBEN AUS DEN MEHREREN TIMEFRAMES UEBER DIE PHASEN DIE AEUSSERE TIMEFRAME BESTIMMEN
        merged_timeframes = merge_timeframes(timeframes, gap=sample_period)
        total_timeframe = TimeFrame(merged_timeframes[0].start,
                                    merged_timeframes[-1].end)

        date_now = datetime.now().isoformat().split('.')[0]
        dataset_metadata = {
            'name': str(EventbasedCombinationDisaggregatorModel),
            'date': date_now,
            'meter_devices': meter_devices,
            'timeframe': total_timeframe.to_dict()
        }
        output_datastore.save_metadata('/', dataset_metadata)
        
        
        # Building metadata
        for i in range(3):
            phase_building = building * 10 + i 
            building_path = '/building{}'.format(phase_building)
            mains_data_location = building_path + '/elec/meter1'


            # Rest meter:
            elec_meters = {
                1: {
                    'device_model': 'rest',
                    #'site_meter': True,
                    'data_location': mains_data_location,
                    'preprocessing_applied': {},  # TODO
                    'statistics': {
                        'timeframe': total_timeframe.to_dict()
                    }
                }
            }

            def update_elec_meters(meter_instance):
                elec_meters.update({
                    meter_instance: {
                        'device_model': 'disaggregate', # self.MODEL_NAME,
                        'submeter_of': 1,
                        'data_location': (
                            '{}/elec/meter{}'.format(
                                building_path, meter_instance)),
                        'preprocessing_applied': {},  # TODO
                        'statistics': {
                            'timeframe': total_timeframe.to_dict()
                        }
                    }
                })

            # Appliances and submeters:
            appliances = []
            if supervised:
                for meter in meters:
                    meter_instance = meter.instance()
                    update_elec_meters(meter_instance)

                    for app in meter.appliances:
                        appliance = {
                            'meters': [meter_instance],
                            'type': app.identifier.type,
                            'instance': app.identifier.instance 
                            # TODO this `instance` will only be correct when the
                            # model is trained on the same house as it is tested on
                            # https://github.com/nilmtk/nilmtk/issues/194
                        }
                        appliances.append(appliance)

                    # Setting the name if it exists
                    if meter.name:
                        if len(meter.name) > 0:
                            elec_meters[meter_instance]['name'] = meter.name
            else:  # Unsupervised
                # Submeters:
                # Starts at 2 because meter 1 is mains.
                for chan in range(2, num_meters[i] + 1): # Additional + 1 because index 0 skipped
                    update_elec_meters(meter_instance=chan)
                    appliance = {
                        'meters': [chan],
                        'type': 'unknown',
                        'instance': chan - 1
                        # TODO this `instance` will only be correct when the
                        # model is trained on the same house as it is tested on
                        # https://github.com/nilmtk/nilmtk/issues/194
                    }
                    appliances.append(appliance)

            building_metadata = {
                'instance': (phase_building),
                'elec_meters': elec_meters,
                'appliances': appliances,
                'original_name': original_building_meta['original_name'],
                'geo_location': original_building_meta['geo_location'] if 'geo_location' in original_building_meta else None,
                'zip': original_building_meta['zip'] if 'zip' in original_building_meta else None,
            }

            output_datastore.save_metadata(building_path, building_metadata)
   


    #region Clustering steps

    
    def _cluster_segments(self, cluster_df, method = 'kmeans'):
        '''
        Does the clustering in two steps to find everything
        '''

        if len(cluster_df)-1 <= len(cluster_df.columns):
            return pd.DataFrame(columns=cluster_df.columns), (-1)*np.ones(len(cluster_df)), []


        # First clustering
        X = cluster_df.values.reshape((len(cluster_df.index), len(cluster_df.columns)))
        clusterer = KMeans(n_jobs = 2)
        clusterer.fit(X)
        clusterer1 = MeanShift(cluster_all = False, min_bin_freq = 20, n_jobs = 1)
        clusterer1.fit(X)
        labels = clusterer1.labels_
        cluster_centers = clusterer1.cluster_centers_
        clusterer = [clusterer1]

        # Do a second round if necessary
        rest_idx = labels == -1
        if rest_idx.sum() > len(cluster_df.columns)+1:
            clusterer2 = MeanShift(cluster_all = False, min_bin_freq = 20, n_jobs = 1)
            clusterer2.fit(cluster_df[rest_idx])
            labels_rest = clusterer2.labels_
            centers_rest = clusterer2.cluster_centers_
            labels_rest[labels_rest != -1] = labels_rest[labels_rest != -1] + len(cluster_centers)
            labels[rest_idx] = labels_rest
            cluster_centers = np.append(cluster_centers, centers_rest, axis=0)
            clusterer.append(clusterer2)

        return pd.DataFrame(cluster_centers, columns=cluster_df.columns), labels, clusterer

    
    def _check_for_membership(self, cluster_df, clusterers, clusters, cluster_limits):
        '''
        The
        '''
        if len(cluster_df) == 0:
            return pd.DataFrame(columns=['label'])

        # Prepare for checking
        original_columns = cluster_df.columns
        X = cluster_df.values.reshape((len(cluster_df.index), len(cluster_df.columns)))
        cluster_df['label'] = np.nan
        labels = []
        prevlabels = 0
        for clusterer in clusterers:
            # Find appliance
            curlabels = clusterer.predict(X)
            curlabels[curlabels != -1] = curlabels[curlabels != -1] + prevlabels
            prevlabels += len(clusterer.cluster_centers_)

            # Check validity
            cluster_df['tmp_label'] = curlabels 
            cluster_df = cluster_df.join(cluster_limits, on='tmp_label', rsuffix="_limit")
            cluster_df = cluster_df.join(clusters, on='tmp_label', rsuffix="_center")
            valid = np.ones(len(cluster_df), dtype=bool)
            for original_column in original_columns:
                delta = cluster_df[original_column] - cluster_df[original_column + '_center']
                valid &= delta < cluster_df[original_column + '_maxdelta']
            
            cluster_df.update(cluster_df[['tmp_label']][valid].rename(columns={'tmp_label':'label'}))
            #pd.DataFrame({'label':((cluster_df['tmp_label'] * (valid.astype(int)*2-1)) -1 + valid.astype(int)).clip(-1)}))
            
        return cluster_df[['label']]
            

    def _cluster_events(self, events, max_num_clusters=5, exact_num_clusters=None, method='kmeans' , all_columns = False):
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
        if all_columns:
            mapper = DataFrameMapper([(column, None) for column in events.columns])
        else:
            mapper = DataFrameMapper([('active transition', None)]) 

        clusteringInput = mapper.fit_transform(events.copy())
    
        # Do the clustering
        num_clusters = -1
        silhouette = -1
        k_means_labels = {}
        k_means_cluster_centers = {}
        k_means_labels_unique = {}
        clusterer = {}

        # If the exact number of clusters are specified, then use that
        if exact_num_clusters is not None:
            labels, centers = _apply_clustering_n_clusters(clusteringInput, exact_num_clusters, method)
            return centers.flatten()

        # Special case:
        if len(events) == 1: 
            return np.array([events.iloc[0]["active transition"]]), np.array([0])

        # If exact cluster number not specified, use cluster validity measures to find optimal number
        for n_clusters in range(3, max_num_clusters): #ACHTUNG AUF 3 gestellt
            try:
                # Do a clustering for each amount of clusters
                labels, centers, clusterer = self._apply_clustering_n_clusters(clusteringInput, n_clusters, method)
                k_means_labels[n_clusters] = labels
                k_means_cluster_centers[n_clusters] = centers
                k_means_labels_unique[n_clusters] = np.unique(labels)
                clusterer[n_clusters] = clusterer

                # Then score each of it and take the best one
                try:
                    sh_n = silhouette_score(events, k_means_labels[n_clusters], metric='euclidean', sample_size = min(len(k_means_labels[n_clusters]),5000))

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

        return pd.DataFrame(k_means_cluster_centers[num_clusters], columns=events.columns), k_means_labels[num_clusters], clusterer[num_clusters]



        # Postprocess and return clusters (weiss noch nicht ob das notwendig ist)
        centroids = np.append(centroids, 0)  # add 'off' state
        centroids = np.round(centroids).astype(np.int32)
        centroids = np.unique(centroids)  # np.unique also sorts
        return centroids
    
    def plot_results(X, Y_, means, covariances, index, title):
        splot = plt.subplot(2, 1, 1 + index)
        for i, (mean, covar, color) in enumerate(zip(
                means, covariances, color_iter)):
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y_ == i):
                continue
            plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)

            plt.xlim(-9., 5.)
            plt.ylim(-3., 6.)
            plt.xticks(())
            plt.yticks(())
            plt.title(title)


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
        elif method == 'gmm':
            gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full')
            gmm.fit(X)
            return (gmm.means_, gmm.covariances_), g
    


    def CombiningClustersToFSM():
        params = self.model['params']
        #h = len()
        Q1 = lambda cluster_powers: (cluster_powers.sum().abs()) / cluster_powers.abs().max()
        Q2 = lambda cluster_power, cluster_content: (cluster_content*cluster_power).sum().abs() / cluster_powers.abs().max()
        #Q3 = lambda ()
        Q = params['gamma1'] * Q1(powers) + params['gamma1'] * Q2(powers) + params['gamma1'] * Q3(powers)


    #endregion



    #region So far unsused 
    

    def calc_tolerance(self, value, match_target):
        if (abs(value - match_target)) < self.model.params['large_transition']:
            matchtol = self.model.params['min_tolerance']
        else: 
            matchtol = self.model.params['percent_tolerance'] * max(np.fabs([value, match_target]))
        return matchtol



    def disaggregate_chunk(self, chunk, prev, transients, phase):
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
        model = self.model

        load_kwargs = self._pre_disaggregation_checks(mains, load_kwargs)
        load_kwargs.setdefault('sample_period', 60)
        load_kwargs.setdefault('sections', mains.good_sections())

        states = pd.DataFrame(
            np.NaN, index=chunk.index, columns= model.centroids[phase].index.values)
        for transient_tuple in transients.itertuples():
            
            # Transient in chunk
            if chunk.index[0] < transient_tuple[0] < chunk.index[-1]:
                # Absolute value of transient
                abs_value = np.abs(transient_tuple[1:])
                positive = transient_tuple[1] > 0
                abs_value_transient_minus_centroid = pd.DataFrame(
                    (model.centroids[phase] - abs_value).abs())
                if len(transient_tuple) == 2:
                    # 1d data
                    index_least_delta = abs_value_transient_minus_centroid.idxmin().values[0]
                    value_delta = abs_value_transient_minus_centroid.iloc[index_least_delta].values[0]
                    # Only accept it if it really fits exactly
                    tolerance = self.calc_tolerance(abs_value[0], model.centroids[phase].iloc[index_least_delta].values[0]) 
                    if value_delta > tolerance:
                        continue

                    # ALSO HIER MUSS ICH SCHAUEN OB ES DURCHLAEUFT. Und sonst koennte ich auch direkt vom Trainieren die Disaggregation nehmen

                else:
                    # 2d data.
                    # Need to find absolute value before computing minimum
                    columns = abs_value_transient_minus_centroid.columns
                    abs_value_transient_minus_centroid["multidim"] = (
                        abs_value_transient_minus_centroid[columns[0]] ** 2
                        +
                        abs_value_transient_minus_centroid[columns[1]] ** 2)
                    index_least_delta = (
                        abs_value_transient_minus_centroid["multidim"].argmin())
                if positive:
                    # Turned on
                    states.loc[transient_tuple[0]][index_least_delta] = model.centroids[phase].ix[index_least_delta].values
                else:
                    # Turned off
                    states.loc[transient_tuple[0]][index_least_delta] = 0
        #prev = states.iloc[-1].to_dict()
        states['rest'] = chunk - states.ffill().fillna(0).sum(axis=1)
        return states.dropna(how='all') #pd.DataFrame(states, index=chunk.index)


    def translate_switches_to_power(self, states_chunk):
        '''
        This function translates the states into power. The intermedite 
        steps are not reconstructed, as this does not have to be stored.
        It will be faster to do this manually after loading.
        '''
        model = self.model
        di = {}
        ndim = len(model.centroids[phase].columns)
        for appliance in states_chunk.columns:
            states_chunk[[appliance]][ states_chunk[[appliance]]==1] = model.centroids[phase].ix[appliance].values
        return states_chunk


    def assign_power_from_states(self, states_chunk, prev, phase):

        di = {}
        ndim = len(self.model.centroids[phase].columns)
        for appliance in states_chunk.columns:
            values = states_chunk[[appliance]].values.flatten()
            if ndim == 1:
                power = np.zeros(len(values), dtype=int)
            else:
                power = np.zeros((len(values), 2), dtype=int)
            # on = False
            i = 0
            while i < len(values) - 1:
                if values[i] == 1:
                    # print("A", values[i], i)
                    on = True
                    i = i + 1
                    power[i] = self.model.centroids[phase].ix[appliance].values
                    while values[i] != 0 and i < len(values) - 1:
                        # print("B", values[i], i)
                        power[i] = self.model.centroids[phase].ix[appliance].values
                        i = i + 1
                elif values[i] == 0:
                    # print("C", values[i], i)
                    on = False
                    i = i + 1
                    power[i] = 0
                    while values[i] != 1 and i < len(values) - 1:
                        # print("D", values[i], i)
                        if ndim == 1:
                            power[i] = 0
                        else:
                            power[i] = [0, 0]
                        i = i + 1
                else:
                    # print("E", values[i], i)
                    # Unknown state. If previously we know about this
                    # appliance's state, we can
                    # use that. Else, it defaults to 0
                    if prev[appliance] == -1 or prev[appliance] == 0:
                        # print("F", values[i], i)
                        on = False
                        power[i] = 0
                        while values[i] != 1 and i < len(values) - 1:
                            # print("G", values[i], i)
                            if ndim == 1:
                                power[i] = 0
                            else:
                                power[i] = [0, 0]
                            i = i + 1
                    else:
                        # print("H", values[i], i)
                        on = True
                        power[i] = self.model.centroids[phase].ix[appliance].values
                        while values[i] != 0 and i < len(values) - 1:
                            # print("I", values[i], i)
                            power[i] = self.model.centroids[phase].ix[appliance].values
                            i = i + 1

            di[appliance] = power
            # print(power.sum())
        return di


    def disaggregate(self, mains, output_datastore = None, **load_kwargs):
        """Disaggregate mains according to the model learnt previously.

        Parameters
        ----------
        mains : nilmtk.ElecMeter or nilmtk.MeterGroup
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        sample_period : number, optional
            The desired sample period in seconds.
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        """
        model = self.model

        load_kwargs = self._pre_disaggregation_checks(mains, load_kwargs)
        load_kwargs.setdefault('sample_period', 60)
        load_kwargs.setdefault('sections', mains.good_sections())

        timeframes = []
        #building_path = '/building{}'.format(mains.building())
        #mains_data_location = building_path + '/elec/meter' + phase
        data_is_available = False

        transients = []
        steady_states = []
        for i in range(len(mains)):
            s, t = find_steady_states_transients_fast(
                mains.meters[i], model.params['cols'], noise_level=model.params['noise_level'], state_threshold= model.params['state_threshold'], **load_kwargs)
            steady_states.append(s.tz_localize('utc'))
            transients.append(t.tz_localize('utc'))

        transients.append(self.separate_simultaneous_events(transients))
        # For now ignoring the first transient
        # transients = transients[1:]

        # Initially all appliances/meters are in unknown state (denoted by -1)
        prev = [OrderedDict()] * len(transients)
        
        timeframes = []
        disaggregation_overall = [None] * len(transients)
        for phase in range(len(transients)):
            building_path = '/building{}'.format(mains.building()* 10 + phase)
            mains_data_location = building_path + '/elec/meter1'

            learnt_meters = model.centroids[phase].index.values
            for meter in learnt_meters:
                prev[phase][meter] = -1

            # Now iterating over mains data and disaggregating chunk by chunk
            to_load = mains.meters[i] if i < 3 else mains
            first = True
            for chunk in to_load.power_series(**load_kwargs):  # HIER MUSS ICH DOCH UEBER DIE EINZELNEN PHASEN LAUFEN!!!
                # Record metadata
                if phase == 1: # Only add once
                    timeframes.append(chunk.timeframe)
                    measurement = chunk.name # gehe davon aus, dass ueberall gleich
                power_df = self.disaggregate_chunk(
                    chunk, prev[phase], transients[phase], phase) # HAT DEN VORTEIL, DASS ICH MICH AUF DIE GUTEN SECTIONS KONZENTRIEREN KANN
                if first:
                    #power_df = pd.DataFrame(0, columns=power_df.columns, index = chunk.index[:1]).append(power_df) # Ich gehe davon aus, dass alle werte ankommen
                    power_df.iloc[0, :] = 0 
                    first = False

                cols = pd.MultiIndex.from_tuples([chunk.name])

                if output_datastore != None:
                    for meter in learnt_meters:
                        data_is_available = True
                        df = power_df[[meter]].dropna()
                        df.columns = cols
                        key = '{}/elec/meter{:d}'.format(building_path, meter + 2) # Weil 0 nicht gibt und Meter1 das undiaggregierte ist und 
                        output_datastore.append(key, df)
                    df = power_df[['rest']].dropna()
                    df.columns = cols
                    output_datastore.append(key=mains_data_location, value= df) #pd.DataFrame(chunk, columns=cols))  # Das Main wird auf Meter 1 gesetzt.
                else:
                    if disaggregation_overall is None:
                        disaggregation_overall = power_df
                    else:
                        disaggregation_overall = disaggregation_overall.append(power_df)
        if output_datastore != None:
            # Write a very last entry. Then at least start and end set
            df = pd.DataFrame(0., columns = cols, index = chunk.index[-1:])
            for phase in range(len(transients)):
                building_path = '/building{}'.format(mains.building()* 10 + phase)
                for meter in model.centroids[phase].index.values:
                    key = '{}/elec/meter{:d}'.format(building_path, meter + 2) # Weil 0 nicht gibt und Meter1 das undiaggregierte ist und 
                    output_datastore.append(key, df) 
            #output_datastore.append(key, df.rename(columns={0:meter})) Ich gehe davon aus, dass alle Daten rein kommen und ich rest nicht setzen muss

            # Then store the metadata
            num_meters = [len(cur) + 1 for cur in self.model.centroids] # Add one for the rest
            if data_is_available:
                self._save_metadata_for_disaggregation(
                    output_datastore=output_datastore,
                    sample_period=load_kwargs['sample_period'],
                    measurement=col,
                    timeframes=timeframes,
                    building=mains.building(),
                    supervised=False,
                    num_meters = num_meters,
                    original_building_meta = mains.meters[0].building_metadata
                )
        else:
            return disaggregation_overall

        # Der NEUE ANSATZ, den ich mir jetzt dovh verkneife
        #    for chunk in mains.meters[phase].power_series(**load_kwargs):
        #        # Record metadata
        #        timeframes[phase].append(chunk.timeframe)
        #        measurement = chunk.name
        #        power_df = self.disaggregate_chunk(
        #            chunk, prev[phase], transients[phase], phase)

        #        cols = pd.MultiIndex.from_tuples([chunk.name])
                
        #        # Die Meter muss ich noch durchiterieren
        #        id = str(datetime.now()).replace(" ", "").replace('-', '').replace(":","")
        #        id = id[:id.find('.')]
        #        keytemplate = '/building{0}/elec/disag/eventbased{1}/meter{2}'
        #        if output_datastore != None:
        #            for meter in learnt_meters:
        #                data_is_available = True
        #                df = power_df[[meter]].dropna() # remove the remaining nans
        #                df.columns = cols
        #                key = keytemplate.format(mains.building(), id, phase) + "/appliance{:d}".format(meter + 2) # Weil 0 nicht gibt und Meter1 der remaining powerflow ist
        #                output_datastore.append(key, df)
         
        #            # Store the remaining powerflow (the power not assigned to any facility) 
        #            df = power_df['rest'].dropna() # remove the remaining nans
        #            df.columns = cols
        #            key = keytemplate.format(mains.building(), id, phase) + "/appliance1"
        #            output_datastore.append(key, df)

        #            # Store the metadata
        #            if data_is_available:
        #                self._save_metadata_for_disaggregation(
        #                    output_datastore=output_datastore,
        #                    key = keytemplate.format(mains.building(), id, phase)
        #                )
                    
        #            #output_datastore.append(key=mains_data_location,
        #            #                    value=pd.DataFrame(chunk, columns=cols))  # Wir sparen uns den Main noch mal neu zu speichern
        #        else:
        #            if disaggregation_overall[phase] is None:
        #                disaggregation_overall[phase] = power_df
        #            else:
        #                disaggregation_overall[phase] = disaggregation_overall.append(power_df)

        #if output_datastore == None:
        #    return disaggregation_overall

    """
    def export_model(self, filename):
        model_copy = {}
        for appliance, appliance_states in self.model.iteritems():
            model_copy[
                "{}_{}".format(appliance.name, appliance.instance)] = appliance_states
        j = json.dumps(model_copy)
        with open(filename, 'w+') as f:
            f.write(j)

    def import_model(self, filename):
        with open(filename, 'r') as f:
            temp = json.loads(f.read())
        for appliance, centroids in temp.iteritems():
            appliance_name = appliance.split("_")[0].encode("ascii")
            appliance_instance = int(appliance.split("_")[1])
            appliance_name_instance = ApplianceID(
                appliance_name, appliance_instance)
            self.model[appliance_name_instance] = centroids
    """


    
    def _save_metadata_for_disaggregation_new_approach(self, output_datastore, key):
        """
        Also urpruenglich wollte ich das anders machen und eben auch die Metadatan mit abspeichern.
        Habe ich aus zeitgruenden dann gelassen und mache es doch so wie es vorher war.
        
        Add metadata for disaggregated appliance estimates to datastore.
        Is a custom version for more sophisticated storing.
        In the future we should introduce the model as a separate object and than
        that one can be used and serialized.

        This method returns nothing.  It sets the metadata
        in `output_datastore`.

        Note that `self.MODEL_NAME` needs to be set to a string before
        calling this method.  For example, we use `self.MODEL_NAME = 'CO'`
        for Combinatorial Optimisation.
        """
        
        # Meter0 wird das was auf allen Phasen ist
        output_datastore.save_metadata(key, self.model.params)

    #endregion