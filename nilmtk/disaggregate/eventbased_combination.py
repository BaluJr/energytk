from __future__ import print_function, division
from collections import OrderedDict, deque
import time
from datetime import datetime
from nilmtk.feature_detectors.cluster import hart85_means_shift_cluster
from nilmtk.feature_detectors.steady_states import find_steady_states_transients
from nilmtk.disaggregate.accelerators import find_steady_states_fast, find_transients_baranskistyle_fast, find_transients_fast, pair_fast, find_sections, myresample_fast, myviterbi_numpy_fast
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
from sklearn import mixture
import matplotlib as mpl
from scipy import linalg, spatial
import itertools
from nilmtk.clustering import CustomGaussianMixture
from sklearn.covariance import EmpiricalCovariance, MinCovDet, EllipticEnvelope

# Fix the seed for repeatability of experiments
SEED = 42
np.random.seed(SEED)


class MyIterativeClusterer():
    '''
    This is a helpclass, which encapsulates the process, that the clustering 
    happens in steps. This enables the possibility to increase the influence of power.
    '''
    pass


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
    ''' 
    Augment the transients by information about the segment 
    '''
    transients, states, threshold, noise_level = params
    
    # Add segment information
    indices = np.array(states.index)
    values = np.array(states.iloc[:,0])
    sections = find_sections((indices, values, noise_level))
    transients['segment'] = pd.Series(data = sections, index = transients.index).shift().bfill()
    transients = transients.join(transients.groupby("segment").size().rename('segmentsize'), on='segment')

    # Don't ask me, why error occured when I do this before the count
    if transients['ends'].dt.tz is None:
        transients['ends'] = transients['ends'].dt.tz_localize('utc')
    transients = transients.reset_index()

    # Additional features, which come in handy
    transients.loc[transients['active transition'] >= 0, 'spike'] = transients[transients['active transition'] >= 0]['signature'].apply(lambda sig: np.cumsum(sig).max())
    transients.loc[transients['active transition'] >= 0, 'spike'] -= transients.loc[transients['active transition'] >= 0, 'active transition']
    #transients.loc[transients['active transition'] >= 0, 'spike'] = transients.loc[transients['active transition'] >= 0, 'spike'].clip(lower=0)
    transients.loc[transients['active transition'] < 0, 'spike'] =  transients[transients['active transition'] < 0]['signature'].apply(lambda sig: np.cumsum(sig).min())
    transients.loc[transients['active transition'] < 0, 'spike'] -= transients.loc[transients['active transition'] < 0, 'active transition']
    #transients.loc[transients['active transition'] < 0, 'spike'] = transients.loc[transients['active transition'] < 0, 'spike'].clip(upper=0)
    transients['length'] = transients['ends'] - transients['starts']

    return transients



def add_segments(params):
    ''' 
    Augment the transients by information about the segment 
    '''
    transients, states, threshold, noise = params
    
    # Add segment information
    pd.eval('baseload_separators = (states<=(states.min() + @threshold))')
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
    pd.eval('transients["length"] = transients["ends"] - transients["starts"]')

    return transients

def fast_groupby(df, use_additional_grp_field = False):
    '''
    Does a fast groupby as used for creating the 3-size and 4-size events.
    '''
    df = df[['segment', 'active transition', 'starts', 'ends', 'spike']]#, 'length']]
    df = df.sort_values(['segment','starts'])
    df['duration'] = (df['starts'] - df.shift()['ends']).dt.seconds
    #df['length'] = df['length'].apply(lambda e: e.seconds)
    df.drop(columns=['starts', 'ends'], inplace=True)
    values = df.values
    keys, values = values[:,0], values[:,1:]
    ukeys,index=np.unique(keys, return_index = True) 
    result = []
    
    # Twoevents
    if (index[1] - index[0]) == 2:
        for arr in np.vsplit(values,index[1:]):
            result.append(np.array([np.sum(np.abs(arr[:,0]))/2, arr[0,0] + arr[1,0], np.clip(arr[0,1] - arr[0,0],0, None), np.clip(arr[1,1]-arr[1,0], None, 0), arr[1,2]])) # arr[:,2]]))#, arr[1:,3]]))
        cols = ['transition_avg', 'transition_delta', 'spike_up', 'spike_down', 'duration']
    # Threeevents
    elif (index[1] - index[0]) == 3:
        for arr in np.vsplit(values,index[1:]):
            result.append(np.concatenate([arr[:,0]]))#, arr[:,1], arr[:,2]]))#, arr[1:,3]]))
        cols = ['fst', 'sec', 'trd']#, 'fst_spike', 'sec_spike', 'trd_spike', ]  #, 'fst_length', 'sec_length', 'trd_length']#, 'fst_duration', 'sec_duration']
    # Fourevents
    else:
        for arr in np.vsplit(values,index[1:]):
            result.append(np.concatenate([arr[:,0], arr[:,1], arr[1:,2]])) # arr[:,2]]))#, arr[1:,3]]))
        cols = ['fst', 'sec', 'trd', 'fth', 'fst_spike', 'sec_spike', 'trd_spike', 'fth_spike', 'fst_duration', 'sec_duration', 'trd_duration'] #'fst_length', 'sec_length', 'trd_length', 'fth_length']#, 'fst_duration', 'sec_duration', 'trd_duration']
    df2=pd.DataFrame(index = ukeys, data = result, columns=cols)
    return df2


def fast_groupby_with_additional_grpfield(df):
    '''
    Does a fast groupby as used for creating the 3-size and 4-size events.
    '''
    df = df[['segment', 'grp', 'active transition', 'starts', 'ends', 'spike']]
    df = df.sort_values(['segment','grp', 'starts'])
    df['duration'] = (df['starts'] - df.shift()['ends']).dt.seconds
    values = df.drop(columns=['starts', 'ends']).values
    keys, values = values[:,:2], values[:,2:]
    ukeys,index=np.unique(keys[:,0] + keys[:,1].astype(str), return_index = True) 
    result = []
    
    # Twoevents    
    for arr in np.vsplit(values,index[1:]):
        result.append(np.array([np.sum(np.abs(arr[:,0]))/2, arr[0,0] + arr[1,0], arr[0,1] - arr[0,0], arr[1,1]-arr[1,0], arr[1,2]])) # arr[:,2]]))#, arr[1:,3]]))
    cols = ['transition_avg', 'transition_delta', 'spike_up', 'spike_down', 'duration']
    idx = pd.MultiIndex.from_arrays([df.loc[::2,'segment'],df.loc[::2,'grp'].values])
    df2 = pd.DataFrame(index = idx, data = result, columns=cols)
    #df2['grp'] = df.loc[::2,'grp'].values
    return df2


def myviterbi(segment, appliances):
    '''
    Own implementation of the viterbi algorithm, which is used to identify the 
    known appliances inside the powerflow.
    '''
    
    # For each event the mapping to which statemachine-step it may belong
    event_to_statechange = [[] for i in range(len(segment))]
    rows = segment[['active transition','spike']]
    for i, appliance in enumerate(appliances):
        for step in range(appliance['length']):
            valids = appliance[step].predict(rows.values) == 1   # -1 means outlier!!!
            for j, valid in enumerate(valids):
                if valid:
                    event_to_statechange[j].append((i, step))

    # Create the lookup for Matrices: T1=Distances, T2=Matches, T3=Path
    idx = pd.IndexSlice
    multindex = pd.MultiIndex(levels=[[] for _ in range(len(appliances))],
                            labels=[[] for _ in range(len(appliances))],
                            names=range(len(appliances)))
    state_table = pd.DataFrame(index = multindex, columns = ['T1', 'T2', 'T3', 'new'])
    startstate = tuple(np.zeros(len(appliances)))
    state_table.loc[startstate,:] = [0, 0, "", -1]
        
    # Find the best path
    t0 = time.time()
    for i, transient in segment.iterrows():
        print(str(i) + ': ' + str(time.time()-t0))
        i = transient['segplace']
        for appliance, step in event_to_statechange[i]:
            # Create retrieval for all availbale states in statestable (All before appliance slices)
            lookup = [slice(None) for _ in range(appliance)]
            lookup.append([step])
            
            # Check if these steps work out
            try: 
                rows = state_table[state_table['new'] != i].loc[tuple(lookup), :].iterrows()
            except KeyError:
                rows = []
            for curstate, outgoing_state in rows: 
                
                # Ich muss hier weiter machen indem ich prüfe warum so viele Events möglich sind! 
                # 1. Unnoetige Appliances entfernen
                # 2. Fehler beheben, dass gleicher State mehrfach appended, weil anstatt dem 'new' flag auch ein 'im step bereits geupdated' flag rein muesste.
                
                cost = appliances[appliance][step].mahalanobis([transient[['active transition','spike']]])[0]
                newstate = list(curstate)
                newstate[appliance] = (newstate[appliance] + 1) % appliances[appliance]['length']
                newstate = tuple(newstate) 

                # Calc the new values
                newcost = outgoing_state['T1'] + cost
                newmatches = outgoing_state['T2']
                if newstate[appliance] == 0:
                    newmatches += 1
                new_path = outgoing_state['T3'] + ";" + str(i) + "," + str(appliance)

                if (not (newstate in state_table.index)):     # not yet existing
                    state_table.loc[newstate,:] = [newcost, newmatches, new_path, i]                   
                elif(                                                                   
                        newmatches > state_table.loc[newstate,'T2'] or                                                  # more matches
                        ((newmatches == state_table.loc[newstate,'T2']) and (newcost < state_table.loc[newstate,'T1'])) # less cost
                    ):
                    state_table.loc[newstate,:3] = [newcost, newmatches, new_path]

    # The best path which ends in zero state is result
    T1, T2, T3, new = state_table.loc[startstate]
    labels = [-1] * len(segment)
    for cur in T3.split(";")[1:]:
        location, appliance = eval(cur)
        labels[location] = appliance
    return labels


    #for state in state_table.index:
    #    best_k = np.argmax(T1[:,i-1] * A[i,state] * B[j,y[y]])
    #    T2[state,i] = best_k
    #    T1[state,i] = T1[best_k,i-1] * A[i,state] * B[j,y[y]]
    # Go backwards and determine the best path
    #z_t = np.argmax(T1[:,-1])
    #for t in range(T,2):
    #    z_t = T2[z_t, t]
    #    result.append(s[z_t])
    #return reversed(X)


def myviterbi_numpy(segment, appliances):
    '''
    Own implementation of the viterbi algorithm, which is used to identify the 
    known appliances inside the powerflow.
    '''
    
    # For each event the mapping to which statemachine-step it may belong
    event_to_statechange = [[] for i in range(len(segment))]
    rows = segment[['active transition','spike']]
    for i, appliance in enumerate(appliances):
        for step in range(appliance['length']):
            valids = appliance[step].predict(rows.values) == 1   # -1 means outlier!!!
            for j, valid in enumerate(valids):
                if valid:
                    event_to_statechange[j].append((i, step))

    # Create the lookup for Matrices: T1=Distances, T2=Matches, T3=Path
    idx = pd.IndexSlice
    multindex = pd.MultiIndex(levels=[[] for _ in range(len(appliances))],
                            labels=[[] for _ in range(len(appliances))],
                            names=range(len(appliances)))
    state_table = pd.DataFrame(index = multindex, columns = ['T1', 'T2', 'T3'])
    startstate = tuple(np.zeros(len(appliances)))
    state_table.loc[startstate,:] = [0, 0, ""]
        
    # Find the best path
    t0 = time.time()
    for i, transient in segment.iterrows():
        #print(str(i) + ": " + str(time.time()-t0))
        i = transient['segplace']

        # Do the calculation for this step
        new_state_table = state_table.copy()
        for appliance, step in event_to_statechange[i]:
            # Create retrieval for all fitting available states in statestable
            lookup = [slice(None) for _ in range(appliance)]
            lookup.append([step])
            rows = state_table.loc[tuple(lookup),:]
            if len(rows) == 0:
                continue

            # Calculate the steps
            cost = spatial.distance.cdist([transient[['active transition','spike']]], np.expand_dims(appliances[appliance][step].location_, 0), 'mahalanobis', VI=linalg.inv(appliances[appliance][step].covariance_)).squeeze()
            #newappliancestate = (step+1 % appliances[appliance]['length'])
            #tst = lambda e, a, ast: tuple(e[0][:a] + (ast,) + e[0][a+1:])
            #newstate = np.apply_along_axis(tst, 0, rows.index.values, a = appliance, ast=newappliancestate)
            #newstate = np.array(*rows.index.values).reshape(-1, len(appliances))
            newstate = np.array([list(e) for e in rows.index.values])
            newstate[:,appliance] = (step+1) % (appliances[appliance]['length'])
            tmp  = list(map(tuple, newstate))
            newstate = np.zeros(len(newstate), dtype='object')
            newstate[:] = tmp 

            # Insert when not availbale yet
            new_introduced = pd.MultiIndex.from_tuples(newstate).difference(new_state_table.index)
            #new_state_table.loc[new_introduced,:] = [0, 1e100, ""] # Somehow bug in Pandas
            new_state_table = new_state_table.append(pd.DataFrame(data = {'T1':1e100, 'T2':0, 'T3':''}, index = new_introduced))

            # Update when better
            isnewmatch = (step == appliances[appliance]['length']-1)
            to_update = ((rows['T2'].values + isnewmatch > new_state_table.loc[newstate,'T2'].values) | ((rows['T2'].values == new_state_table.loc[newstate,'T2'].values) & ((rows['T1'].values + cost)< new_state_table.loc[newstate,'T1'].values))) # more matches or less cost
            to_update_in_new = newstate[to_update]
            new_state_table.loc[to_update_in_new,'T1'] = rows.loc[to_update,'T1'].values + cost
            new_state_table.loc[to_update_in_new,'T2'] = rows.loc[to_update,'T2'].values + isnewmatch
            new_state_table.loc[to_update_in_new,'T3'] = rows.loc[to_update,'T3'].values + ";" + str(i) + "," + str(appliance)
        state_table = new_state_table

    # The best path which ends in zero state is result
    T1, T2, T3 = state_table.loc[startstate]
    labels = [-1] * len(segment)
    for cur in T3.split(";")[1:]:
        location, appliance = eval(cur)
        labels[location] = appliance
    return labels




def remove_inconfident_elements(transients):
    '''
    Remove elements which are insecure
    '''
    shorten_segment = lambda seg: seg[:seg.rfind('_', 0, -2)]
    new_segments = transients[~transients['confident']]['segment'].apply(shorten_segment)
    transients[~transients['confident']]['new_segments'] = new_segments 
    transients.drop(columns=['segmentsize'], inplace = True)
    return transients.join(transients.groupby("segment").size().rename('segmentsize'), on='segment')

def find_appliances(params):
    ''' 
    Augment the transients by information about the appliances 
    '''
    transients, state_threshold = params
    clusterers = {}
    
    # Separate the two-flag events 
    twoevents = transients[transients['segmentsize'] == 2]
    twoevents = fast_groupby(twoevents)
    #twofunc = {'starts':min, 'ends':max, 'active transition': [min, max], 'spike': max}
    #twoevents = twoevents.groupby('segment').agg(twofunc
    #twoevents[('duration','max')] = (twoevents[('ends','max')]-twoevents[('starts','min')]).apply(lambda e: e.seconds)
    #twoevents[('transition_avg', 'avg')] = (twoevents[('active transition','max')].abs() + twoevents[('active transition','min')].abs())/2
    #twoevents[('transition_delta', 'avg')] = twoevents[('active transition','max')].abs() - twoevents[('active transition','min')].abs()
    #twoevents.drop([('active transition', 'min'), ('active transition', 'max'), ('ends','max'),('starts','min')], axis = 1, inplace=True)
    #twoevents.columns = twoevents.columns.droplevel(1)
    #twoevents = twoevents[['up_transition', 'spike']]
    clusterer, labels = _gmm_clustering(twoevents, max_num_clusters = 10, dim_scaling = {'transition_avg':3})
    clusterers['2'] = clusterer
    tmp = _gmm_confidence_check(twoevents, labels, clusterer, ['transition_avg'])
    twoevents['confident'] = tmp
    twoevents['appliance'] = labels
    transients = transients.join(twoevents[['appliance', 'confident']], on="segment")
    transients['appliance'] = transients['appliance'].fillna(-1)
    transients['confident'] = transients['confident'].fillna(0).astype(bool)
    #transients = remove_inconfident_elements(transients)

    ## Separate the three-flag events
    threeevents = transients[transients['segmentsize'] == 3]
    threeevents = fast_groupby(threeevents)
    threeevents['segsubtype'] = (threeevents['sec'] < 0).astype(int)
    threeevents['appliance'] = -1
    threeevents['confident'] = False
    for type in range(2):
        cur_threeevents = threeevents[threeevents['segsubtype'] == type].drop(columns=['segsubtype', 'appliance', 'confident'])
        clusterer, labels = _gmm_clustering(cur_threeevents , max_num_clusters = 10)
        clusterers['3_' + str(type)] = clusterer
        threeevents.loc[threeevents['segsubtype'] == type, 'confident'] = _gmm_confidence_check(cur_threeevents , labels, clusterer, ['fst', 'sec', 'trd'])
        threeevents.loc[threeevents['segsubtype'] == type, 'appliance'] = labels
    transients = transients.join(threeevents[['appliance','segsubtype', 'confident']], on="segment", rsuffix="three")
    transients.update(transients[['appliancethree', 'confidentthree', 'segsubtype']].rename(columns={'appliancethree':'appliance', 'confidentthree':'confident'}))
    transients.drop(["appliancethree", "confidentthree"],axis=1, inplace = True)
    transients['confident'] = transients['confident'].astype(bool) # whyever this is needed
    transients['segsubtype'] = transients['segsubtype'].fillna(0)
    #transients = remove_inconfident_elements(transients)

    ## Separate the four-flag events
    allfourevents = transients[transients['segmentsize']==4]

    ## First look, whether just overlapping or sequential twoevents
    fourevents = fast_groupby(allfourevents)
    overlapping = (fourevents['fst']>0) & (fourevents['sec']>0) & (fourevents['trd']<0) & (fourevents['fth']<0)  
    d1 = (fourevents['fst'] + fourevents['trd']).abs() + (fourevents['sec'] + fourevents['fth']).abs()
    d2 = (fourevents['fst'] + fourevents['fth']).abs() + (fourevents['sec'] + fourevents['trd']).abs()
    fourevents['overlap1'] = overlapping & (d1 <= d2)
    fourevents['overlap2'] = overlapping & (d2 < d1)
    fourevents['sequential'] = ((fourevents['fst'] + fourevents['sec']) < state_threshold)
    allfourevents = allfourevents.join(fourevents[['overlap1','overlap2','sequential']], on='segment')
            
    allflank = allfourevents[allfourevents['overlap1']].sort_values(['segment','starts']) # starts for easier debugging
    allflank['grp'] = np.array([0,1,0,1]*(len(allflank)//4)) + np.outer(range(len(allflank)//4),np.array([2,2,2,2])).flatten()
    flank = allfourevents[allfourevents['overlap2']].sort_values(['segment','starts'])
    flank['grp'] = np.array([0,1,1,0]*(len(flank)//4)) + np.outer(range(len(flank)//4),np.array([2,2,2,2])).flatten()
    allflank = allflank.append(flank)
    flank = allfourevents[allfourevents['sequential']].sort_values(['segment','starts'])
    flank['grp'] = np.array([0,0,1,1]*(len(flank)//4)) + np.outer(range(len(flank)//4),np.array([2,2,2,2])).flatten()
    allflank = allflank.append(flank)

    possible_twoevents = fast_groupby_with_additional_grpfield(allflank)
    labels =  clusterers['2'].predict(possible_twoevents)
    possible_twoevents['confident'] = _gmm_confidence_check(possible_twoevents, labels, clusterers['2'], ['transition_avg'])
    possible_twoevents['appliance'] = labels
    #possible_twoevents = possible_twoevents.set_index(['grp'], append=True) #.reset_index().set_index(['segment'])
    allflank = allflank.drop(['appliance','confident'],axis=1).join(possible_twoevents[['appliance', 'confident']], on=["segment",'grp'])
    # Only keep 4 events where both fitted to twoevent
    allflank = allflank.drop(columns=['confident']).join(allflank[['segment','confident']].groupby('segment').all(), on='segment')
    
    allflank.loc[allflank['confident'],'segmentsize'] = 2  # Really 2-events
    allflank.loc[~allflank['confident'],'grp'] = -1         # All non confident elements are one 4 group per segment 
    transients = transients.join(allflank[['appliance','segmentsize', 'grp']], rsuffix = '_new')
    transients.update(transients[['appliance_new']].rename(columns={'appliance_new':'appliance'}))
    transients.update(transients[['segmentsize_new']].rename(columns={'segmentsize_new':'segmentsize'}))
    transients.drop(["appliance_new", "segmentsize_new"], axis=1, inplace = True)
    transients['grp'] = transients['grp'].fillna(0)
    
    # All other events are fourevents
    fourevents = fourevents.join(allflank.groupby('segment').first()['confident']).fillna(False)
    fourevents['segsubtype'] = ((fourevents['sec'] > 0).astype(int) + (fourevents['trd'] > 0).astype(int)*2,0)[0]
    rest_fourevents = fourevents[~fourevents['confident'].astype(bool)] 
    for type in range(4):
        clusterer, labels = _gmm_clustering(rest_fourevents[rest_fourevents['segsubtype'] == type].dropna(axis=1), max_num_clusters = 10)
        clusterers['4_' + str(type)] = clusterer
        rest_fourevents.loc[rest_fourevents['segsubtype'] == type, 'confident'] = _gmm_confidence_check(rest_fourevents[rest_fourevents['segsubtype'] == type].dropna(axis=1), labels, clusterer, ['fst', 'sec', 'trd', 'fth'])
        rest_fourevents.loc[rest_fourevents['segsubtype'] == type, 'appliance'] = labels
    transients = transients.join(rest_fourevents[['appliance', 'segsubtype', 'confident']], on="segment", rsuffix="four")
    transients.update(transients[['appliancefour',  'segsubtypefour', 'confidentfour']].rename(columns={'appliancefour':'appliance', 'segsubtypefour':'segsubtype', 'confidentfour':'confident'}))
    transients.drop(["appliancefour", 'segsubtypefour', 'confidentfour'],axis=1, inplace = True)

    # Build identified appliances (outlier detections for the events of the appliances, and remove too small)
    transients['segplace'] = transients.groupby(['segment', 'grp']).cumcount()
    appliances = []
    for (length, subtype, appliance), rows in transients[(transients['segmentsize']<=4) & transients['confident']].groupby(['segmentsize', 'segsubtype', 'appliance']):        
        cur_appliance = {'length':int(length), 'appliance':appliance, 'subtype': subtype}
        if len(rows) // length < 5:
            transients[(transients['segmentsize'] == length) & (transients['segsubtype'] == subtype) & (transients['appliance'] == appliance), 'confident'] = False
        infostr = "__"
        for place, concrete_events in rows.groupby('segplace'):
            cur_appliance[place] = EllipticEnvelope().fit(concrete_events[['active transition','spike']])
            infostr += str(cur_appliance[place].location_[0]) + "_"
        cur_appliance['aaa'] = infostr + "_"
        appliances.append(cur_appliance)
    print("##################### " + str(len(appliances)))

    # Process longer sections Baranski Style
    all_segments = str(len(transients[transients['segmentsize'] > 4]['segment'].unique()))
    segi = 0
    t0 = time.time()
    for segmentid, segment in transients[transients['segmentsize'] > 4].groupby('segment'):
        print(str(segi) + "/" + all_segments + ": " + str(time.time()-t0))
        segi += 1
        labels = myviterbi_numpy_fast(segment[['active transition', 'spike']].values, appliances)

        # Translate labels and update transients
        reallabels = pd.DataFrame(columns=['segmentsize','segsubtype','appliance'], index = transients.loc[transients['segment'] == segmentid].index)
        for i, lbl in enumerate(labels):
            a = appliances[lbl]
            reallabels.iloc[i,:] = [a['length'], a['subtype'], a['appliance']]
        transients.loc[transients['segment'] == segmentid, ['segmentsize','segsubtype','appliance']] = reallabels


    # In the very last step, really only look onto remaining part
    #transients = pckl.load(open('remaining_transients.pckl', 'rb'))
    # Calculate the characteristics: Mean Power, StdDev, Cumulative, Length 
    #transients["confident"] = transients["confident"].astype(bool)
    #for segment, transients in transients[~transients["confident"]].groupby('segment'):
    #    num = len(transients)
    #    length = transients.iloc[-1]['ends'] - transients.iloc[1]['starts']
    #    cum = transients['active transition'].cumsum()
    #    deltas =(transients['starts'].shift(-1) - transients['starts']).dt.seconds
    #    energy = cum * deltas
    #    cum_energy = energy.cumsum()
    #    bins = pd.cut(cum_energy,4, retbins=True)[1]
    # Cluster segments after their criteria
    #clusterer, labels = _gmm_clustering(rest_fourevents[rest_fourevents['segsubtype'] == type].dropna(axis=1), max_num_clusters = 10)
    #_gmm_confidence_check(rest_fourevents[rest_fourevents['segsubtype'] == type].dropna(axis=1), labels, clusterer, ['fst', 'sec', 'trd', 'fth'])
    #remaining_transients = transients[(transients['segmentsize']>4) & (transients['appliance'] == -1)]
    #for seg, rows in remaining_tranients.groupby['segment']:
    #    pass    
    #cluster, labels, clusterer = self._cluster_events(cluster_ext, max_num_clusters=self.model.params['max_baranski_clusters'], all_columns = True)
    #cluster_ext['labels'] = labels
    #segment.join(cluster_ext['labels'], on="labels", rsuffix='_cluster')

    # Now check between segments whether same 
    #f = lambda a: set(a)
    #segmentsperappliance = groupby('appliance').agg({'segment':f})
    # Cluster the segmentevents 
    #cluster, label, clusterer = self._cluster_events(segmentevents)
    #    # Die Label anhaengen

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
    
    transients['segmentsize'] = transients['segmentsize'].astype(int)
    return {'transients':transients, 'clusterer':clusterers}



def create_appliances(params):
    ''' 
    Create the appliances after the labels have been added 
    '''

    transients, overall_powerflow, min_appearance = params
    appliances = []
    for (size, subtype, appliance), group in transients[transients['confident']].groupby(['segmentsize','segsubtype','appliance']):
        print(str(size) + "-" + str(appliance))
        if appliance == -1 or size == 1 or (len(group) / size) < 5:
            continue

        # Filter out overlapping events (Merge fitting and remove too short sections)
        group['switches'] = (group['segment'] != group.shift()['segment']).cumsum()
        group = group.join(group.groupby('switches').size().rename('switchsize'), on='switches')
        group = group[group['switchsize'] == group['segmentsize']] 
                
        # Correction of errors
        power = group.set_index('starts')['active transition'] # Sollte bereits sortiert sein
        error = power.groupby(np.outer(range(len(group)//size), np.ones(size)).flatten().astype(int)).sum()
        power.update(power[::size] - error.values.flatten())
        appliance = power.cumsum()

        # Resample the appliances to 5min (weighted mean)
        def myresample(d):
            if len(d) == 1:
                return d[0]
            weights = np.append(np.diff(d.index),np.timedelta64(5,'m') - (d.index[-1] - d.index[0]))
            weights = weights.astype('timedelta64[ms]').astype(int)
            return np.average(d, weights=weights)
        new_idx = pd.DatetimeIndex(start = overall_powerflow.index[0], end = overall_powerflow.index[-1], freq = '5min').tz_convert('utc')
        newSeries = pd.Series(index = new_idx)
        resampledload = appliance.append(newSeries).sort_index()
        resampledload = resampledload.ffill().bfill()
        appliance = pd.DataFrame(resampledload.resample('5min', how=myresample_fast), columns=overall_powerflow.columns)
        appliance = appliance.fillna(0)

        # Prepare output and subtract from overallpowerflow
        appliances.append(appliance)
        overall_powerflow  = pd.eval('overall_powerflow - appliance')

    return { 'appliances':appliances, 'overall_powerflow':overall_powerflow }



def _gmm_clustering(events, max_num_clusters=5, exact_cluster=None, dim_scaling = {}, dim_emph = {}):
    '''
    This is the clustering method used in the productive system

    Paramters:
    dim_scaling: Scaling of certain dimensions of the input. Won't affect GMM but the k-means initialization.
    dim_emph: Increase importance of certain dimensions during E-step of EM-algorithm. All other dimensions' 
                covariances are scaled by this value.
    '''
    
    # Special case:
    if len(events) < len(events.columns): 
        return None, (np.ones(len(events))*-1)
        #return np.array([events.iloc[0]["active transition"]]), np.array([0])

    # Preprocess dataframe
    mapper = DataFrameMapper([(column, None) for column in events.columns])
    clustering_input = mapper.fit_transform(events.copy())
    
    # Find scaling dimensions (makes a difference for kmeans-init)
    scaling_dims = list(dim_emph.keys())
    indices = np.in1d(events.columns, scaling_dims)
    clustering_input[:,indices] *= list(dim_scaling.values())
    # Translate dim_emph into the positions
    emph_dims = list(dim_scaling.keys())
    indices = np.in1d(events.columns, emph_dims)
    dim_emph = dict(zip(indices, list(dim_emph.values())))

    # Do exact clustering if demanded
    if not(exact_cluster is None):
        gmm = CustomGaussianMixture(n_components=exact_cluster, covariance_type='full', n_init = 10, dim_emph = dim_emph)
        gmm.fit(clustering_input)
        return gmm, gmm.predict(clustering_input)

    # Do the clustering
    best_gmm = CustomGaussianMixture(n_components=1, covariance_type='full', n_init = 5, dim_emph = dim_emph)
    best_gmm.fit(clustering_input)
    best_bic = best_gmm.bic(clustering_input)
    for n_clusters in range(2, max_num_clusters):   
        gmm = CustomGaussianMixture(n_components=n_clusters, covariance_type='full', n_init = 5, dim_emph = dim_emph)
        gmm.fit(clustering_input)
        cur_bic = gmm.bic(clustering_input)
        if cur_bic < best_bic:
            best_gmm = gmm #1 - color import matplotlib.cm as cm; ax.scatter(clustering_input[:2000][:,1],clustering_input[:2000][:,2],clustering_input[:2000][:,3], c = list((color*1000).astype(int)), cm = 'binary', s = 2)
            best_bic = cur_bic #np.repeat(color, 3, axis= 0).reshape(len(color),3)
            
    return best_gmm, best_gmm.predict(clustering_input)


    
def _gmm_check_for_membership(events, clusterer, threshold = 0.9):
    '''
    This function checks for certain elements, to which they belong
    '''
    if len(events) == 0:
        return pd.DataFrame(columns=['label'])

    # Prepare for checking
    mapper = DataFrameMapper([(column, None) for column in events.columns])
    clustering_input = mapper.fit_transform(events.copy())

    # Find appliance
    cur_probas = clusterer.predict_proba(clustering_input)
    appliance = np.argmax(cur_probas, axis = 1)
    appliance_probas = np.max(cur_probas, axis = 1)
    appliance[(appliance_probas < threshold)] = -1

    return appliance


def _gmm_confidence_check(X, prediction, clusterer, check_for_variance_dims):
    '''
    This function checks whether we can be sure about the assignment of an event.
    We have two checks we use:
    - 1. When the cluster has a too large StdDev 
    - 2. When the cluster is too small checks whether the points X, Y lie in the 
    - 3. If probability is lower than 80% 
    - 4. If the event is higher than 80% but lies outside the 3 sigma confidence intervall of the gmm distribution.
    Returns an array with the entries set to true, when element lies in cofidence interval.

    Paramters:
        -check_for_variance_dims: The dimensions which are looked at to check for variance.
    '''

    if (prediction == -1).all():
        return np.zeros(len(X)).astype(bool)

    # Take when probability > 90% and not outside 2 sigma interval or when inside the 1sigma interval
    print_ellipse = False
    print_ellipse = True
    filter = True
    filter = False
    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold', 'darkorange'])
    fig = plt.figure()
    confident = np.zeros(X.shape[0]).astype(bool)
    unique, counts = np.unique(prediction, return_counts=True)
    avg_clustersize = np.mean(counts)
    counts = dict(zip(unique, counts))
    for i, (mean, covar, color) in enumerate(zip(clusterer.means_, clusterer.covariances_, color_iter)):
        # 1. Exclude too small clusters!
        if (not i in counts) or (counts[i] < 0.1 * avg_clustersize) or (counts[i] < 5):
            continue

        # 2. Exclude clusters with too high stddev
        indices = np.where(np.in1d(X.columns, check_for_variance_dims))[0]
        stddevs = np.sqrt(covar.diagonal()[indices])
        means =  np.abs(mean[indices])
        if ((stddevs > 0.3 * means) & (stddevs > 10)).any() and not ((stddevs < 0.01 * means).any()):
            continue

        # Take points of current cluster
        cur = (prediction == i)
        cur_X = X[cur].values
        
        # 3. Check for confidence by the probability 
        probas = clusterer.predict_proba(cur_X)
        confident[cur] = probas.max(axis=1) > 0.9
        
        # 4. If not 90% sure, take at least the ones inside the one sigma environment
        confident[cur] |= (spatial.distance.cdist(cur_X, np.expand_dims(mean, 0), 'mahalanobis', VI=linalg.inv(covar)) < 1).squeeze()

        
        # Transform the points into the coordinate system of the covar ellipsoid
        #cur_X = cur_X - mean
        #v, w = np.linalg.eigh(covar)
        #std_dev = np.sqrt(v) # * np.sqrt(2)
        #transformed_points = cur_X.dot(w.T)
        
        #tst = spatial.distance.mahalanobis(cur_X, mean, linalg.inv(covar))
        #tst = spatial.distance.cdist(cur_X+mean, np.expand_dims(mean, 0), 'mahalanobis', VI=linalg.inv(covar))
        # If not 90% sure take at least the ones inside the one sigma environment
        #confident_tmp = (np.sum(transformed_points**2 / (1*std_dev)**2 , axis = 1) < 1)
        #for confidence_intervall in range(1.1,2.6,0.1):
        #    confident_tmp.append(np.sum(transformed_points**2 / (confidence_intervall*std_dev)**2 , axis = 1) < 1)
        #confident[cur] |= (np.sum(transformed_points**2 / (1*std_dev)**2 , axis = 1) < 1)
        #tst = spatial.distance.mahalanobis(cur_X, mean, linalg.inv(covar))

        # Plot an ellipse to show the Gaussian component
        #plt.scatter((cur_X+mean)[:,2], (cur_X+mean)[:, 3], 5, c=confident[cur])
        #if print_ellipse:
        #    v = 2. * np.sqrt(2.) * np.sqrt(v)      #Wurzeln, weil StdDev statt varianz, *2 da Diameter statt Radius, *sqrt(2)??
        #    #u = w[0] / linalg.norm(w[0])          Normalisiert ist es eigentlich schon!
        #    angle = np.arctan(w[0][1] / w[0][0])
        #    angle = 180. * angle / np.pi  # convert to degrees
        #    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color='gold')
        #    ell.set_clip_box(fig.bbox)
        #    ell.set_alpha(0.5)
        #    fig.axes[0].add_artist(ell)

    return confident.astype(bool)


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

    """ The relate my model """
    model_class = EventbasedCombinationDisaggregatorModel

    def Test_My_NILM(self):
        '''
        This function tests my NILM by using randomly generated powerflows.
        '''

        
        #[segment, appliances] = pckl.load(open('tst_myviterbi.pckl', 'rb'))
        #labels = myviterbi_numpy(segment, appliances)
        #labels = myviterbi(segment[['active transition', 'spike']].values, segment['segplace'].values, appliances)
        #appliances.loc[appliances['section'] == segmentid, "appliance"] = labels
        #kaputt()

        # Define what has to be simulated
        duration = 60*60*24*100
        # Each entry is Means,(transient, spike, duration), StdDevs
        # Pay attention: No cutting, results must be over event treshold
        specs =[[((2000, 20, 10), (20, 10, 4)), ((-2000, 10, 15), (10, 3, 4))],                # Heater 1
                [((1500, 30, 14), (10, 15, 4)), ((-1500, 10, 15), (10, 20, 4))],               # Heater 2
                [((130, 10, 90), (10, 5, 30)),  ((-130, 10, 600), (10, 6, 100))],              # Fridge
                [((300, 0, 60*60),(10, 5, 10)), ((-300, 0, 60*60*10),(10, 5, 10))],

                [((40, 0, 50), (6, 1, 10)),      ((120, 0, 40), (15, 1, 10)),    ((-160, 20, 200), (10, 1, 30))],
                [((100, 0, 40), (10, 5, 10)),     ((-26, 0, 180), (5, 1, 50)),    ((-74,5, 480), (15,1,50))]]
        # Breaks as appearances, break duration, stddev
        break_spec = [[4, 60, 10], [6, 10*60,10], [7, 10*60,10], [2, 60,10], [4, 60, 10], [2, 60, 10]]
        for i, bs in enumerate(break_spec): 
            bs[0] = bs[0]*len(specs[i])

        # Generate powerflow for each appliance
        appliances = []
        debug_num_events = []
        for i, spec in enumerate(specs):
            avg_event_duration = sum(map(lambda e: e[0][-1], spec)) + (break_spec[i][0]*60)
            num_events = duration // avg_event_duration
            debug_num_events.append(num_events)
            flags = []
            for flag_spec in spec:
                flags.append(np.random.normal(flag_spec[0], flag_spec[1], (num_events, 3)))
            flags = np.hstack(flags)
            cumsum = flags[:,:-3:3].cumsum(axis=1) # 2d vorrausgesetzt np.add.accumulate
            flags[:,:-3:3][cumsum < 5] += 5 - cumsum[cumsum < 5]
            flags[:,-3] = -flags[:,:-3:3].sum(axis=1) # 2d vorrausgesetzt
            flags = flags.reshape((-1,3))   

            # Put appliance to the input format
            appliance = pd.DataFrame(flags, columns=['active transition', 'spike', 'starts'])
            num_breaks = (len(appliance) // break_spec[i][0])-1
            breaks = np.random.normal(break_spec[i][1],break_spec[i][2], num_breaks)
            appliance.loc[break_spec[i][0]:num_breaks*break_spec[i][0]:break_spec[i][0],'starts'] += (breaks * 60)
            appliance.index = pd.DatetimeIndex(appliance['starts'].clip(5).cumsum()*1e9, tz='utc')
            appliance['ends'] = appliance.index + pd.Timedelta('6s')
            appliance.drop(columns=['starts'], inplace=True)
            appliance.loc[appliance['active transition'] < 0, 'signature'] = appliance['active transition'] - appliance['spike']
            appliance.loc[appliance['active transition'] >= 0, 'signature'] = appliance['active transition'] + appliance['spike']
            appliance['original_appliance'] = i
            appliances.append(appliance)
        transients = pd.concat(appliances, verify_integrity = True)
        transients = transients.sort_index()
        steady_states = transients[['active transition']].cumsum()#.rename({'active transition':'active average'})
        steady_states[['active transition']] += 60
        
        # Separate segments
        t1 = time.time()
        transients = add_segments_improved([transients, steady_states, self.model.params['state_threshold'], self.model.params['noise_level']])
        print('Segment separation: ' + str(time.time() - t1))

        # Create all events which per definition have to belong together
        t2 = time.time()
        transient, clusterer = find_appliances([transients, self.model.params['state_threshold']])
        print("Find appliances: " + str(time.time() - t2))

        # Create the appliances
        t3 = time.time()
        input_params, results = [], []
        for i in range(len(model.transients)):
            input_params.append((self.model.transients[i], self.model.overall_powerflow[i], self.model.params['min_appearance']))
        appliances, overall_powerflow = create_appliances([self.model.transients[i], self.model.overall_powerflow[i], self.model.params['min_appearance']])
        print("Put together appliance powerflows: " + str(time.time() - t3))
        



    def __init__(self, model = None):
        if model == None:
            model = self.model_class();
        self.model = model;
        super(EventbasedCombination, self).__init__()

    def _plot_clustering(self, events):
        ax = plt.axes(projection='3d'); 
        ax.scatter(tst[('active transition', 'avg')].values,tst[('duration', 'max')].values, tst[('spike', 'max')].values, s=0.1)
        events.plot.scatter(('active transition', 'avg'),('duration', 'log'), c=(events['color']), s=1, colormap='gist_rainbow')
        events.plot.scatter(('active transition', 'avg'),('duration', 'max'), c=events['color'], s=1)


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
        pool = Pool(processes=3)
        metergroup = metergroup.sitemeters() # Only the main elements are interesting
        model = self.model        
        model.steady_states = []
        model.transients = []
        model.appliances = []
        model.clusterer = [{}] * len(metergroup)
        model.overall_powerflow = overall_powerflow = [] 

                
        # 1. Load the events from the poweflow data
        print('Extract events')
        t1 = time.time()
        loader = []
        steady_states_list = []
        transients_list = []   
        #try:
        #    self.model = model = pckl.load(open('E:disaggregation_events/' + str(metergroup.identifier) + '.pckl', 'rb'))
        #    model.appliances = []
        #except:
        #    for i in range(len(metergroup)):
        #        overall_powerflow.append(None)
        #        steady_states_list.append([])
        #        transients_list.append([])
        #        loader.append(metergroup.meters[i].load(cols=self.model.params['cols'], chunksize = 31000000, **kwargs))
        #    try:
        #        while(True):
        #            input_params = []
        #            for i in range(len(metergroup)):
        #                power_dataframe = next(loader[i]).dropna()
        #                if overall_powerflow[i] is None:
        #                    overall_powerflow[i] = power_dataframe.resample('5min').agg('mean')  
        #                else:
        #                    overall_powerflow[i] = overall_powerflow[i].append(power_dataframe.resample('5min', how='mean'))
        #                indices = np.array(power_dataframe.index)
        #                values = np.array(power_dataframe.iloc[:,0])
        #                input_params.append((indices, values, model.params['min_n_samples'], model.params['state_threshold'], model.params['noise_level']))
        #            #states_and_transients, tst_original_fast, tst_original  = [], [], []
        #            #for i in range(2,len(metergroup)):
        #            #    states, transients = find_transients_fast(input_params[i])
        #            #    steady_states_list[i].append(states)
        #            #    transients_list[i].append(transients)
        #            states_and_transients = pool.map(find_transients_fast, input_params)
        #            for i in range(len(metergroup)):
        #                steady_states_list[i].append(states_and_transients[i][0])
        #                transients_list[i].append(states_and_transients[i][1])

        #    except StopIteration:
        #        pass
        #    # set model (timezone is lost within c programming)
        #    for i in range(len(metergroup)):
        #        model.steady_states.append(pd.concat(steady_states_list[i]).tz_localize('utc'))
        #        model.transients.append(pd.concat(transients_list[i]).tz_localize('utc')) #pd.merge(self.steady_states[0], self.steady_states[1], how='inner', suffixes=['_1', '_2'], indicator=True, left_index =True, right_index =True)
        #        model.transients[-1].index.rename("starts", inplace = True)
        #    pckl.dump(model, open('E:disaggregation_events/' + str(metergroup.identifier) + '.pckl', 'wb'))
        #t2 = time.time()
        #print("Eventloading: " + str(t2-t1))

        ## Create a fourth powerflow with events, common to all powerflows
        ## model.transients.append(self.separate_simultaneous_events(self.model.transients))
        
        ## Accelerate for testing
        #for i in range(len(model.transients)):
        #    self.model.transients[i] = self.model.transients[i][:len(self.model.transients[i])] # 3000
        #    self.model.steady_states[i] = self.model.steady_states[i][:len(self.model.steady_states[i])] # 3000
        #    self.model.overall_powerflow[i] = self.model.overall_powerflow[i][:self.model.steady_states[i].index[-1] + pd.Timedelta('5min')] # 80000

        ## 2. Separate segments between base load        
        #t1 = time.time()
        #input_params = []
        #for i in range(len(model.transients)):
        #    input_params.append((self.model.transients[i], self.model.steady_states[i], self.model.params['state_threshold'], self.model.params['noise_level']))
        #    #self.model.transients[i] = add_segments_improved(input_params[-1])
        #self.model.transients = pool.map(add_segments_improved, input_params)
        #print('Segment separation: ' + str(time.time() - t1))


        ## 3. Create all events which per definition have to belong together (tuned Hart)
        #t2 = time.time()
        #result = []
        #input_params = []
        #for i in range(len(model.transients)):
        #    input_params.append((model.transients[i], self.model.params['state_threshold']))
        #    #result.append(find_appliances(input_params[-1]))
        #result = pool.map(find_appliances, input_params)
        #for i in range(len(model.transients)):
        #    model.transients[i] = result[i]['transients']
        #    model.clusterer[i] = result[i]['clusterer']
        #print("Find appliances: " + str(time.time() - t2))
        #pckl.dump(model, open('E:disaggregation_events/' + str(metergroup.identifier) + '_appfound.pckl', 'wb'))


        ## 4. Create the appliances (Pay attention, id per size and subtype) and rest powerflow
        #t3 = time.time()
        #input_params, results = [], []
        #for i in range(len(model.transients)):
        #    input_params.append((self.model.transients[i], self.model.overall_powerflow[i], self.model.params['min_appearance']))
        #    results.append(create_appliances(input_params[-1]))
        ##results = pool.map(create_appliances, input_params)
        #for i in range(len(model.transients)):
        #    model.appliances.append(results[i]['appliances'])
        #    model.overall_powerflow[i] = results[i]['overall_powerflow']
        #print("Put together appliance powerflows: " + str(time.time() - t3))
        #pckl.dump(model, open('E:disaggregation_events/' + str(metergroup.identifier) + '_appcreated.pckl', 'wb'))

        # 5. Store the results (Not in parallel since writing to same file)
        self.model = model = pckl.load(open('E:disaggregation_events/' + str(metergroup.identifier) + '_appcreated.pckl', 'rb'))
        print('Store')
        t4 = time.time()
        for phase in range(len(model.transients)):
            building_path = '/building{}'.format(metergroup.building() * 10 + phase)
            for i, appliance in enumerate(self.model.appliances[phase]):
                key = '{}/elec/meter{:d}'.format(building_path, i + 2) # Weil 0 nicht gibt und Meter1 das undiaggregierte ist und 
                print(key)
                output_datastore.append(key, appliance) 
            output_datastore.append('{}/elec/meter{:d}'.format(building_path, 1), self.model.overall_powerflow[phase if phase < 3 else phase-1])
        print('Meta')
        num_meters = [len(cur) + 1 for cur in self.model.appliances] 
        self._save_metadata_for_disaggregation(
            output_datastore=output_datastore,
            sample_period = 300, #kwargs['sample_period'] if 'sample_period' in kwargs else 2,  Set to 5 minutes
            measurement=self.model.overall_powerflow[0].columns,
            timeframes=list(kwargs['sections']),
            building=metergroup.building(),
            supervised=False,
            num_meters=num_meters,
            original_building_meta=metergroup.meters[0].building_metadata
        )
        print("Stored: " + str(time.time()-t4))

    

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
        pq = 3
        meter_devices = {
            'disaggregate' : {
                'model': str(EventbasedCombinationDisaggregatorModel), #self.model.MODEL_NAME,
                'sample_period': 0,         # This makes it possible to use the special load functionality later
                'max_sample_period': sample_period,
                'measurements': [{
                    'physical_quantity': 'power', #measurement.levels[0][0],
                    'type': 'active' #measurement.levels, #[1][0]
                }]
            },
            'rest': {
                'model': 'rest',
                'sample_period': sample_period,
                'max_sample_period': sample_period,
                'measurements': [{
                    'physical_quantity': 'power', #measurement.levels, #[0][0],
                    'type': 'active' #measurement.levels, #[1][0]
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
            print(building_path)
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
            

    def _cluster_events(self, events, max_num_clusters=5, exact_num_clusters=None):
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
        mapper = DataFrameMapper([(column, None) for column in events.columns])
        clustering_input = mapper.fit_transform(events.copy())
    
        # Do the clustering
        best = -1
        labels = {}
        cluster_centers = {}
        labels_unique = {}
        clusterer = {}
        score = {}

        # If the exact number of clusters are specified, then use that
        if exact_num_clusters is not None:
            labels, centers = _apply_clustering_n_clusters(clustering_nput, exact_num_clusters, method)
            return centers.flatten()

        # Special case:
        if len(events) == 1: 
            return np.array([events.iloc[0]["active transition"]]), np.array([0])

        # If exact cluster number not specified, use cluster validity measures to find optimal number
        for n_clusters in range(3, max_num_clusters): #ACHTUNG AUF 3 gestellt
            try:
                # Do a clustering for each amount of clusters
                labels, centers, clusterer, score = self._apply_clustering_n_clusters(clustering_nput, n_clusters, method, score = True)
                labels[n_clusters] = labels
                cluster_centers[n_clusters] = centers
                labels_unique[n_clusters] = np.unique(labels)
                clusterer[n_clusters] = clusterer
                score[n_clusters] = score
                if score < score[best]:
                    best = n_clusters

            except Exception:
                if num_clusters > -1:
                    return cluster_centers[num_clusters]
                else:
                    return np.array([0])

        return pd.DataFrame(cluster_centers[num_clusters], columns=events.columns), labels[num_clusters], clusterer[num_clusters]
    
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
            sh_n = silhouette_score(events, labels[n_clusters], metric='euclidean', sample_size = min(len(labels[n_clusters]),5000))
            return k_means.labels_, k_means.cluster_centers_,k_means,shn
        elif method == 'gmm':
            gmm = mixture.GaussianMixture(n_components=n_clusters, covariance_type='full')
            gmm.fit(X)
            
            try:
                sh_n = silhouette_score(events, labels[n_clusters], metric='euclidean', sample_size = min(len(labels[n_clusters]),5000))
                if sh_n > silhouette:
                    silhouette = sh_n
                    num_clusters = n_clusters
            except Exception as inst:
                num_clusters = n_clusters
            return (gmm.means_, gmm.covariances_), gmm, gmm.bic
    


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