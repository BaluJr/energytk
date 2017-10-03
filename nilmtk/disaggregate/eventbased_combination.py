from __future__ import print_function, division
from collections import OrderedDict, deque
import pandas as pd
import time
from datetime import datetime
from nilmtk.feature_detectors.cluster import hart85_means_shift_cluster
from nilmtk.feature_detectors.steady_states import find_steady_states_transients
from nilmtk.disaggregate.accelerators import find_steady_states_transients_fast, pair_fast
from nilmtk.timeframe import merge_timeframes, TimeFrame

from nilmtk.disaggregate import SupervisedDisaggregator, UnsupervisedDisaggregatorModel
from functools import reduce, partial
from multiprocessing import Pool

# Hier will ich jerzt Cython benutzten. Wenn das geht will ich die Disaggregatore heute noch fertig bauen
# Erst Minimallast abziehen, Erst Hardt für alles was klar ist. -> Der Rest Baranski was klar ist -> Der Rest freiform

# Fix the seed for repeatability of experiments
SEED = 42
import numpy as np

np.random.seed(SEED)


class EventbasedCombinationDisaggregatorModel(UnsupervisedDisaggregatorModel):
            
    params = {
        #buffer_size: int, optional
        #    size of the buffer to use for finding edges
        # Only necessary for the old approach
        'buffer_size': 20, 

        'noise_level': 70, 
        
        'state_threshold': 15,

        #min_tolerance: int, optional
        #    variance in power draw allowed for pairing a match
        'min_tolerance': 100, 

        # percent_tolerance: float, optional
        # if transition is greater than large_transition,
        # then use percent of large_transition
        'percent_tolerance': 0.035,
        
        # large_transition: float, optional: power draw of a Large transition
        'large_transition': 1000,

        # cols: nilmtk.Measurement, should be one of the following
        #  [('power','active')]
        #  [('power','apparent')]
        #  [('power','reactive')]
        #  [('power','active'), ('power', 'reactive')]
        'cols': [('power','active')],

        # int
        #   The sample period, in seconds, used for both the
        #   mains and the disaggregated appliance estimates.
        'sample_period':120
        }
    
    def extendModels(otherModel):
        '''
        For unsupervised learning, the extend function for the models 
        is even more important as for the supervised/transfer case,
        because each data to disaggregate can be seen as new 
        training data.        
        '''
        pass


class MyDeque(deque):
    def popmiddle(self, pos):
        self.rotate(-pos)
        ret = self.popleft()
        self.rotate(pos)
        return ret 


class PairBuffer(object):
    """
    Attributes:
    * transitionList (list of tuples)
    * matchedPairs (dataframe containing matched pairs of transitions)
    """

    def __init__(self, buffer_size, min_tolerance, percent_tolerance,
                 large_transition, num_measurements):
        """
        Parameters
        ----------
        buffer_size: int, optional
            size of the buffer to use for finding edges
        min_tolerance: int, optional
            variance in power draw allowed for pairing a match
        percent_tolerance: float, optional
            if transition is greater than large_transition, then use percent of large_transition
        large_transition: float, optional
            power draw of a Large transition
        num_measurements: int, optional
            2 if only active power
            3 if both active and reactive power
        """
        # We use a deque here, because it allows us quick access to start and end popping
        # and additionally, we can set a maxlen which drops oldest items. This nicely
        # suits Hart's recomendation that the size should be tunable.
        self._buffer_size = buffer_size
        self._min_tol = min_tolerance
        self._percent_tol = percent_tolerance
        self._large_transition = large_transition
        self.transition_list = MyDeque([], maxlen=self._buffer_size)
        self._num_measurements = num_measurements
        if self._num_measurements == 3:
            # Both active and reactive power is available
            self.pair_columns = ['T1 Time', 'T1 Active', 'T1 Reactive',
                                 'T2 Time', 'T2 Active', 'T2 Reactive']
        elif self._num_measurements == 2:
            # Only active power is available
            self.pair_columns = ['T1 Time', 'T1 Active',
                                 'T2 Time', 'T2 Active']
        self.matched_pairs = pd.DataFrame(columns=self.pair_columns)

    def clean_buffer(self):
        # Remove any matched transactions
        for idx, entry in enumerate(self.transition_list):
            if entry[self._num_measurements]:
                self.transition_list.popmiddle(idx)
                self.clean_buffer()
                break
                # Remove oldest transaction if buffer cleaning didn't remove anything
                # if len(self.transitionList) == self._bufferSize:
                #    self.transitionList.popleft()

    def add_transition(self, transition):
        # Check transition is as expected.
        assert isinstance(transition, (tuple, list))
        # Check that we have both active and reactive powers.
        assert len(transition) == self._num_measurements
        # Convert as appropriate
        if isinstance(transition, tuple):
            mtransition = list(transition)
        # Add transition to List of transitions (set marker as unpaired)
        mtransition.append(False)
        self.transition_list.append(mtransition)
        # checking for pairs
        # self.pairTransitions()
        # self.cleanBuffer()

    def pair_transitions(self):
        """
        Hart 85, P 33.
        When searching the working buffer for pairs, the order in which 
        entries are examined is very important. If an Appliance has 
        on and off several times in succession, there can be many 
        pairings between entries in the buffer. The algorithm must not
        allow an 0N transition to match an OFF which occurred at the end 
        of a different cycle, so that only ON/OFF pairs which truly belong 
        together are paired up. Otherwise the energy consumption of the 
        appliance will be greatly overestimated. The most straightforward 
        search procedures can make errors of this nature when faced with 
        types of transition sequences.

        Hart 85, P 32.
        For the two-state load monitor, a pair is defined as two entries
        which meet the following four conditions:
        (1) They are on the same leg, or are both 240 V,
        (2) They are both unmarked, 
        (3) The earlier has a positive real power component, and 
        (4) When added together, they result in a vector in which the 
        absolute value of the real power component is less than 35 
        Watts (or 3.5% of the real power, if the transitions are 
        over 1000 W) and the absolute value of the reactive power 
        component is less than 35 VAR (or 3.5%).

        ... the correct way to search the buffer is to start by checking 
        elements which are close together in the buffer, and gradually 
        increase the distance. First, adjacent  elements are checked for 
        pairs which meet all four requirements above; if any are found 
        they are processed and marked. Then elements two entries apart 
        are checked, then three, and so on, until the first and last 
        element are checked...

        """

        tlength = len(self.transition_list)
        pairmatched = False
        if tlength < 2:
            return pairmatched

        # Can we reduce the running time of this algorithm?
        # My gut feeling is no, because we can't re-order the list...
        # I wonder if we sort but then check the time... maybe. TO DO
        # (perhaps!).

        # Start the element distance at 1, go up to current length of buffer
        for eDistance in range(1, tlength):
            idx = 0
            while idx < tlength - 1:
                # We don't want to go beyond length of array
                compindex = idx + eDistance
                if compindex < tlength:
                    val = self.transition_list[idx]
                    # val[1] is the active power and
                    # val[self._num_measurements] is match status
                    if (val[1] > 0) and (val[self._num_measurements] is False):
                        compval = self.transition_list[compindex]
                        if compval[self._num_measurements] is False:
                            # Add the two elements for comparison
                            vsum = np.add(
                                val[1:self._num_measurements],
                                compval[1:self._num_measurements])
                            # Set the allowable tolerance for reactive and
                            # active
                            matchtols = [self._min_tol, self._min_tol]
                            for ix in range(1, self._num_measurements):
                                matchtols[ix - 1] = self._min_tol if (max(np.fabs([val[ix], compval[ix]]))
                                                                      < self._large_transition) else (self._percent_tol
                                                                                                      * max(
                                    np.fabs([val[ix], compval[ix]])))
                            if self._num_measurements == 3:
                                condition = (np.fabs(vsum[0]) < matchtols[0]) and (
                                    np.fabs(vsum[1]) < matchtols[1])

                            elif self._num_measurements == 2:
                                condition = np.fabs(vsum[0]) < matchtols[0]

                            if condition:
                                # Mark the transition as complete
                                self.transition_list[idx][
                                    self._num_measurements] = True
                                self.transition_list[compindex][
                                    self._num_measurements] = True
                                pairmatched = True

                                # Append the OFF transition to the ON. Add to
                                # dataframe.
                                matchedpair = val[0:self._num_measurements] + compval[0:self._num_measurements]
                                self.matched_pairs.loc[len(self.matched_pairs),:] = matchedpair

                    # Iterate Index
                    idx += 1
                else:
                    break

        return pairmatched

    
class FastPairBuffer(object):
    """
    This is an improved version of the pairing. It limits the 
    reach of the buffer by the minimum powerflow and not by the 
    pure distance.
    This is also the function which will be later made faster 
    by using cython.



    Attributes:
    Improved pairing function
    * transitionList (list of tuples)
    * matchedPairs (dataframe containing matched pairs of transitions)
    """

    def __init__(self, buffer_size, min_tolerance, percent_tolerance,
                 large_transition, num_measurements):
        """
        Parameters
        ----------
        buffer_size: int, optional
            size of the buffer to use for finding edges
        min_tolerance: int, optional
            variance in power draw allowed for pairing a match
        percent_tolerance: float, optional
            if transition is greater than large_transition, then use percent of large_transition
        large_transition: float, optional
            power draw of a Large transition
        num_measurements: int, optional
            2 if only active power
            3 if both active and reactive power
        """
        # We use a deque here, because it allows us quick access to start and end popping
        # and additionally, we can set a maxlen which drops oldest items. This nicely
        # suits Hart's recommendation that the size should be tunable.
        self._buffer_size = buffer_size
        self._min_tol = min_tolerance
        self._percent_tol = percent_tolerance
        self._large_transition = large_transition
        self.transition_list = MyDeque([], maxlen=self._buffer_size)
        self._num_measurements = num_measurements
        if self._num_measurements == 3:
            # Both active and reactive power is available
            self.pair_columns = ['T1 Time', 'T1 Active', 'T1 Reactive',
                                 'T2 Time', 'T2 Active', 'T2 Reactive']
        elif self._num_measurements == 2:
            # Only active power is available
            self.pair_columns = ['T1 Time', 'T1 Active',
                                 'T2 Time', 'T2 Active']
        self.matched_pairs = pd.DataFrame(columns=self.pair_columns)

    def clean_buffer(self):
        # Remove any matched transactions
        for idx, entry in enumerate(self.transition_list):
            if entry[self._num_measurements]:
                self.transition_list.popmiddle(idx)
                self.clean_buffer()
                break
                # Remove oldest transaction if buffer cleaning didn't remove anything
                # if len(self.transitionList) == self._bufferSize:
                #    self.transitionList.popleft()

    def add_transition(self, transition):
        # Check transition is as expected.
        assert isinstance(transition, (tuple, list))
        # Check that we have both active and reactive powers.
        assert len(transition) == self._num_measurements
        # Convert as appropriate
        if isinstance(transition, tuple):
            mtransition = list(transition)
        # Add transition to List of transitions (set marker as unpaired)
        mtransition.append(False)
        self.transition_list.append(mtransition)
        # checking for pairs
        # self.pairTransitions()
        # self.cleanBuffer()


    def pair_transitions_fast(self):
       
        self.transition_list = MyDeque([], maxlen=self._buffer_size)
        self.fast_trans_list = np.zeros(self._buffer_size)   
        # No pair when only one element contained
        tlength = len(self.transition_list)
        pairmatched = False
        if tlength < 2:
            return pairmatched

        # Create an array from the availa
        numpy.roll


        # Start the element distance at 1, go up to current length of buffer
        for eDistance in range(1, tlength):
            idx = 0
            while idx < tlength - 1:
                # We don't want to go beyond length of array
                compindex = idx + eDistance
                if compindex < tlength:
                    val = self.transition_list[idx]
                    # val[1] is the active power and
                    # val[self._num_measurements] is match status
                    if (val[1] > 0) and (val[self._num_measurements] is False):
                        compval = self.transition_list[compindex]
                        if compval[self._num_measurements] is False:
                            # Add the two elements for comparison
                            vsum = np.add(
                                val[1:self._num_measurements],
                                compval[1:self._num_measurements])
                            # Set the allowable tolerance for reactive and
                            # active
                            matchtols = [self._min_tol, self._min_tol]
                            for ix in range(1, self._num_measurements):
                                matchtols[ix - 1] = self._min_tol if (max(np.fabs([val[ix], compval[ix]]))
                                                                      < self._large_transition) else (self._percent_tol
                                                                                                      * max(
                                    np.fabs([val[ix], compval[ix]])))
                            if self._num_measurements == 3:
                                condition = (np.fabs(vsum[0]) < matchtols[0]) and (
                                    np.fabs(vsum[1]) < matchtols[1])

                            elif self._num_measurements == 2:
                                condition = np.fabs(vsum[0]) < matchtols[0]

                            if condition:
                                # Mark the transition as complete
                                self.transition_list[idx][
                                    self._num_measurements] = True
                                self.transition_list[compindex][
                                    self._num_measurements] = True
                                pairmatched = True

                                # Append the OFF transition to the ON. Add to
                                # dataframe.
                                matchedpair = val[0:self._num_measurements] + compval[0:self._num_measurements]
                                self.matched_pairs.loc[len(self.matched_pairs),:] = matchedpair

                    # Iterate Index
                    idx += 1
                else:
                    break

        return pairmatched


class EventbasedCombination(SupervisedDisaggregator):
    """ This is the final used disaggregator, which bases on the the 
    combination of the available event based disaggregators. In our case
    Baranski and Hardt.

    It works a little bit different concerning how results are stored. As 
    it only disaggregates state machines, it only stores the flanks and all
    other parts have to be reconstructed by using ffill.
    That reduces load times significantly. 
    (This approach makes accumulation in next step also much faster,
    sothat only the final results have to be in memory. Top!)

    Attributes
    ----------
    model : dict
        Each key is either the instance integer for an ElecMeter,
        or a tuple of instances for a MeterGroup.
        Each value is a sorted list of power in different states.
    """
    Requirements = {
        'max_sample_period': 10,
        'physical_quantities': [['power','active']]
    }

    # The related model
    model_class = EventbasedCombinationDisaggregatorModel


    def __init__(self, model = None):
        if model == None:
            model = self.model_class();
        self.model = model;
        super(EventbasedCombination, self).__init__()


    def train(self, metergroup, **kwargs):
        """
        Train using Hart85. Places the learnt model in `model` attribute.

        Parameters
        ----------
        metergroup : a nilmtk.MeterGroup object
        """
        
        metergroup = metergroup.sitemeters() # Only the main elements are interesting
        
        # Find the events
        model = self.model
        model.steady_states = []
        model.transients = []
        for i in range(len(metergroup)):
            #t1 = time.time()
            #steady_states, transients = find_steady_states_transients(
            #    metergroup.meters[i], cols, noise_level, state_threshold, **kwargs)
            #t2 = time.time()
            cols=self.model.params['cols']
            steady_states, transients = find_steady_states_transients_fast(
                metergroup.meters[i], cols, model.params['noise_level'], model.params['state_threshold'], **kwargs)
            #t3 = time.time()
            # somehow the timezone is lost within c programming
            model.steady_states.append(steady_states.tz_localize('utc'))
            model.transients.append(transients.tz_localize('utc')) #pd.merge(self.steady_states[0], self.steady_states[1], how='inner', suffixes=['_1', '_2'], indicator=True, left_index =True, right_index =True)
        
        # Create a third powerflow with events, common to all powerflows
        model.transients.append(self.separate_simultaneous_events(self.model.transients))

        # Do the pairing in parallel for the four powerflows
        pool = Pool(processes=4)
        #t1 = time.time()
        #self.pair_df = pool.map(functools.partial(self.pair, 
        #           buffer_size = buffer_size,     
        #           min_tolerance = min_tolerance, 
        #           percent_tolerance = percent_tolerance,
        #           large_transition = large_transition), self.transients)
        t2 = time.time()
        
        # Manually create the paramters because not working with partial somehow
        input_params = []
        for cur in model.transients:
            input_params.append((cur, model.params['min_tolerance'], model.params['percent_tolerance'], model.params['large_transition']))
        model.pair_df_new = pool.map(pair_fast, input_params)
        t3 = time.time()

        # Do the clustering
        #tx1 = []
        tx2 = []
        model.centroids = []
        for i in range(len(model.transients)):


            tx2.append(time.time())
            model.centroids.append(hart85_means_shift_cluster(model.pair_df_new[i], model.params['cols']))
        t4 = time.time()
                        






    def pair(self, transients, buffer_size, min_tolerance, percent_tolerance,
             large_transition): 
        #subset = list(self.transients.itertuples())
        buffer = PairBuffer(
            min_tolerance=min_tolerance, buffer_size=buffer_size,
            percent_tolerance=percent_tolerance,
            large_transition=large_transition,
            num_measurements=len(transients.columns) + 1)
        for s in transients.itertuples():
            # if len(buffer.transitionList) < bsize
            if len(buffer.transition_list) == buffer_size:
                buffer.clean_buffer()
            buffer.add_transition(s)
            buffer.pair_transitions()
        return buffer.matched_pairs

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
        
        # AHHHRG! Hier muss ich irgendwie vorne und hinten eine 0 ergaenzen, damit auf jeden Fall etwas im Storage landet
        # UND DANN HABE ICH ES!

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
                    measurement=measurement,
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
                'sample_period': sample_period,
                'max_sample_period': sample_period,
                'measurements': [{
                    'physical_quantity': measurement[0],
                    'type': measurement[1]
                }]
            },
            'rest': {
                'model': 'rest',
                'sample_period': sample_period,
                'max_sample_period': sample_period,
                'measurements': [{
                    'physical_quantity': measurement[0],
                    'type': measurement[1]
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
        for i in range(4):
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
                'geo_location': original_building_meta['geo_location'],
                'zip': original_building_meta['zip'],
            }

            output_datastore.save_metadata(building_path, building_metadata)

    

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


    def calc_tolerance(self, value, match_target):
        if (abs(value - match_target)) < self.model.params['large_transition']:
            matchtol = self.model.params['min_tolerance']
        else: 
            matchtol = self.model.params['percent_tolerance'] * max(np.fabs([value, match_target]))
        return matchtol
