import pandas as pd
DEBUG = False
import time 
import sys
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from cython.parallel import parallel, prange

from ..timeframe import TimeFrame
from numpy import diff, concatenate
from ..utils import timedelta64_to_secs
import gc
from nilmtk.elecmeter import ElecMeter

# This is my first test for accelerating using c
def primes(int kmax):
    cdef int n, k, i
    cdef int p[1000]
    result = []
    if kmax > 1000:
        kmax = 1000
    k = 0
    n = 2
    while k < kmax:
        i = 0
        while i < k and n % p[i] != 0:
            i = i + 1
        if i == k:
            p[k] = n
            k = k + 1
            result.append(n)
        n = n + 1
    return result



def find_steady_states_fast(np.ndarray indices, np.ndarray values, int min_n_samples=2, int state_threshold=15,
                       int noise_level=70):
    """Finds steady states given a DataFrame of power with only
    a single column.

    Parameters
    ----------
    dataframe: pd.DataFrame with DateTimeIndex
    min_n_samples(int): number of samples to consider constituting a
        steady state.
    stateThreshold: maximum difference between highest and lowest
        value in steady state.
    noise_level: the level used to define significant
        appliances, transitions below this level will be ignored.
        See Hart 1985. p27.

    Returns
    -------
    steady_states, transitions
    """
    # Tells whether we have both real and reactive power or only real power
    #cdef int num_measurements = len(dataframe.columns)
    cdef float estimated_steady_power# = np.array([0] * num_measurements)
    cdef float last_steady_power# = np.array([0] * num_measurements)
    cdef float previous_measurement#= np.array([0] * num_measurements)
    cdef float last_transition

    # These flags store state of power

    instantaneous_change = False  # power changing this second
    ongoing_change = False  # power change in progress over multiple seconds

    index_transitions = []  # Indices to use in returned Dataframe
    index_steady_states = []
    transitions = []  # holds information on transitions
    steady_states = []  # steadyStates to store in returned Dataframe
    cdef int N = 0  # N stores the number of samples in state
    #curTime = dataframe.iloc[0].name  # first state starts at beginning

    # Iterate over the rows performing algorithm
    print ("Finding Edges, please wait ...")
    sys.stdout.flush()

    #oldTime = time.clock()
    #i = 0
    cdef size_t i  
    for i in range(len(indices)):#row in dataframe.itertuples():
        #print(row)
        #i += 1
        #if DEBUG and i == 100000:
        #    tmpTime = time.clock()
        #    print(str(tmpTime - oldTime) + " seconds for 100000 steps")
        #    oldTime = tmpTime
        #    i = 0

        # test if either active or reactive moved more than threshold
        # http://stackoverflow.com/questions/17418108/elegant-way-to-perform-tuple-arithmetic
        # http://stackoverflow.com/questions/13168943/expression-for-elements-greater-than-x-and-less-than-y-in-python-all-in-one-ret

        # Step 2: this does the threshold test and then we sum the boolean
        # array.
        this_measurement = values[i] #row[1]#:3]

        # logging.debug('The current measurement is: %s' % (thisMeasurement,))
        # logging.debug('The previous measurement is: %s' %
        # (previousMeasurement,))

        # Elementwise absolute differences
        state_change = abs(this_measurement - previous_measurement)
        #if i < 10:
        #    print('i: ' + str(state_change))
        #np.fabs(np.subtract(this_measurement, previous_measurement))
        # logging.debug('The State Change is: %s' % (stateChange,))

        if state_change > state_threshold: #np.sum(state_change > state_threshold):
            instantaneous_change = True
        else:
            instantaneous_change = False

        # Step 3: Identify if transition is just starting, if so, process it
        if instantaneous_change and (not ongoing_change):

            # Calculate transition size
            last_transition = estimated_steady_power - last_steady_power #np.subtract(estimated_steady_power, last_steady_power)
            # logging.debug('The steady state transition is: %s' %
            # (lastTransition,))

            # Sum Boolean array to verify if transition is above noise level
            if abs(last_transition) > noise_level: #np.sum(np.fabs(last_transition) > noise_level):
                # 3A, C: if so add the index of the transition start and the
                # power information

                # Avoid outputting first transition from zero
                index_transitions.append(curTime)
                # logging.debug('The current row curTime is: %s' % (curTime))
                transitions.append(last_transition)

                # I think we want this, though not specifically in Hart's algo notes
                # We don't want to append a steady state if it's less than min samples in length.
                # if N > min_n_samples:
                index_steady_states.append(curTime)
                # logging.debug('The ''curTime'' stored is: %s' % (curTime))
                # last states steady power
                steady_states.append(estimated_steady_power)
                
            # 3B
            last_steady_power = estimated_steady_power
            # 3C
            curTime = indices[i] #row[0]

        # Step 4: if a new steady state is starting, zero counter
        if instantaneous_change:
            N = 0

        # Hart step 5: update our estimate for steady state's energy
        estimated_steady_power = (N * estimated_steady_power + this_measurement) / (N + 1)
        #estimated_steady_power = np.divide(
        #    np.add(np.multiply(N, estimated_steady_power),
        #           this_measurement), (N + 1))
        # logging.debug('The steady power estimate is: %s' %
        #    (estimatedSteadyPower,))
        # Step 6: increment counter
        N += 1

        # Step 7
        ongoing_change = instantaneous_change

        # Step 8
        previous_measurement = this_measurement

    # Appending last edge
    last_transition = np.subtract(estimated_steady_power, last_steady_power)
    if np.sum(np.fabs(last_transition) > noise_level):
        index_transitions.append(curTime)
        transitions.append(last_transition)
        index_steady_states.append(curTime)
        steady_states.append(estimated_steady_power)

    # Removing first edge if the starting steady state power is more
    # than the noise threshold
    #  https://github.com/nilmtk/nilmtk/issues/400

    if len(steady_states)>0 and np.sum(steady_states[0] > noise_level) and index_transitions[0] == index_steady_states[0]:# == dataframe.iloc[0].name:
        transitions = transitions[1:]
        index_transitions = index_transitions[1:]
        steady_states = steady_states[1:]
        index_steady_states = index_steady_states[1:]

    print("Edge detection complete.")

    print("Creating transition frame ...")
    sys.stdout.flush()

    #cols_transition = {1: ['active transition'],
    #                   2: ['active transition', 'reactive transition']}

    #cols_steady = {1: ['active average'],
    #               2: ['active average', 'reactive average']}

    
    #if len(index_transitions) == 0:
    #    # No events
    #    return pd.DataFrame(), pd.DataFrame()
    #else:
    transitions = pd.DataFrame(data=transitions, index=index_transitions,
                                columns=['active transition']) #cols_transition[num_measurements])
    print("Transition frame created.")

    print("Creating states frame ...")
    sys.stdout.flush()
    
    steady_states = pd.DataFrame(data=steady_states, index=index_steady_states,
                                    columns=['active average']) #cols_steady[num_measurements])
    print("States frame created.")
    print("Finished.")
    return steady_states, transitions



def find_steady_states_transients_fast(metergroup, cols, int noise_level,
                                  int state_threshold, **load_kwargs):
    """
    Creates the states transient list over multiple meters in a 
    meter group.

    Returns
    -------
    steady_states, transients : pd.DataFrame
    """
    steady_states_list = []
    transients_list = []
    
    cdef np.ndarray indices
    cdef np.ndarray values 
    for power_df in metergroup.load(cols=cols, **load_kwargs): # Load brings it also to a single powerline
        """
        if len(power_df.columns) <= 2:
            # Use whatever is available
            power_dataframe = power_df
        else:
            # Active, reactive and apparent are available
            power_dataframe = power_df[[('power', 'active'), ('power', 'reactive')]]
        """
        power_dataframe = power_df.dropna()
        indices = np.array(power_dataframe.index)
        values = np.array(power_dataframe.iloc[:,0])

        x, y = find_steady_states_fast(indices, values, #power_dataframe, 
            noise_level=noise_level,
            state_threshold=state_threshold)
        steady_states_list.append(x)
        transients_list.append(y)
    return [pd.concat(steady_states_list), pd.concat(transients_list)]



def pair_fast(params):
    '''
    This is the dramatically accelerated version of the pairing.
    '''
    transitions, min_tolerance, percent_tolerance, large_transition = params
    if transitions.size < 2:
        return pd.DataFrame(columns= ['T1 Time', 'T1 Active', 'T2 Time', 'T2 Active'])
       
    if len(transitions.columns) == 1:
        indices = np.array(transitions.index)
        values = np.array(transitions.iloc[:,0])
        return pair_fast_inner_activeonly(indices, values, min_tolerance, percent_tolerance, large_transition)
    else:
        raise NotImplementedError("So far only active power optimized.")


def pair_fast_inner_activeonly(np.ndarray index, np.ndarray values, float min_tolerance, float percent_tolerance, float large_transition):
    '''
    index brauche ich ggf gar nicht

    This is the dramatically accelerated version of the pairing.
    Here the values are passed as an array to accelerate the process dramatically.
    '''
    # Create variables used later
    matched_pairs = pd.DataFrame(columns= ['T1 Time', 'T1 Active', 'T2 Time', 'T2 Active'])
    cdef size_t i  
    cdef size_t j 
    cdef float matchtol

    # Create a cur_power array
    cdef bint *matched = <bint *>malloc(len(values) * sizeof(bint))
    cdef float *cur_power = <float *>malloc(len(values) * sizeof(float))
    cdef float prev_power = 0
    for i in range(len(values)-1, -1, -1):
        cur_power[i] = prev_power + values[i]
        prev_power = cur_power[i]
        matched[i] = False

    # Go from end to beginning
    for i in range(len(values)-1, -1, -1):
        #print("i:{0} = {1}".format(i, values[i]))
        if (values[i] < 0):
            for j in range(i-1, -1, -1):
                #print("j:{0} = {1}".format(j, values[j]))

                if matched[j] or values[j] < 0:
                    # Only positive switches may fit
                    #print("CONTINUE")
                    continue

                # Add the two elements for comparison
                if (abs(values[i] - values[j])) < large_transition:
                    matchtol = min_tolerance
                else: 
                    matchtol = percent_tolerance * max(np.fabs([values[i], values[j]]))
              
                # Check whether a valid pair
                if abs(values[i] + values[j]) < matchtol:
                    # Mark the transition as complete
                    matched[i] = True
                    matched[j] = True

                    # Append the OFF transition to the ON. Add to dataframe.
                    matched_pairs.loc[len(matched_pairs),:] = [index[j], values[j], index[i], values[i]]
                    #print("MATCH")
                    break
                
                # Stop when falling below the events value
                if cur_power[j] < (values[i] - matchtol):
                    #print("Too low")
                    break

    # Clean and return
    free(matched)
    free(cur_power)
    return matched_pairs



def pair_fast_inner_activeonly2(np.ndarray index, np.ndarray values, float min_tolerance, float percent_tolerance, float large_transition):
    '''
    index brauche ich ggf gar nicht

    This is the dramatically accelerated version of the pairing.
    Here the values are passed as an array to accelerate the process dramatically.
    '''
    # Create variables used later
    matched_pairs = pd.DataFrame(columns= ['T1 Time', 'T1 Active', 'T2 Time', 'T2 Active'])
    cdef size_t i  
    cdef size_t j 
    cdef float matchtol

    # Create a cur_power array
    cdef bint *matched = <bint *>malloc(len(values) * sizeof(bint))
    cdef float *cur_power = <float *>malloc(len(values) * sizeof(float))
    cdef float prev_power = 0
    for i in range(len(values)-1, -1, -1):
        cur_power[i] = prev_power + values[i]
        prev_power = cur_power[i]
        matched[i] = False

    # Go from end to beginning
    for i in range(len(values)-1, -1, -1):
        #print("i:{0} = {1}".format(i, values[i]))
        if (values[i] < 0):
            for j in range(i-1, -1, -1):
                #print("j:{0} = {1}".format(j, values[j]))

                if matched[j] or values[j] < 0:
                    # Only positive switches may fit
                    #print("CONTINUE")
                    continue

                # Add the two elements for comparison
                if (abs(values[i] - values[j])) < large_transition:
                    matchtol = min_tolerance
                else: 
                    matchtol = percent_tolerance * max(np.fabs([values[i], values[j]]))
              
                # Check whether a valid pair
                if abs(values[i] + values[j]) < matchtol:
                    # Mark the transition as complete
                    matched[i] = True
                    matched[j] = True

                    # Append the OFF transition to the ON. Add to dataframe.
                    matched_pairs.loc[len(matched_pairs),:] = [index[j], values[j], index[i], values[i]]
                    #print("MATCH")
                    break
                
                # Stop when falling below the events value
                if cur_power[j] < (values[i] - matchtol):
                    #print("Too low")
                    break

    # Clean and return
    free(matched)
    free(cur_power)
    return matched_pairs





def _free_enumerable_fast(element):
    if isinstance(element, (list, np.ndarray, pd.DatetimeIndex)):
        last_index = element[-1]
    elif isinstance(element, pd.DataFrame):
        last_index = element.index[-1]
    else:
        raise("Function not working for this type.")
    del element
    gc.collect()
    return last_index


#def get_good_sections_fast(df, max_sample_period, look_ahead=None,
#                      previous_chunk_ended_with_open_ended_good_section=False):
#    """
#    Parameters
#    ----------
#    df : pd.DataFrame
#    look_ahead : pd.DataFrame
#    max_sample_period : number

#    Returns
#    -------
#    sections : list of TimeFrame objects
#        Each good section in `df` is marked with a TimeFrame.
#        If this df ends with an open-ended good section (assessed by
#        examining `look_ahead`) then the last TimeFrame will have
#        `end=None`.  If this df starts with an open-ended good section
#        then the first TimeFrame will have `start=None`.
#    """
#    index = df.dropna().sort_index().index
#    _free_enumerable_fast(df)

#    if len(index) < 2:
#        return []

#    # Determine where there are missing samples
#    timedeltas_sec = timedelta64_to_secs(diff(index.values))
#    timedeltas_check = timedeltas_sec <= max_sample_period
#    _free_enumerable_fast(timedeltas_sec)
    
#    # Determine start and end of good sections (after/before missing samples regions)
#    timedeltas_check = concatenate(
#        [[previous_chunk_ended_with_open_ended_good_section],
#         timedeltas_check])
#    transitions = diff(timedeltas_check.astype(np.int))
#    last_timedeltas_check  = _free_enumerable_fast(timedeltas_check)
#    good_sect_starts = list(index[:-1][transitions ==  1])
#    good_sect_ends   = list(index[:-1][transitions == -1])
#    last_index  = _free_enumerable_fast(index)

#    # Use look_ahead to see if we need to append a 
#    # good section start or good section end.
#    look_ahead_valid = look_ahead is not None and not look_ahead.empty
#    if look_ahead_valid:
#        look_ahead_timedelta = look_ahead.dropna().index[0] - last_index
#        look_ahead_gap = look_ahead_timedelta.total_seconds()
#    if last_timedeltas_check: # current chunk ends with a good section
#        if not look_ahead_valid or look_ahead_gap > max_sample_period:
#            # current chunk ends with a good section which needs to 
#            # be closed because next chunk either does not exist
#            # or starts with a sample which is more than max_sample_period
#            # away from df.index[-1]
#            good_sect_ends += [last_index]
#    elif look_ahead_valid and look_ahead_gap <= max_sample_period:
#        # Current chunk appears to end with a bad section
#        # but last sample is the start of a good section
#        good_sect_starts += [last_index]

#    # Work out if this chunk ends with an open ended good section
#    all_sections_closed = (
#        len(good_sect_ends) > len(good_sect_starts) or 
#        len(good_sect_ends) == len(good_sect_starts) and not previous_chunk_ended_with_open_ended_good_section)
#    ends_with_open_ended_good_section = not all_sections_closed

#    # If this chunk starts or ends with an open-ended good 
#    # section then the missing edge is remembered by a None
#    # at the begging/end. (later in the overallresult this
#    # can then be stacked together above multiple chunks )
#    if previous_chunk_ended_with_open_ended_good_section:
#        good_sect_starts = [None] + good_sect_starts
#    if ends_with_open_ended_good_section:
#        good_sect_ends += [None]

#    # Merge together starts and ends and return sections 
#    # as result for timeframe of this chunk
#    assert len(good_sect_starts) == len(good_sect_ends)
#    sections = [TimeFrame(start, end)
#                for start, end in zip(good_sect_starts, good_sect_ends)
#                if not (start == end and start is not None)]
#    _free_enumerable_fast([good_sect_starts, good_sect_ends])
#    return sections