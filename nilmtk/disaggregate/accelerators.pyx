import pandas as pd
DEBUG = False
import time 
import sys
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
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


cdef class MyType:
    cdef vector[int] v


cdef class PowerStack:

    cdef vector[vector[int]] v
    cdef vector[float] avg

    def push(self, x, val):
        cdef vector[int] xx = vector[int]()
        xx.push_back(x)
        self.v.push_back(xx)
        self.avg.push_back(val)

    def cur_avg(self):
        return self.avg.back() # / len(self.v.back()) #np.mean(self.v.back())

    def add(self, x, val):
        self.v.back().push_back(x)
        #print('Before add ' + str(self.avg.back()))
        self.avg[len(self.avg)-1] = val #  + self.avg.back()
        #print('After add ' + str(self.avg.back()))

    def pop(self):
        if self.v.empty():
            raise IndexError()
        x = self.v.back()
        self.v.pop_back()
        self.avg.pop_back()
        return x

    def last(self):
        return self.v.back().back()

def find_sections(inputs):
    """
    This is the self invented function to identify sections in the powerflow which can be 
    analyzed separately.
    
    
    Parameters
    ----------
    indices: as
    states:  as
    stateThreshold: 
    """
    cdef np.ndarray indices
    cdef np.ndarray steady_states
    cdef float state_threshold
    indices, states, state_threshold = inputs

    segments = np.array([""]*len(indices),dtype=object)
    cdef float cur_power
    cdef float comp_power  
    cdef PowerStack stack = PowerStack()
    cdef size_t i  
    cdef np.ndarray to_set
    cur_power = 0
    stack.push(-1, 0)
    for i in range(len(indices)):

        # Reduce stack if dropped below certain level
        while states[i] < (stack.cur_avg() - state_threshold):
            #print(str(i) + ': Down to ' + str(states[i]) + ": popped " + str(stack.cur_avg()))
            cur = stack.pop()
            # When a stable state identify new subclasses
            if len(cur) >= 3: 
                for j in range(len(cur)-1):
                    #print('With append to ids' + str(i) +":" + str(j) + "_" + str(segments[cur[j]:cur[j+1]]))
                    segments[cur[j]:cur[j+1]] = str(j) + "_" + segments[cur[j]:cur[j+1]] 
            # Always add a layer information to all events inside the current hill (from last element on stack on)
            to_set = (segments[stack.last():i] != '')
            segments[stack.last(): i][to_set] = "|" + segments[stack.last(): i][to_set]


        # The new state (must be going up)
        if np.abs(states[i] - stack.cur_avg()) <= state_threshold:
            #print(str(i) + ': Reached existing state ' + str(stack.cur_avg()) + ": " + str(states[i]) + " added!" )
            stack.add(i, states[i])
        else:
            #print(str(i) + ': Reached new state ' + str(states[i]) + ": push " + str(states[i]))
            stack.push(i, states[i])
        cur_power = states[i]

    return segments


#region Event Detection
def find_steady_states_fast(inputs): #np.ndarray indices, np.ndarray values, int min_n_samples=2, int state_threshold=15, int noise_level=70):
    """Finds steady states given a DataFrame of power with only
    a single column.

    As the name already implies: Finds transitions and not events! That means switches between steady states.

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
    cdef np.ndarray indices
    cdef np.ndarray values 
    cdef int min_n_samples 
    cdef int state_threshold
    cdef int noise_level

    # Tells whether we have both real and reactive power or only real power
    #cdef int num_measurements = len(dataframe.columns)
    cdef float estimated_steady_power = 0 # = np.array([0] * num_measurements)
    cdef float last_steady_power = 0 # = np.array([0] * num_measurements)
    cdef float previous_measurement = 0 #= np.array([0] * num_measurements)
    cdef float last_transition

    # These flags store state of power
    instantaneous_change = False  # power changing this second
    ongoing_change = False  # power change in progress over multiple seconds
    
    indices, values, min_n_samples, state_threshold, noise_level = inputs

    index_transitions = []      # Indices to use in returned Dataframe
    index_transitions_end = []  # stopping of events
    index_steady_states = []
    transitions = []  # holds information on transitions
    steady_states = []  # steadyStates to store in returned Dataframe
    cdef int N = 0  # N stores the number of samples in state
    curTime = indices[0] # dataframe.iloc[0].name  # first state starts at beginning

    # Iterate over the rows performing algorithm
    print ("Finding Edges, please wait ...")
    sys.stdout.flush()

    #oldTime = time.clock()
    #i = 0
    cdef size_t i  
    for i in range(len(indices)):#row in dataframe.itertuples():
        print(i)
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
            print(str(i) + ': Instant change')
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
    return (steady_states, transitions)




def find_steady_states_transients_fast(metergroup, cols, int noise_level,
                                  int state_threshold, **load_kwargs):
    """
    Wrapper not used at the moment.
    Creates the states transient list over multiple meters in a 
    meter group. Not used any more, since parallelism is not possible then.

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


def find_transients_fast(inputs): #np.ndarray indices, np.ndarray values, int min_n_samples=2, int state_threshold=15, int noise_level=70):
    """Finds steady states given a DataFrame of power with only
    a single column.

    As the name already implies: Finds transitions and not events! That means switches between steady states.
    This is the extended version, which also extracts the signature of the events.

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
    cdef np.ndarray indices
    cdef np.ndarray values 
    cdef int min_n_samples 
    cdef float state_threshold
    cdef float noise_level

    # Tells whether we have both real and reactive power or only real power
    #cdef int num_measurements = len(dataframe.columns)
    cdef float estimated_steady_power = 0 # = np.array([0] * num_measurements)
    cdef float last_steady_power = 0 # = np.array([0] * num_measurements)
    cdef float previous_measurement = 0 #= np.array([0] * num_measurements)
    cdef float last_transition

    # These flags store state of power
    instantaneous_change = False  # power changing this second
    ongoing_change = False  # power change in progress over multiple seconds
    
    indices, values, min_n_samples, state_threshold, noise_level = inputs

    index_transitions = []      # Indices to use in returned Dataframe
    index_transitions_end = []  # stopping of events
    index_steady_states = []
    transitions = []  # holds information on transitions
    transitions_signatures = [[0, -1, 0]]
    steady_states = []  # steadyStates to store in returned Dataframe
    cdef int N = 0  # N stores the number of samples in state
    cur_event_start = indices[0] # dataframe.iloc[0].name  # first state starts at beginning
    cur_event_end = indices[0] # dataframe.iloc[0].name  # first state starts at beginning
    cur_event_signature = []

    # Iterate over the rows performing algorithm
    print ("Finding Edges, please wait ...")
    sys.stdout.flush()

    #oldTime = time.clock()
    #i = 0
    cdef size_t i  
    for i in range(len(indices)):

        # Step 2: this does the threshold test and then we sum the boolean
        # array.
        this_measurement = values[i] #row[1]#:3]

        # Elementwise absolute differences
        delta = this_measurement - previous_measurement
        state_change = abs(delta)

        if state_change > state_threshold: #np.sum(state_change > state_threshold):
            instantaneous_change = True
        else:
            instantaneous_change = False

        # Step 3: Identify if transition is just starting, if so, process it
        if instantaneous_change and (not ongoing_change):

            # Calculate transition size
            last_transition = estimated_steady_power - last_steady_power 

            # Sum Boolean array to verify if transition is above noise level
            if abs(last_transition) > noise_level: #np.sum(np.fabs(last_transition) > noise_level):
                # 3A, C: if so add the index of the transition start and the
                # power information

                # Avoid outputting first transition from zero
                index_transitions.append(cur_event_start)
                index_transitions_end.append(cur_event_end)
                transitions.append(last_transition)
                transitions_signatures.append(cur_event_signature)

                # I think we want this, though not specifically in Hart's algo notes
                # We don't want to append a steady state if it's less than min samples in length.
                # if N > min_n_samples:
                index_steady_states.append(cur_event_start)
                # logging.debug('The ''curTime'' stored is: %s' % (curTime))
                # last states steady power
                steady_states.append(estimated_steady_power)
                
                # 3B
                last_steady_power = estimated_steady_power
            # 3C
            cur_event_start = indices[i] #row[0] # Wichtig! Wird nur geupdated wenn kein ongoing_state
            cur_event_signature.append(delta)

        # Added sothat we can track lenght of transitions
        if not instantaneous_change and ongoing_change:
            cur_event_end = indices[i]
            cur_event_signature.append(delta) # Fuege ich auch mal noch hinzu das Ende

        # Remember the signature of the switching events
        if instantaneous_change and ongoing_change:
            cur_event_signature.append(delta)
        
        # Always prepare that an event might occur and store current step
        if not instantaneous_change:
            cur_event_signature = [delta]

        # Step 4: if a new steady state is starting, zero counter
        if instantaneous_change:
            N = 0

        # Hart step 5: update our estimate for steady state's energy
        estimated_steady_power = (N * estimated_steady_power + this_measurement) / (N + 1)

        # Step 6: increment counter
        N += 1

        # Step 7
        ongoing_change = instantaneous_change

        # Step 8
        previous_measurement = this_measurement

    # Appending last edge (because there is no event which could start this)
    last_transition = np.subtract(estimated_steady_power, last_steady_power)
    if np.sum(np.fabs(last_transition) > noise_level):
        index_transitions.append(cur_event_start)
        index_transitions_end.append(cur_event_end)
        transitions.append(last_transition)
        transitions_signatures.append(cur_event_signature)
        index_steady_states.append(cur_event_start)
        steady_states.append(estimated_steady_power)

    if len(steady_states)>0 and np.sum(steady_states[0] > noise_level) and index_transitions[0] == index_steady_states[0]:# == dataframe.iloc[0].name:
        transitions = transitions[1:]
        index_transitions = index_transitions[1:]
        index_transitions_end = index_transitions_end[1:]
        transitions_signatures = transitions_signatures[1:]
        steady_states = steady_states[1:]
        index_steady_states = index_steady_states[1:]
    else:
        transitions_signatures[0][1] = steady_states[0]
    
    print("Edge detection complete.")

    print("Creating transition frame ...")
    sys.stdout.flush()
    transitions = pd.DataFrame(data={'active transition': transitions, 'ends': index_transitions_end, 'signature': transitions_signatures[:-1]}, index=index_transitions)
    #transitions = pd.DataFrame(data=transitions, index=index_transitions, columns=['active transition']) #cols_transition[num_measurements])

    print("Transition frame created.")

    print("Creating states frame ...")
    sys.stdout.flush()
    
    steady_states = pd.DataFrame(data=steady_states, index=index_steady_states,
                                    columns=['active average']) #cols_steady[num_measurements])
    print("States frame created.")
    print("Finished.")
    return (steady_states, transitions)




def find_transients_baranskistyle_fast(inputs): #np.ndarray indices, np.ndarray values, int min_n_samples=2, int state_threshold=15, int noise_level=70):
    """ 3.Finds steady states given a DataFrame of power with only a single column.

    This is the self-written version, which works as specified in the baranski paper. 
    Does not work properly.

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
    cdef np.ndarray indices
    cdef np.ndarray values 
    cdef int min_n_samples 
    cdef int state_threshold
    cdef int noise_level

    cdef float cur_steady_state = 0   
    cdef float cur_measurement = 0    
    cdef float prev_measurement = 0    
    cdef bint prev_dir
    cdef bint cur_dir

    # Map inputs
    indices, values, min_n_samples, state_threshold, noise_level = inputs
    print("noise_level: " + str(noise_level))
    print("state_threshold: " + str(state_threshold))
    print("len: " + str(len(values)))
    time.sleep(1)

    # These flags store state of power
    instantaneous_change = False    # power changing this second
    ongoing_change = False          # power change in progress over multiple seconds
    curTime = indices[0]            # dataframe.iloc[0].name  # first state starts at beginning
    prev_dir = False                # True = up, False = down
    
    transitions = []                # holds delta of transitions (adds up over whole 
    transitions_boost = []          # highest single delta within a transition
    index_transitions_start = []    # Indices where transition i starts
    index_transitions_end = []      # Indices where transition i ends

    cur_steady_state = 0            # Sum of the powers within the current steady state (exp. smoothing)


    # Iterate over the rows performing algorithm
    print("A")
    cdef size_t i = 0
    while i < len(indices):

        # Calc delta
        cur_index = indices[i]
        cur_measurement = values[i]
        cur_delta = cur_measurement - prev_measurement
        cur_dir = cur_delta >= 0
        cur_steady_delta = cur_measurement - cur_steady_state

        if ongoing_change:
            # Ongoing changes belonging to an existing event
            still_ongoing = (cur_dir == prev_dir) and (abs(cur_delta) > noise_level)
            # event still continuing
            if still_ongoing:
                transitions_boost[-1] = max(transitions_boost[-1], cur_delta) if cur_dir else min(transitions_boost[-1], cur_delta)
                transitions[-1] += cur_delta
            else:
                # event finished
                cur_steady_state = cur_measurement
                ongoing_change = False
                index_transitions_end.append(cur_index)
                # Start from current index to allow event from current position on
                print("Continue " + str(i))
                continue
        
        elif abs(cur_steady_delta) > state_threshold:
            # A new event starts
            transitions.append(cur_delta)
            transitions_boost.append(cur_delta)
            index_transitions_start.append(cur_index)
            ongoing_change = True

        else:
            # Just another step in current steady state
            cur_steady_state = cur_steady_state * 0.8 + cur_measurement * 0.2
        
        prev_dir = cur_dir
        prev_measurement = cur_measurement
        i+=1

        ## Step 3: Identify if transition is just starting, if so, process it
        #if instantaneous_change:
        #    last_transition = estimated_steady_power - last_steady_power
        #    if ongoing_change:
        #        transitions[-1] += last_transition
        #    else:
        #        # 3A Add when transition over noise level
        #        if abs(last_transition) > noise_level: 
        #            index_transitions_start.append(curTime)
        #            transitions.append(last_transition)
        #            index_steady_states.append(curTime)
        #            steady_states.append(estimated_steady_power)
                
        #        # 3B
        #        last_steady_power = estimated_steady_power
        #        # 3C
        #        curTime = indices[i]

        #    # Step 4: if a new steady state is starting, zero counter
        #    if instantaneous_change:
        #        N = 0
        ## When there was a ongoing change which is now over
        #elif ongoing_change:
        #    last_transition = estimated_steady_power - last_steady_power
        #    if abs(last_transition) > noise_level: 
        #        index_transitions_end.append(curTime)

    print("B")

    ## Appending last edge
    #last_transition = estimated_steady_power - last_steady_power
    #if last_transition> noise_level:
    #    index_transitions_start.append(curTime)
    #    transitions.append(last_transition)
    #    index_steady_states.append(curTime)
    #    steady_states.append(estimated_steady_power)
    #print("C")
    ## Removing first edge if the starting steady state power is more
    ## than the noise threshold  https://github.com/nilmtk/nilmtk/issues/400
    #if len(steady_states)>0 and steady_states[0] > noise_level and index_transitions_start[0] == index_steady_states[0]:
    #    transitions = transitions[1:]
    #    index_transitions_start = index_transitions_start[1:]
    #    steady_states = steady_states[1:]
    #    index_steady_states = index_steady_states[1:]

    # Return the output
    print(len(transitions))
    print(len(index_transitions_end))
    print(len(index_transitions_start))  
    transitions = pd.DataFrame(data={'active transition': transitions, 'ends': index_transitions_end, 'boost': transitions_boost}, index=index_transitions_start)
    return transitions

#endregion





#region Pairing

def pair_fast(params):
    '''
    This is the dramatically accelerated version of the pairing.
    '''
    transitions, min_tolerance, percent_tolerance, large_transition = params
    if transitions.size < 2:
        return pd.DataFrame(columns= ['T1 Time', 'T1 Active', 'T2 Time', 'T2 Active'])
       
    if len(transitions.columns) <= 10:
        indices = np.array(transitions.index)
        values = np.array(transitions['active transition']) #.iloc[:,0])
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
        # Look for switch-off events
        if values[i] > 0:
            continue
        for j in range(i-1, -1, -1):
            #print("j:{0} = {1}".format(j, values[j]))
            # Look for unmatched switch-on events
            if matched[j] or values[j] < 0:
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

            ## Stop when all events within eventlength checkedprint(max_eventlenght)
            #if index[i] - index[j] > max_eventlength:
            #    break

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
#endregion

#BACKUP
   #for i in range(len(indices)):
   #     if i % 10000 == 0:
   #         print(i)
   #     if states[i] > (cur_power + state_threshold): 
   #         #print(str(i) + ': Up to ' + str(states[i]) + ": push " + str(states[i]))
   #         stack.push(i, states[i])
   #     else:
   #         while states[i] < (stack.cur_avg() - state_threshold):
   #             #print(str(i) + ': Down to ' + str(states[i]) + ": popped " + str(stack.cur_avg()))
   #             cur = stack.pop()
   #             if len(cur) >= 3: 
   #                 for j in range(len(cur)-1):
   #                     #print('With append to ids' + str(i) +":" + str(j) + "_" + str(segments[cur[j]:cur[j+1]]))
   #                     segments[cur[j]:cur[j+1]] = str(j) + "_" + segments[cur[j]:cur[j+1]] 
   #             else:
   #                 segments[cur[0]:i] = "X_" + segments[cur[0]:i] 
   #         if np.abs(states[i] - stack.cur_avg()) < state_threshold:
   #             #print(str(i) + ': Down to ' + str(states[i]) + ": added!" )
   #             stack.add(i, states[i])
   #         else:
   #             #print(str(i) + ': Down to ' + str(states[i]) + ": push " + str(states[i]))
   #             stack.push(i, states[i])
   #     cur_power = states[i]