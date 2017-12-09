

class GaussianStateMachines(object):
    """
    This class is a basic simulator, which creates sample loads by randomizing signatures 
    of some predefined statemachine appliances.
    The randomization is performed by a perfect gaussian distribution, whose stddev can be 
    defined.
    The signatures of all the appliances are superimposed to yield a final load profile.
    """

    def simulate(self, appliance_specs, duration = = 8640000):
        '''
        Performs the simulation of a defined interval of load profile.
        The style of the output is heavily linked to the EventbasedCombination
        disaggregator.

        Parameters
        ----------
        duration: pd.Timedelta
            Circa duration of the created load profile.
            Default 100 days
        Returns
        -------
        transients: pd.DataFrame
            The transients of the load profile
        steady_states: pd.DataFrame
            The steady states of the load profile
        '''

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

        return transients, steady_states


