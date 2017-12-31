import numpy as np
import pandas as pd 
from nilmtk import TimeFrame
import datetime

class GaussianStateMachines(object):
    """
    This class is a basic simulator, which creates sample loads by randomizing signatures 
    of some predefined statemachine appliances.
    The randomization is performed by a perfect gaussian distribution, whose stddev can be 
    defined.
    The signatures of all the appliances are superimposed to yield a final load profile.
    """

    def simulate(self, output_datastore, appliance_specs = None, duration = 8640000):
        '''
        Performs the simulation of a defined interval of load profile.
        The style of the output is heavily linked to the EventbasedCombination
        disaggregator.
        The target file is filled with submeters for each appliance and a single 
        site_meter.

        Parameters
        ----------
        appliance_specs: 
            The specification of the appliances. See the default appliances 
            created in the constructor to have a description of the default
            format
        target_file: str
            The path to the file where the powerflow shall be created.
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
                [((300, 0, 60*60),(10, 5, 10)), ((-300, 1, 60*60*10),(10, 5, 10))],
                [((40, 0, 50), (6, 2, 10)),     ((120, 0, 40), (15, 2, 10)),    ((-160, 10, 200), (10, 1, 30))],
                [((100, 0, 40), (10, 3, 10)),   ((-26, 0, 180), (5, 2, 50)),    ((-74,5, 480), (15,1,50))]]
        # Breaks as appearances, break duration, stddev
        break_spec = [[4, 60, 10], [6, 10*60,10], [7, 10*60,10], [2, 60,10], [4, 60, 10], [2, 60, 10]]
        for i, bs in enumerate(break_spec): 
            bs[0] = bs[0]*len(specs[i])

        appliance_names = ['Synthetic Heater1', 'Synthetic Heater2', 'Fridge', 'Long appliance', 'Complex1', 'Complex2']

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
        
        # Create the overall powerflow as mixture of single appliances
        transients = pd.concat(appliances, verify_integrity = True)
        transients = transients.sort_index()
        
        # Write into file
        building_path = '/building{}'.format(1)
        for appliance in range(len(appliances)):
            key = '{}/elec/meter{:d}'.format(building_path, appliance + 2)
            data = appliances[appliance]['active transition'].append(pd.Series(0, name='power active', index=appliances[appliance]['active transition'].index - pd.Timedelta('0.5sec')))
            data = pd.DataFrame(data.sort_index().cumsum())
            data.columns = pd.MultiIndex.from_tuples([('power', 'active')], names=['physical_quantity', 'type'])
            output_datastore.append(key, data)
        overall = transients['active transition'].append(pd.Series(0, name='power active', index=appliances[appliance]['active transition'].index - pd.Timedelta('0.5sec')))
        overall = pd.DataFrame(overall.sort_index().cumsum())
        overall.columns = pd.MultiIndex.from_tuples([('power', 'active')], names=['physical_quantity', 'type'])
        output_datastore.append('{}/elec/meter{:d}'.format(building_path, 1), overall)
        num_meters = len(appliances) + 1

        # Write the metadata
        timeframe = TimeFrame(start = transients.index[0], end = transients.index[-1])
        self._save_metadata_for_disaggregation(output_datastore, timeframe, num_meters, appliance_names)

        # The immediate result
        steady_states = transients[['active transition']].cumsum()#.rename({'active transition':'active average'})
        steady_states[['active transition']] += 60
        transients = transients[['active transition', 'spike', 'signature', 'ends']]
        return transients, steady_states
    
        
    def _save_metadata_for_disaggregation(self, output_datastore, timeframe, num_meters, appliancetypes):
        """ 
        Stores the metadata within the storage.

        REMINDER: Also urpruenglich wollte ich das anders machen und eben auch die Metadatan mit abspeichern.
                  Habe ich aus zeitgruenden dann gelassen und mache es doch so wie es vorher war.
        
        This function first checks whether there are already metainformation in the file.
        If zes, it extends them and otherwise it removes them.

        Note that `self.MODEL_NAME` needs to be set to a string before
        calling this method.  For example, we use `self.MODEL_NAME = 'CO'`
        for Combinatorial Optimisation.

        TODO:`preprocessing_applied` for all meters
        TODO: submeter measurement should probably be the mains
              measurement we used to train on, not the mains measurement.

        Parameters
        ----------
        output_datastore : nilmtk.DataStore subclass object
            The datastore to write metadata into.
        timeframe : list of nilmtk.TimeFrames or nilmtk.TimeFrameGroup
            The TimeFrames over which this data is valid for.
        num_meters : [int]
            Required if `supervised=False`, Gives for each phase amount of meters
        appliancetypes: [str]
            The names for the different appliances. Is used in plots and error metric
            tables.
        """

        # Global metadata
        meter_devices = {
            'synthetic' : {
                'model': "Synth",
                'sample_period': 0, # Makes it possible to use special load functionality
                'max_sample_period': 1,
                'measurements': [{
                    'physical_quantity': 'power',
                    'type': 'active'
                }]
            }}
        date_now = datetime.datetime.now().isoformat().split('.')[0]
        dataset_metadata = {
            'name': "Synthetic Gaussian Statemachine",
            'date': date_now,
            'meter_devices': meter_devices,
            'timeframe': timeframe.to_dict()
        }
        output_datastore.save_metadata('/', dataset_metadata)

        # Building metadata always stored for the new buildings
        phase_building = 1
        building_path = '/building{}'.format(phase_building)
        mains_data_location = building_path + '/elec/meter1'

        # Main meter is sum of all single appliances:
        elec_meters = {}
        elec_meters[1] = {
            'device_model': 'synthetic',
            'site_meter': True,
            'data_location': mains_data_location,
            'preprocessing_applied': {},  # TODO
            'statistics': {
                'timeframe': timeframe.to_dict()
            }
        }
        
        def update_elec_meters(meter_instance):
            elec_meters.update({
                meter_instance: {
                    'device_model': 'synthetic', # self.MODEL_NAME,
                    'submeter_of': 1,
                    'data_location': (
                        '{}/elec/meter{}'.format(
                            building_path, meter_instance)),
                    'preprocessing_applied': {},  # TODO
                    'statistics': {
                        'timeframe': timeframe.to_dict()
                    }
                }
            })

        # Appliances and submeters:
        appliances = []
        # Submeters (Starts at 2 because meter 1 is mains and 0 not existing)
        for chan in range(2, num_meters):
            update_elec_meters(meter_instance=chan)
            appliance = {
                'original_name': appliancetypes[chan-2],
                'meters': [chan],
                'type': appliancetypes[chan-2],
                'instance': chan - 1
            }
            appliances.append(appliance)

        building_metadata = {
            'instance': (phase_building),
            'elec_meters': elec_meters,
            'appliances': appliances,
        }
        output_datastore.save_metadata(building_path, building_metadata)
   
