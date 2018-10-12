import numpy as np
import pandas as pd 
from nilmtk import TimeFrame
import datetime

import matplotlib.pyplot as plt
# transients['active transition'].cumsum().resample('2s', how='ffill').plot()

class PredefinedStateMachines(object):
    """
    This class creates a predetermined statemachine. 
    It is mainly usefull for testing purposes.
    """

    def simulate(self, output_datastore, appliance_specs = None):
        '''
        Builds the appliances from the specification

        Parameters
        ----------
        appliance_specs: 
            The specification of the appliances. It is a dataframe with relative values and 
            one column for each appliance.
        target_file: str
            The path to the file where the powerflow shall be created.
        duration: pd.Timedelta
            The timestep every entry is translated to.
        
        Returns
        -------
        transients: pd.DataFrame
            The transients of the load profile
        steady_states: pd.DataFrame
            The steady states of the load profile
        '''
        duration = "60m"

        index = pd.DatetimeIndex(start = pd.Timestamp("01.01.2018"), periods = len(appliance_specs), freq = pd.Timedelta(duration))
        appliance_specs.index = index
        # Write into file
        building_path = '/building{}'.format(1)
        for i, appliance in enumerate(appliance_specs):
            key = '{}/elec/meter{:d}'.format(building_path, i + 2)
            data = appliance_specs[[appliance]]
            data.columns = pd.MultiIndex.from_tuples([('power', 'active')], names=['physical_quantity', 'type'])
            output_datastore.append(key, data)
        num_meters = len(appliance_specs.columns)

        # Write the metadata
        timeframe = TimeFrame(start = index[0], end = index[-1])
        self._save_metadata_for_disaggregation(output_datastore, timeframe, num_meters, appliance_specs.columns)
    
        
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
                'sample_period': 3600, # Makes it possible to use special load functionality
                'max_sample_period': 3600,
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
        for chan in range(2, num_meters+2):
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
   
