from __future__ import print_function, division
from datetime import datetime
from nilmtk.timeframe import merge_timeframes, TimeFrame
from nilmtk.timeframegroup import TimeFrameGroup
from nilmtk.disaggregate import Disaggregator, DisaggregatorModel


class UnsupervisedDisaggregatorModel(DisaggregatorModel):
    
    '''
    Amount of meters detected within the trainingdata
    '''
    num_meters = None,
    
    '''
    This is the model used for the UnsupevisedDisaggregator.
    It contains the learned model data as well as the 
    paramterization of how it has been created. In contrast to 
    the other models, the all unsupervised models are always 
    anonymous, that means no ids of meters are stored.
    '''
    def extendModels(otherModel):
        '''
        For unsupervised learning, the extend function for the models 
        is even more important as for the supervised/transfer case,
        because each data to disaggregate can be seen as new 
        training data.        
        '''
        raise NotImplementedError("Not yet created!")



class UnsupervisedDisaggregator(Disaggregator):
    """Provides a common interface to all disaggregation classes.

    See https://github.com/nilmtk/nilmtk/issues/271 for discussion, and
    nilmtk/docs/manual/development_guide/writing_a_disaggregation_algorithm.md
    for the development guide.

    Attributes
    ----------
    model :
        Each subclass should internally store models learned from training.

    MODEL_NAME : string
        A short name for this type of model.
        e.g. 'CO' for combinatorial optimisation.
        Used by self._save_metadata_for_disaggregation.
    """
    

    def _save_metadata_for_disaggregation(self, output_datastore, sample_period, measurement, timeframes,
                                          building,meters=None, num_meters=None, supervised=True,
                                          original_building_meta = None, rest_powerflow_included = False):
        """Add metadata for disaggregated appliance estimates to datastore.

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

        # DataSet and MeterDevice metadata only when not already available
        try:
            metadata = output_datastore.load_metadata()
            timeframes.append(TimeFrame(start=metadata["timeframe"]["start"], end=metadata["timeframe"]["end"]))
            total_timeframe = TimeFrameGroup(timeframes).get_timeframe()
            dataset_metadata = {
                'name': metadata["name"],
                'date': metadata["date"],
                'meter_devices': metadata["meter_devices"],
                'timeframe': total_timeframe.to_dict()
            }
            output_datastore.save_metadata('/', dataset_metadata)
        except:
            pq = 3
            meter_devices = {
                'disaggregate' : {
                    'model': type(self), #self.model.MODEL_NAME,
                    'sample_period': sample_period if rest_powerflow_included else 0, # Makes it possible to use special load functionality
                    'max_sample_period': sample_period,
                    'measurements': [{
                        'physical_quantity': 'power', #measurement.levels[0][0],
                        'type': 'active' #measurement.levels, #[1][0]
                    }]
                }}

            if rest_powerflow_included:
                meter_devices['rest'] = {
                        'model': 'rest',
                        'sample_period': sample_period,
                        'max_sample_period': sample_period,
                        'measurements': [{
                            'physical_quantity': 'power', #measurement.levels, #[0][0],
                            'type': 'active' #measurement.levels, #[1][0]
                        }]
                    }
            total_timeframe = TimeFrameGroup(timeframes).get_timeframe()

            date_now = datetime.now().isoformat().split('.')[0]
            dataset_metadata = {
                'name': type(self),
                'date': date_now,
                'meter_devices': meter_devices,
                'timeframe': total_timeframe.to_dict()
            }
            output_datastore.save_metadata('/', dataset_metadata)


        # Building metadata always stored for the new buildings
        for i in range(len(num_meters)):
            phase_building = building * 10 + i 
            building_path = '/building{}'.format(phase_building)
            mains_data_location = building_path + '/elec/meter1'

            # Rest meter:
            elec_meters = {}
            if rest_powerflow_included:
                elec_meters[1] = {
                    'device_model': 'rest',
                    #'site_meter': True,
                    'data_location': mains_data_location,
                    'preprocessing_applied': {},  # TODO
                    'statistics': {
                        'timeframe': total_timeframe.to_dict()
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

            if len(appliances) == 0:
                continue 

            building_metadata = {
                'instance': (phase_building),
                'elec_meters': elec_meters,
                'appliances': appliances,
                'original_name': original_building_meta['original_name'] if 'original_name' in original_building_meta else None,
                'geo_location': original_building_meta['geo_location'] if 'geo_location' in original_building_meta else None,
                'zip': original_building_meta['zip'] if 'zip' in original_building_meta else None,
            }
            print(building_path)
            output_datastore.save_metadata(building_path, building_metadata)
   