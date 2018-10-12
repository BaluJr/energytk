from __future__ import print_function, division
from datetime import datetime
from nilmtk.timeframe import merge_timeframes, TimeFrame
#from nilmtk.elecmeter import ElecMeter
from nilmtk.processing import Processing


class DisaggregatorModel(object):
    '''
    This is the base class for all models of the disaggregator classes.
    The models always contain the paramters and the model itself.
    '''

    # This dictionary shall be overwritten by each disaggregator with 
    # fields for the parameters it requires.
    params = {}



class Disaggregator(Processing):
    """ Provides the baseclass for all disaggregation classes.
    Is further subdevided by the two subclasses SupervisedDisaggregtor 
    and UnsupervisedDisaggregator.

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


    '''
    This attribute declares which data is necessary to use the disaggregator.
    Whenever a disaggregation or training is performed, the dataset is checked 
    for the fullfillment of these requirements
    '''
    Requirements = {
        'max_sample_period': 900,
        'physical_quantities': [['power','active']]
    }
    
    def disaggregate(self, mains, output_datastore = ""):
        """Passes each chunk from mains generator to disaggregate_chunk() and
        passes the output to _write_disaggregated_chunk_to_datastore()
        Will have a default implementation in super class.  Can be
        overridden for more simple in-memory disaggregation, or more
        complex out-of-core disaggregation.

        Parameters
        ----------
        mains : nilmtk.ElecMeter (single-phase) or
            nilmtk.MeterGroup (multi-phase)
        output_datastore : instance of nilmtk.DataStore or str of
            datastore location
        """
        raise NotImplementedError()

    def disaggregate_chunk(self, mains):
        """In-memory disaggregation.

        Parameters
        ----------
        mains : pd.DataFrame

        Returns
        -------
        appliances : pd.DataFrame where each column represents a
            disaggregated appliance
        """
        raise NotImplementedError()


    def _save_metadata_for_disaggregation(self, output_datastore,
                                          sample_period, measurement,
                                          timeframes, building,
                                          meters=None, num_meters=None,
                                          supervised=True):
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
        num_meters : int
            Required if `supervised=False`
        """

        # TODO: `preprocessing_applied` for all meters
        # TODO: submeter measurement should probably be the mains
        #       measurement we used to train on, not the mains measurement.

        # DataSet and MeterDevice metadata:
        building_path = '/building{}'.format(building)
        mains_data_location = building_path + '/elec/meter1'

        meter_devices = {
            self.MODEL_NAME : {
                'model': self.MODEL_NAME,
                'sample_period': sample_period,
                'max_sample_period': sample_period,
                'measurements': [{
                    'physical_quantity': measurement[0],
                    'type': measurement[1]
                }]
            },
            'mains': {
                'model': 'mains',
                'sample_period': sample_period,
                'max_sample_period': sample_period,
                'measurements': [{
                    'physical_quantity': measurement[0],
                    'type': measurement[1]
                }]
            }
        }

        merged_timeframes = merge_timeframes(timeframes, gap=sample_period)
        total_timeframe = TimeFrame(merged_timeframes[0].start,
                                    merged_timeframes[-1].end)

        date_now = datetime.now().isoformat().split('.')[0]
        dataset_metadata = {
            'name': self.MODEL_NAME,
            'date': date_now,
            'meter_devices': meter_devices,
            'timeframe': total_timeframe.to_dict()
        }
        output_datastore.save_metadata('/', dataset_metadata)

        # Building metadata

        # Mains meter:
        elec_meters = {
            1: {
                'device_model': 'mains',
                'site_meter': True,
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
                    'device_model': self.MODEL_NAME,
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
            for chan in range(2, num_meters + 2):
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
            'instance': building,
            'elec_meters': elec_meters,
            'appliances': appliances
        }

        output_datastore.save_metadata(building_path, building_metadata)

    def _write_disaggregated_chunk_to_datastore(self, chunk, datastore):
        """ Writes disaggregated chunk to NILMTK datastore.
        Should not need to be overridden by sub-classes. Only writes 
        non zero sections to storage.

        Parameters
        ----------
        chunk : pd.DataFrame representing a single appliance
            (chunk needs to include metadata)
        datastore : nilmtk.DataStore
        """
        raise NotImplementedError()
