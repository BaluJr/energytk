from __future__ import print_function, division
from warnings import warn
from collections import namedtuple
from copy import deepcopy
import numpy as np
import pandas as pd
from six import iteritems
from .preprocessing import Clip
from .stats import TotalEnergy, GoodSections, DropoutRate, NonZeroSections, OverBasepowerSections
from .hashable import Hashable
from .measurement import (select_best_ac_type, PHYSICAL_QUANTITIES,
                          check_ac_type, check_physical_quantity)
from .node import Node
from .electric import Electric
from nilmtk.exceptions import MeasurementError
from .utils import flatten_2d_list, capitalise_first_letter
from nilmtk.timeframegroup import TimeFrameGroup
#from nilmtk.stats.accelerators_stat import get_good_sections_fast
import nilmtk


ElecMeterID = namedtuple('ElecMeterID', ['instance', 'building', 'dataset'])

class test():
    pass

class ElecMeter(Hashable, Electric):

    """Represents a physical electricity meter.

    Todo: I have to formalize it further. I defined meters with a sampling_rate of 0.
    In this case, which is used for disaggregation of state_machines, the storing
    takes place by storing only the changes instead of the continuous powerflow. This
    allows a far more efficient storing. If no sampling_rate is given, they are
    returned as 1 sec (In the future one should additionally store a native sampling rate
    which sould be set to the rate which is recommenden. Eg. based on the meter, which
    was used to create the highres meter.)
    One remaining problem for the high_res meters: Only the nonzero sections are loaded and
    therefore handed in to the interpolator. Therefore the loaded section does not necessarily
    has to contain the full intervall. As a result one has to ffill and bfill the whole
    timeframe.

    Attributes
    ----------
    appliances : list of Appliance objects connected immediately downstream
      of this meter.  Will be [] if no appliances are connected directly
      to this meter.

    store : nilmtk.DataStore

    key : string
        key into nilmtk.DataStore to access data.

    metadata : dict.
        See http://nilm-metadata.readthedocs.org/en/latest/dataset_metadata.html#elecmeter

    STATIC ATTRIBUTES
    -----------------

    meter_devices : dict, static class attribute
        See http://nilm-metadata.readthedocs.org/en/latest/dataset_metadata.html#meterdevice
    """

    meter_devices = {}

    def __init__(self, store=None, metadata=None, meter_id=None):
        # Store and check parameters
        self.appliances = []
        self.metadata = {} if metadata is None else metadata
        assert isinstance(self.metadata, dict)
        self.store = store
        self.identifier = meter_id

        # Insert self into nilmtk.global_meter_group
        if self.identifier is not None:
            assert isinstance(self.identifier, ElecMeterID)
            if self not in nilmtk.global_meter_group.meters:
                nilmtk.global_meter_group.meters.append(self)


    #region PROPERTIES
    @property
    def key(self):
        return self.metadata['data_location']

    def instance(self):
        return self._identifier_attr('instance')

    def building(self):
        return self._identifier_attr('building')

    def dataset(self):
        return self._identifier_attr('dataset')

    @property
    def name(self):
        return self.metadata.get('name')

    @name.setter
    def name(self, value):
        self.metadata['name'] = value

    def _identifier_attr(self, attr):
        '''
        Gets a certain attribute from the identifier.
        Main attributes are stored in it.
        '''
        if self.identifier is None:
            return
        else:
            return getattr(self.identifier, attr)

    def values_for_meter_or_building_metadata_key(self, key):
        """
        Included as a distinct function to allow polimorphism 
        with MeterGroup
        Parameters
        ----------
        key : str
            e.g. 'zip'
        
        Returns
        -------
        list
        """
        if key in self.metadata:
            return [self.metadata[key]]
        else:
            building_metadata = self.building_metadata
            if key in building_metadata:
                return [building_metadata[key]]
        return []

    #endregion


    #region INFORMATION
    def get_timeframe(self):
        self._check_store()
        return self.store.get_timeframe(key=self.key)

    def _check_store(self):
        if self.store is None:
            raise RuntimeError("ElecMeter needs `store` attribute set to an"
                               " instance of a `nilmtk.DataStore` subclass")

    def upstream_meter(self, raise_warning=True):
        """
        Returns
        -------
        ElecMeterID of upstream meter or None if is site meter.
        """
        if self.is_site_meter():
            if raise_warning:
                warn("There is no meter upstream of this meter '{}' because"
                     " it is a site meter.".format(self.identifier))
            return

        submeter_of = self.metadata.get('submeter_of')

        # Sanity checks
        if submeter_of is None:
            raise ValueError(
                "This meter has no 'submeter_of' metadata attribute.")
        if submeter_of < 0:
            raise ValueError("'submeter_of' must be >= 0.")
        upstream_meter_in_building = self.metadata.get(
            'upstream_meter_in_building')
        if (upstream_meter_in_building is not None and
                upstream_meter_in_building != self.identifier.building):
            raise NotImplementedError(
                "'upstream_meter_in_building' not implemented yet.")

        id_of_upstream = ElecMeterID(instance=submeter_of,
                                     building=self.identifier.building,
                                     dataset=self.identifier.dataset)

        upstream_meter = nilmtk.global_meter_group[id_of_upstream]
        if upstream_meter is None:
            warn("No upstream meter found for '{}'.".format(self.identifier))
        return upstream_meter

    @classmethod
    def load_meter_devices(cls, store):
        dataset_metadata = store.load_metadata('/')
        ElecMeter.meter_devices.update(
            dataset_metadata.get('meter_devices', {}))

    def save(self, destination, key):
        """
        Convert all relevant attributes to a dict to be
        saved as metadata in destination at location specified
        by key
        """
        # destination.write_metadata(key, self.metadata)
        # then save data
        raise NotImplementedError

    @property
    def device(self):
        """
        Returns
        -------
        dict describing the MeterDevice for this meter (sample period etc).
        """
        device_model = self.metadata.get('device_model')
        if device_model:
            return deepcopy(ElecMeter.meter_devices[device_model])
        else:
            return {}

    @property
    def building_metadata(self):
        device_model = self.store.load_metadata('/building' + str(self.building()))
        if device_model:
            return deepcopy(device_model)
        else:
            return {}

    def sample_period(self):
        device = self.device
        if device:
            return device['sample_period']

    def is_site_meter(self):
        return self.metadata.get('site_meter', False)

    def dominant_appliance(self):
        """Tries to find the most dominant appliance on this meter,
        and then returns that appliance object.  Will return None
        if there are no appliances on this meter.
        """
        n_appliances = len(self.appliances)
        if n_appliances == 0:
            return
        elif n_appliances == 1:
            return self.appliances[0]
        else:
            for app in self.appliances:
                if app.metadata.get('dominant_appliance'):
                    return app
            warn('Multiple appliances are associated with meter {}'
                 ' but none are marked as the dominant appliance. Hence'
                 ' returning the first appliance in the list.', RuntimeWarning)
            return self.appliances[0]

    def label(self, pretty=True):
        """Returns a string describing this meter.

        Parameters
        ----------
        pretty : boolean
            If True then just return the type name of the dominant appliance
            (without the instance number) or metadata['name'], with the
            first letter capitalised.

        Returns
        -------
        string : A label listing all the appliance types.
        """
        if pretty:
            return self._pretty_label()

        meter_names = []
        if self.is_site_meter():
            meter_names.append('SITE METER')
        elif "name" in self.metadata:
            meter_names.append(self.metadata["name"])
        else:
            for appliance in self.appliances:
                appliance_name = appliance.label()
                if appliance.metadata.get('dominant_appliance'):
                    appliance_name = appliance_name.upper()
                meter_names.append(appliance_name)
        label = ", ".join(meter_names)
        return label

    def _pretty_label(self):
        name = self.metadata.get("name")
        if name:
            label = name
        elif self.is_site_meter():
            label = 'Site meter'
        elif self.dominant_appliance() is not None:
            label = self.dominant_appliance().identifier.type
        else:
            meter_names = []
            for appliance in self.appliances:
                appliance_name = appliance.label()
                if appliance.metadata.get('dominant_appliance'):
                    appliance_name = appliance_name.upper()
                meter_names.append(appliance_name)
            label = ", ".join(meter_names)
            return label

        label = capitalise_first_letter(label)
        return label

    def available_ac_types(self, physical_quantity):
        """Finds available alternating current types for a specific physical quantity.

        Parameters
        ----------
        physical_quantity : str or list of strings

        Returns
        -------
        list of strings e.g. ['apparent', 'active']
        """
        if isinstance(physical_quantity, list):
            ac_types = [self.available_ac_types(pq) for pq in physical_quantity]
            return list(set(flatten_2d_list(ac_types)))

        if physical_quantity not in PHYSICAL_QUANTITIES:
            raise ValueError("`physical_quantity` must by one of '{}', not '{}'"
                             .format(PHYSICAL_QUANTITIES, physical_quantity))

        measurements = self.device['measurements']
        return [m['type'] for m in measurements
                if m['physical_quantity'] == physical_quantity
                and 'type' in m]

    def available_physical_quantities(self):
        """
        Returns
        -------
        list of strings e.g. ['power', 'energy']
        """
        measurements = self.device['measurements']
        return list(set([m['physical_quantity'] for m in measurements]))

    def available_columns(self):
        """
        Returns
        -------
        list of 2-tuples of strings e.g. [('power', 'active')]
        """
        measurements = self.device['measurements']
        return list(set([(m['physical_quantity'], m.get('type', ''))
                         for m in measurements]))

    def __repr__(self):
        string = super(ElecMeter, self).__repr__()
        # Now add list of appliances...
        string = string[:-1]  # remove last bracket

        # Site meter
        if self.metadata.get('site_meter'):
            string += ', site_meter'

        # Appliances
        string += ', appliances={}'.format(self.appliances)

        # METER ROOM
        room = self.metadata.get('room')
        if room:
            string += ', room={}'.format(room)

        string += ')'
        return string
    #endregion


    #region LOADING
    def matches(self, key):
        """
        Parameters
        ----------
        key : dict

        Returns
        -------
        Bool
        """

        if not key:
            return True

        if not isinstance(key, dict):
            raise TypeError()

        match = True
        for k, v in iteritems(key):
            if hasattr(self.identifier, k):
                if getattr(self.identifier, k) != v:
                    match = False

            elif k in self.metadata:
                if self.metadata[k] != v:
                    match = False

            elif k in self.device:
                metadata_value = self.device[k]
                if (isinstance(metadata_value, list) and
                        not isinstance(v, list)):
                    if v not in metadata_value:
                        match = False
                elif metadata_value != v:
                    match = False

            elif k in self.building_metadata:
                if self.building_metadata[k] != v:
                    match = False
            else:
                raise KeyError("'{}' not a valid key.".format(k))

        return match

    def load(self, **load_kwargs):
        """Returns a generator of DataFrames loaded from the DataStore.

        By default, `load` will load all available columns from the DataStore.
        Specific columns can be selected in one or two mutually exclusive ways:

        1. specify a list of column names using the `cols` parameter.
        2. specify a `physical_quantity` and/or an `ac_type` parameter to ask
           `load` to automatically select columns.

        If 'resample' is set to 'True' then the default behaviour is for
        gaps shorter than max_sample_period will be forward filled.

        Parameters
        ---------------
        physical_quantity : string or list of strings
            e.g. 'power' or 'voltage' or 'energy' or ['power', 'energy'].
            If a single string then load columns only for that physical quantity.
            If a list of strings then load columns for all those physical
            quantities.

        ac_type : string or list of strings, defaults to None
            Where 'ac_type' is short for 'alternating current type'.  e.g.
            'reactive' or 'active' or 'apparent'.
            If set to None then will load all AC types per physical quantity.
            If set to 'best' then load the single best AC type per
            physical quantity.
            If set to a single AC type then load just that single AC type per
            physical quantity, else raise an Exception.
            If set to a list of AC type strings then will load all those
            AC types and will raise an Exception if any cannot be found.

        cols : list of tuples, using NILMTK's vocabulary for measurements.
            e.g. [('power', 'active'), ('voltage', ''), ('energy', 'reactive')]
            `cols` can't be used if `ac_type` and/or `physical_quantity` are set.

        sample_period : int, defaults to None
            Number of seconds to use as the new sample period for resampling.
            If None then will use self.sample_period()

        resample : boolean, defaults to False
            If True then will resample data using `sample_period`.
            Defaults to True if `sample_period` is not None.

        resample_kwargs : dict of key word arguments (other than 'rule') to
            `pass to pd.DataFrame.resample()`.  Defaults to set 'limit' to
            `sample_period / max_sample_period` and sets 'fill_method' to ffill.

        preprocessing : list of Node subclass instances
            e.g. [Clip()].

        **load_kwargs : any other key word arguments to pass to `self.store.load()`

        Returns
        -------
        Always return a generator of DataFrames (even if it only has a single
        column).

        Raises
        ------
        nilmtk.exceptions.MeasurementError if a measurement is specified
        which is not available.
        """
        verbose = load_kwargs.get('verbose')
        if verbose:
            print()
            print("ElecMeter.load")
            print(self)

        if 'sample_period' in load_kwargs:
            load_kwargs.setdefault('resample', True)

        if self.sample_period() == 0:
            load_kwargs["high_res"] = True

        if load_kwargs.get('resample'):
            # Set default key word arguments for resampling.
            resample_kwargs = load_kwargs.setdefault('resample_kwargs', {})
            resample_kwargs.setdefault('fill_method', 'ffill')
            if 'limit' not in resample_kwargs:
                sample_period = load_kwargs.get('sample_period', self.sample_period())
                max_number_of_rows_to_ffill = int(
                    np.ceil(self.device['max_sample_period'] / sample_period))
                resample_kwargs.update({'limit': max_number_of_rows_to_ffill})

        if verbose:
            print("kwargs after setting resample setting:")
            print(load_kwargs)

        load_kwargs = self._prep_kwargs_for_sample_period_and_resample(**load_kwargs)

        if verbose:
            print("kwargs after processing")
            print(load_kwargs)

        # Get source node
        preprocessing = load_kwargs.pop('preprocessing', [])
        last_node = self.get_source_node(**load_kwargs)
        generator = last_node.generator

        # Connect together all preprocessing nodes
        for node in preprocessing:
            node.upstream = last_node
            last_node = node
            generator = last_node.process()

        return generator

    def _ac_type_to_columns(self, ac_type):
        if ac_type is None:
            return []

        if isinstance(ac_type, list):
            cols2d = [self._ac_type_to_columns(a_t) for a_t in ac_type]
            return list(set(flatten_2d_list(cols2d)))

        check_ac_type(ac_type)
        cols_matching = [col for col in self.available_columns()
                         if col[1] == ac_type]
        return cols_matching

    def _physical_quantity_to_columns(self, physical_quantity):
        if physical_quantity is None:
            return []

        if isinstance(physical_quantity, list):
            cols2d = [self._physical_quantity_to_columns(p_q)
                      for p_q in physical_quantity]
            return list(set(flatten_2d_list(cols2d)))

        check_physical_quantity(physical_quantity)
        cols_matching = [col for col in self.available_columns()
                         if col[0] == physical_quantity]
        return cols_matching

    def _get_columns_with_best_ac_type(self, physical_quantity=None):
        if physical_quantity is None:
            physical_quantity = self.available_physical_quantities()

        if isinstance(physical_quantity, list):
            columns = set()
            for pq in physical_quantity:
                best = self._get_columns_with_best_ac_type(pq)
                if best:
                    columns.update(best)
            return list(columns)

        check_physical_quantity(physical_quantity)
        available_pqs = self.available_physical_quantities()
        if physical_quantity not in available_pqs:
            return []

        ac_types = self.available_ac_types(physical_quantity)
        try:
            best_ac_type = select_best_ac_type(ac_types)
        except KeyError:
            return []
        else:
            return [(physical_quantity, best_ac_type)]

    def _convert_physical_quantity_and_ac_type_to_cols(
            self, physical_quantity=None, ac_type=None, cols=None,
            **load_kwargs):
        """Returns kwargs dict with physical_quantity and ac_type removed
        and cols populated appropriately. Adapts the load_kwargs."""
        if cols:
            if (ac_type or physical_quantity):
                raise ValueError("Cannot use `ac_type` and/or `physical_quantity`"
                                 " with `cols` parameter.")
            else:
                if set(cols).issubset(self.available_columns()):
                    load_kwargs['cols'] = cols
                    return load_kwargs
                elif 'ignore_missing_columns' in load_kwargs and load_kwargs['ignore_missing_columns']:
                    # This option allows to load only the available data
                    load_kwargs['cols'] = list(set(cols).intersection(self.available_columns()))
                    return load_kwargs
                else:
                    msg = ("'{}' is not a subset of the available columns: '{}'"
                           .format(cols, self.available_columns()))
                    raise MeasurementError(msg)

        msg = ""
        if not (ac_type or physical_quantity):
            cols = self.available_columns()
        elif ac_type == 'best':
            cols = self._get_columns_with_best_ac_type(physical_quantity)
            if not cols:
                msg += "No AC types for physical quantity {}".format(physical_quantity)
        else:
            if ac_type:
                cols = self._ac_type_to_columns(ac_type)
                if not cols:
                    msg += "AC type '{}' not available. ".format(ac_type)

            if physical_quantity:
                cols_matching_pq = self._physical_quantity_to_columns(physical_quantity)
                if not cols_matching_pq:
                    msg += ("Physical quantity '{}' not available. "
                            .format(physical_quantity))
                if cols:
                    cols = list(set(cols).intersection(cols_matching_pq))
                    if not cols:
                        msg += ("No measurement matching ({}, {}). "
                                .format(physical_quantity, ac_type))
                else:
                    cols = cols_matching_pq

        if msg:
            msg += "Available columns = {}. ".format(self.available_columns())
            raise MeasurementError(msg)

        load_kwargs['cols'] = cols
        return load_kwargs

    def dry_run_metadata(self):
        '''
        The function which is used by the node system to validate, whether all
        necessary data is available. While nodes insert their postconditions, 
        the elecmeter, which is the source of a pipeline, inserts its metadata.
        '''
        return self.metadata

    def get_metadata(self):
        return self.metadata

    def get_source_node(self, **loader_kwargs):
        '''
        Data processing is always performed via nodes which perform 
        calculations on the data and forward the data to the next node. 
        This function creates an initial node from the datastore which
        delivers the initial data into the stream processing.
        TODO: The datastores could immediatly implement node.
        '''
        if self.store is None:
            raise RuntimeError(
                "Cannot get source node if meter.store is None!")

        loader_kwargs = self._convert_physical_quantity_and_ac_type_to_cols(**loader_kwargs)
        generator = self.store.load(key=self.key, **loader_kwargs)
        self.metadata['device'] = self.device
        return Node(self, generator=generator)
    #endregion


    #region STATS
    def total_energy(self, **loader_kwargs):
        """
        Parameters
        ----------
        full_results : bool, default=False
        **loader_kwargs : key word arguments for DataStore.load()

        Returns
        -------
        if `full_results` is True then return TotalEnergyResults object
        else returns a pd.Series with a row for each AC type.
        """
        nodes = [Clip, TotalEnergy]
        return self._get_stat_from_cache_or_compute(
            nodes, TotalEnergy.results_class(), loader_kwargs)

    def multistattest(self, **loader_kwargs):
        """
        Todo: Not yet implemented

        Parameters
        ----------
        full_results : bool, default=False
        **loader_kwargs : key word arguments for DataStore.load()

        Returns
        -------
        if `full_results` is True then return TotalEnergyResults object
        else returns a pd.Series with a row for each AC type.
        """
        nodes = [Clip, TotalEnergy, GoodSections]
        return self._get_stat_from_cache_or_compute(
            nodes, TotalEnergy.results_class(), loader_kwargs)

    def dropout_rate(self, ignore_gaps=True, **loader_kwargs):
        """
        Parameters
        ----------
        ignore_gaps : bool, default=True
            If True then will only calculate dropout rate for good sections.
        full_results : bool, default=False
        **loader_kwargs : key word arguments for DataStore.load()

        Returns
        -------
        DropoutRateResults object if `full_results` is True,
        else float
        """
        nodes = [DropoutRate]
        if ignore_gaps:
            loader_kwargs['sections'] = self.good_sections(**loader_kwargs)

        return self._get_stat_from_cache_or_compute(
            nodes, DropoutRate.results_class(), loader_kwargs)

    def good_sections(self, **load_kwargs):
        """
        Parameters
        ----------
        full_results : bool, default=False
        **loader_kwargs : key word arguments for DataStore.load()

        Returns
        -------
        if `full_results` is True then return nilmtk.stats.GoodSectionsResults
        object otherwise return list of TimeFrame objects.
        """
        load_kwargs.setdefault('n_look_ahead_rows', 10)
        nodes = [GoodSections]
        results_obj = GoodSections.results_class(self.device['max_sample_period'])
        return self._get_stat_from_cache_or_compute(
            nodes, results_obj, load_kwargs)

    def nonzero_sections(self, **load_kwargs):
        """
        Needs the good sections to be calculated before

        Parameters
        ----------
        full_results : bool, default=False
        **loader_kwargs : key word arguments for DataStore.load()

        Returns
        -------
        if `full_results` is True then return nilmtk.stats.GoodSectionsResults
        object otherwise return list of TimeFrame objects.
        """
                
        nodes = [NonZeroSections]
        results_obj = NonZeroSections.results_class(self.device['max_sample_period'])
        try:
            return self._get_stat_from_cache_or_compute(
                nodes, results_obj, load_kwargs)
        except TypeError as e:
            # Return empty Is a workarround because just don't knwo how to handle
            tmp = NonZeroSections.results_class(self.device['max_sample_period'])
            tmp.finalize()
            return tmp._data

    
    def overbasepower_sections(self, **load_kwargs):
        """
        Needs the good sections to be calculated before

        Parameters
        ----------
        full_results : bool, default=False
        **loader_kwargs : key word arguments for DataStore.load()
        ignore_gaps : bool, default=True
            If True then will only look for overbasepower sections in good sections.

        Returns
        -------
        if `full_results` is True then return nilmtk.stats.GoodSectionsResults
        object otherwise return list of TimeFrame objects.
        """
        
        nodes = [OverBasepowerSections]
        results_obj = OverBasepowerSections.results_class(self.device['max_sample_period'])
        try:
            return self._get_stat_from_cache_or_compute(
                nodes, results_obj, load_kwargs)
        except TypeError as e:
            # Return empty Is a workarround because just don't knwo how to handle
            tmp = OverBasepowerSections.results_class(self.device['max_sample_period'])
            tmp.finalize()
            return tmp._data


    def _get_stat_from_cache_or_compute(self, nodes, results_obj, loader_kwargs):
        """General function for computing statistics and/or loading them from
        cache.

        Cached statistics lives in the DataStore at
        'building<I>/elec/cache/meter<K>/<statistic_name>' e.g.
        'building1/elec/cache/meter1/total_energy'.  We store the
        'full' statistic... i.e we store a representation of the `Results._data`
        DataFrame. Some times we need to do some conversion to store
        `Results._data` on disk.  The logic for doing this conversion lives
        in the `Results` class or subclass.  The cache can be cleared by calling
        `ElecMeter.clear_cache()`.
        When 'preprocessing' is set, then the cache is not used because the cache
        is only valid for the version without preprocessing.

        Parameters
        ----------
        nodes : list of nilmtk.Node classes
        results_obj : instance of nilmtk.Results subclass. This is THE result 
                      instance which is afterwards filled by all the results
                      coming from the different chunks.
        loader_kwargs : dict

        Returns
        -------
        if `full_results` is True then return nilmtk.Results subclass
        instance otherwise return nilmtk.Results.simple().

        See Also
        --------
        clear_cache
        _compute_stat
        key_for_cached_stat
        get_cached_stat
        """
        full_results = loader_kwargs.pop('full_results', False)
        verbose = loader_kwargs.get('verbose')
        if 'ac_type' in loader_kwargs or 'physical_quantity' in loader_kwargs:
            loader_kwargs = self._convert_physical_quantity_and_ac_type_to_cols(**loader_kwargs)
        cols = loader_kwargs.get('cols', [])
        ac_types = set([m[1] for m in cols if m[1]])
        results_obj_copy = deepcopy(results_obj)

        # Prepare `sections` list
        sections = loader_kwargs.get('sections')
        if sections is None:
            tf = self.get_timeframe()
            tf.include_end = True
            sections = [tf]
        sections = TimeFrameGroup(sections) # Takes care that NILMTK timeframe
        sections = [s for s in sections if not s.empty]

        # Retrieve usable stats from cache
        key_for_cached_stat = self.key_for_cached_stat(results_obj.name)
        cached_stat = None
        if loader_kwargs.get('preprocessing') is None:
            cached_stat = self.get_cached_stat(key_for_cached_stat)
            #results_obj.import_from_cache(cached_stat, sections) # Fill results_obj with cache

            #def find_sections_to_compute():
            #    # Get sections_to_compute
            #    results_obj_timeframes = results_obj.timeframes()
            #    sections_to_compute = set(sections) - set(results_obj_timeframes)
            #    t1 = TimeFrameGroup(sections)
            #    t2 = TimeFrameGroup(results_obj_timeframes)
            #    sections_to_compute = t1.diff(t2)   # HIER IST DAS DIFF, DAS ICH NEU GEBAUT HABE!!! NUR WARUM GEHT ES GERADE NICHT MEHR???
            #    sections_to_compute = sorted(sections_to_compute)
            #    return sections_to_compute
            #try:
            #    ac_type_keys = results_obj.keys() #.simple().keys()
            #except:
            #    sections_to_compute = find_sections_to_compute()
            #else:
            #    if ac_types.issubset(ac_type_keys):
            #        # IF ac_type in cache, only calculate remaining sections
            #        sections_to_compute = find_sections_to_compute()
            #    else:
            #        # If false ac_type cached, still have to compute all
            #        sections_to_compute = sections
            #        results_obj = results_obj_copy
        #else:
        #    sections_to_compute = sections
        if verbose and not cached_stat is None: #._data.empty:
            print("Using cached result.")
            
        # If necessary compute stats for missing sections 
        if cached_stat is None: #sections_to_compute:
            # If we need everything either way, we don't need expensive index lookup during load
            #if not self.get_timeframe() in sections_to_compute:
            #    loader_kwargs['sections'] = sections_to_compute

            #computed_result = self._compute_stat(nodes, loader_kwargs)

            # Merge newly computed stats into the main stat result         
            # DAS HIER BAUT MAN BESSER DIREKT IN DEN NODE EIN!!! DASS SEIN RESULT ERWEITERT WIRD
            # DANN KANN MAN IMMER NOCH DAS RESULT VOM CACHING NEHMEN UND VERBINDEN!
            # MAN SETZT DANN VON ALLEN NODE ELEMENTEN DAS RESULT MIT IN DIE PIPELINE ELEMENTE! 
            # DANN KANN MAN SIE HINTEN RAUSHOLEN. SO KANN MAN DIE BERECHNUNG IN EINEM ZUG MACHEN.  
            # => Ist ja so gemacht. Nur eben fuer jede Section!
            # => Die einzige rweiterung waere das durchreichen von Results.                                                                 
            #results_obj.update(computed_result.results)
            
            results_obj = self._compute_stat(nodes, loader_kwargs).results
            
            # For Nonzero section exclude where there are not good sections
            if  results_obj.name == 'nonzero_sections' or results_obj.name == 'overbasepower_sections':
                good_sections = self.good_sections(**loader_kwargs) #_data
                results_obj._data = results_obj._data.intersection(good_sections)

            # Save to disk newly computed stats
            stat_for_store = results_obj.export_to_cache()
            try:
                #self.store.remove(key_for_cached_stat)
                self.store.put(key_for_cached_stat, stat_for_store, fixed = True)
                # Temporary workarround to store the good sections also for the other meters
                if results_obj.name == 'good_sections':
                    for i in range(2,4):
                        self.store.put(key_for_cached_stat.replace('meter1', 'meter' +str(i)), stat_for_store, fixed = True) 

            except ValueError:
                # the old table probably had different columns
                self.store.remove(key_for_cached_stat)
                self.store.put(key_for_cached_stat, results_obj.export_to_cache())
        else:
            results_obj.import_from_cache(cached_stat, sections) # Fill results_obj with cache

        # Return the correct value depending on options
        if full_results:
            return results_obj
        res = results_obj #.simple()
        if ac_types:
            try:
                ac_type_keys = res.keys()
            except:
                return res
            else:
                return pd.Series(res[ac_types], index=ac_types)
        return res._data

    def _compute_stat(self, nodes, loader_kwargs):
        """
        This function computes the stats by attaching the  
        nodes from the nodes list paramter to each other and 
        calculating the whole pipeline.
        RETURNS THE NODE, NOT THE RESULT CLASS!!!

        Parameters
        ----------
        nodes : list of nilmtk.Node subclass objects
        loader_kwargs : dict

        Returns
        -------
        Node subclass object

        See Also
        --------
        clear_cache
        _get_stat_from_cache_or_compute
        key_for_cached_stat
        get_cached_stat
        """
        last_node = self.get_source_node(**loader_kwargs)
        for node in nodes:
            last_node = node(last_node)
        last_node.run()
        last_node.results.finalize()
        return last_node

    def key_for_cached_stat(self, stat_name):
        """
        This function puts together the key, where the 
        cached stat can be found in the HDF hierarchy.

        Parameters
        ----------
        stat_name : str

        Returns
        -------
        key : str

        See Also
        --------
        clear_cache
        _compute_stat
        _get_stat_from_cache_or_compute
        get_cached_stat
        """
        if isinstance(self.instance(), tuple):
            meter_str = "_".join([str(i) for i in (self.instance())])
        else:
            meter_str = "{:d}".format(self.instance())

        relative_path = ("building{:d}/elec/cache/meter{}/{:s}"
                .format(self.building(), meter_str, stat_name))
        #absolute_path = os.path.dirname(self.store.filename) + relative_path
        #os.path.normpath(absolute_path) # clean for os
        return relative_path

    def calc_and_cache_stats(self, verbose = False, ):
        '''
        This function calls all available status calculation functions sothat 
        the results are placed inside the stats-buffer. 
        This comes handy when your want to preprocess the dataset in the
        beginning sothat you can access the values in a fast way afterwards.
        '''

        try:
            self.good_sections(chunksize=100000000, verbose = verbose)
            self.nonzero_sections(chunksize=100000000, verbose = verbose)
        except Exception as e:
            print("STILL BROKEN")

        # except Exception as e:
        #     try:
        #         self.nonzero_sections(chunksize=100000000, verbose = verbose)
        #         self.clear_cache(verbose = True)
        #         self.good_sections(chunksize=100000000, verbose=verbose)
        #         self.nonzero_sections(chunksize=100000000, verbose = verbose)
        #     except Exception as e:
        #         print("STILL BROKEN")
        #self.total_energy(chunksize=1000000, verbose = verbose)
        #self.dropout_rate(verbose=False)#, chunksize=5000000)
        #tst2 = tst.invert()
        #tst3 = tst2.remove_shorter_than(60)
        #tst3.plot()
        #i = 0



    def clear_cache(self, caches_to_delete = None, verbose=False):
        """
        See Also
        --------
        _compute_stat
        _get_stat_from_cache_or_compute
        key_for_cached_stat
        get_cached_stat
        """
        if caches_to_delete is None:
            caches_to_delete = ['overbasepower_sections', 'nonzero_sections', 'good_sections', 'total_energy']
        elif not isinstance(caches_to_delete, list):
            caches_to_delete = list(caches_to_delete)
        if self.store is not None:
            for cache in caches_to_delete:
                key_for_cache = self.key_for_cached_stat(cache)
                try:
                    self.store.remove(key_for_cache)
                except KeyError:
                    if verbose:
                        print("No existing cache for", key_for_cache)
                else:
                    print("Removed", key_for_cache)

    def get_cached_stat(self, key_for_stat):
        """
        Tries to load the cache from the store. Returns 
        empty DataFrame if not available.

        Parameters
        ----------
        key_for_stat : str

        Returns
        -------
        pd.DataFrame

        See Also
        --------
        _compute_stat
        _get_stat_from_cache_or_compute
        key_for_cached_stat
        clear_cache
        """
        if self.store is None:
            return #pd.DataFrame()
        try:
            stat_from_cache = self.store[key_for_stat]
            #stat_from_cache = pd.load(key_for_stat)
        except KeyError:
            return None #pd.DataFrame()
        else:
            return pd.DataFrame() if stat_from_cache is None else stat_from_cache
    #endregion


    #region FUTURE PROPOSALS
    # def total_on_duration(self):
    #     """Return timedelta"""
    #     raise NotImplementedError

    # def on_durations(self):
    #     raise NotImplementedError

    # def activity_distribution(self, bin_size, timespan):
    #     raise NotImplementedError

    # def on_off_events(self):
    # use self.metadata.minimum_[off|on]_duration
    #     raise NotImplementedError

    # def discrete_appliance_activations(self):
    #     """
    #     Return a Mask defining the start and end times of each appliance
    #     activation.
    #     """
    #     raise NotImplementedError

    # def contiguous_sections(self):
    #     """retuns Mask object"""
    #     raise NotImplementedError

    # def clean_and_export(self, destination_datastore):
    #     """Apply all cleaning configured in meter.cleaning and then export.  Also identifies
    #     and records the locations of gaps.  Also records metadata about exactly which
    #     cleaning steps have been executed and some summary results (e.g. the number of
    #     implausible values removed)"""
    #     raise NotImplementedError
    #endregion