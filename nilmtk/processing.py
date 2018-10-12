from nilmtk.elecmeter import ElecMeter



class Processing(object):
    """This is the baseclass for all predictors which process ElecMeters.
    At the moment disaggregator, clustering and forecasting inherits from this class.

    Do not confuse it with the node system which is only used for calculating the stats.
    """

    # This has to be overwritten by the subclasses sothat one can check whether 
    # the input elecmeter fullfills the necessary data.
    Requirements = {}
    
    def _pre_disaggregation_checks(self, site_meters, load_kwargs):
        '''
        This is the basic check, which is called before disaggregation is performed.
        It takes care.

        site_meters: A Group of site meters or a single site-meter 
        '''

        if site_meters is ElecMeter:
            self._check_meter(site_meters)
            if not site_meters.metadata['site_meter']:
                raise RuntimeError("Only site meters can be disaggregated")
        else:
            for meter in site_meters.all_elecmeters():
                self._check_meter(meter)
                if not ('site_meter' in meter.metadata and meter.metadata['site_meter']):
                    raise RuntimeError("Only site meters can be disaggregated")

        if 'resample_seconds' in load_kwargs:
            DeprecationWarning("'resample_seconds' is deprecated."
                               "  Please use 'sample_period' instead.")
            load_kwargs['sample_period'] = load_kwargs.pop('resample_seconds')

        return load_kwargs


    def _check_meter(self, meter):
        '''
        This functin checks, whether the given data fullfills the requirements 
        of the disaggregator. It uses the metadata of the measurement devices
        to do this.
        '''
    
        if meter.device['sample_period'] > self.Requirements['max_sample_period']:
            raise RuntimeError(
                "The data you provided does not meet the requirements for"
                " the chosen disaggregator."
                " The sample period has to be below " +
                str(self.Requirements['max_sample_period']) +
                ". But it is " + str(device.sample_period) + ".")

        required = set([cur[0] + cur[1] for cur in self.Requirements['physical_quantities']])
        measurements = set([cur['physical_quantity'] + cur['type'] if 'type' in cur else "" for cur in meter.device['measurements']])
        missing_quantities = required.difference(measurements)
        if len(missing_quantities) > 0:
            raise RuntimeError(
                "The data you provided does not meet the requirements for"
                " the chosen disaggregator."
                " The following quantities are missing: " + ', '.join(missing_quantities))






