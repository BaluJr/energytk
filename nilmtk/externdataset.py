from __future__ import print_function, division
import os
from collections import OrderedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from six import iteritems
from .building import Building
from nilmtk import MeterGroup
from .datastore.datastore import join_key
from nilmtk import DataSet

# AHH! Die Klasse hatte ich ja zu erst fuer die Prediction gebaut

class ExternDataSet(DataSet):
    """
    This is just like the normal dataset but extends it by the simple functionality to detect the correct 
    dataset for a certain metering device. 
    Each ExternDataSet has stored in its metadata under which property its data is referencable. Eg. zip.
    When a meter is then handed in, it automatically determines the best dataset.
    """
        
    def get_data_for_meter(self, meter):
        """ Returns the meter, which contains the fitting data 
        for the building. External Data is aways related to buildings not measurements as 
        things which are that specific that they belong to the meter would be counted as 
        measurement data itself.
        AT THE MOMENT ALWAYS JUST LOOKING FOR THE FIRST THREE NUMBERS OF THE ZIP BECAUSE 
        ONLY THING I NEED AT THE MOMENT.

        Parameters
        ----------
        meter : ElecMeter
            The ElecMeter to gather external data for.
        """

        zip = meter.buildingmetadata()['zip']
        group = self.buildings[grouping_variable] 
        meters = group.elec
        timewindow = meter.metadata.get_timewindow()        
        return meters.load(cols = variables, ignore_missing_columns = True, timewindow=timewindow)


    def get_data_for_elec(self, elec):
        '''
        This function returns the data for the given input elec. (Meter or MeterGroup)
        In the future I should cluster by the zips. But now I take the very first one
        '''
        if isinstance(elec, MeterGroup):
            elec = elec.meters[0]
        zip = elec.building_metadata['zip']
        return self.buildings[zip] 


    
    def get_data_for_group(self, grouping_variable, timeframe, sample_period, variables):
        '''
        In contrast to get data for meter, this function loads the whole load.
        This is usefull when the data is shared for calculations of multiple
        elec meters.
        Da ich im Moment noch auf den ElecMetern basiere könnte ich das auch ohne 
        die Funktion hier erreichen, indem ich einfach das Building raussuche.

        grouping_variable: str
            The variable after which the dataset is ordered. Would be the building number 
            in the default meter datasets. For external data it is for example the zip.
        variables:
            The variables which shall be loaded from the dataset.
        '''
        output = pd.DataFrame(columns=variables)
        group = self.buildings[grouping_variable]   # zip
        meters = group.elec                         # weather + holidays
        for chunk in meters.load(cols = variables, sections = {timeframe}, sample_period = sample_period, ignore_missing_columns = True, chunksize = 100000):
            output = output.append(chunk)

        return output