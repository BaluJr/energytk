from __future__ import print_function, division
import os
from collections import OrderedDict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from six import iteritems
from .building import Building
from .datastore.datastore import join_key
from nilmtk import DataSet


class ExternDataSet(DataSet):
    """
    This is just like the normal dataset but extends it by the simple functionality to detect the correct 
    dataset for a certain metering device. 
    Each ExternDataSet has stored in its metadata under which property its data is referencable. Eg. zip.
    When a meter is then handed in, it automatically determines the best dataset.
    """

    
    def get_data_for_meter(self, building):
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

        zip = building.metadata['zip']
        # Ich muss hier auf jedne Fall noch das Timewindow irgendwo setzen, da sonst die externen Daten ggf VIEL mehr sind.
        return self.buildings[zip]

