from __future__ import print_function, division
import pandas as pd
import numpy as np
from collections import Counter
from builtins import zip
from warnings import warn
import scipy.spatial as ss
from scipy import fft
from pandas.tools.plotting import lag_plot, autocorrelation_plot
from scipy.special import digamma,gamma
from math import log,pi
import numpy.random as nr
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta
import gc
import pytz

from .timeframe import TimeFrame
from .measurement import select_best_ac_type
from .utils import (offset_alias_to_seconds, convert_to_timestamp,
                    flatten_2d_list, append_or_extend_list,
                    timedelta64_to_secs, safe_resample)
from .plots import plot_series
from .preprocessing import Apply
from nilmtk.stats.histogram import histogram_from_generator
from nilmtk.appliance import DEFAULT_ON_POWER_THRESHOLD

MAX_SIZE_ENTROPY = 10000

class MeterSeries(object):
    """ This is a superclass from which all different meters inherit from.
    It wraps the pandas series by additional out-of-memory support
    and some usefull functions. This class is inherited from the different
    types of data: elec, weather, holidays and whatever else.
    The implementing classes also introduce metadata etc.
    """
    

    def load_series(self, **load_kwargs):
        """
        Parameters
        ----------
        ac_type : str
        physical_quantity : str
            We sum across ac_types of this physical quantity.
        **load_kwargs : passed through to load().

        Returns
        -------
        generator of pd.Series.  If a single ac_type is found for the
        physical_quantity then the series.name will be a normal tuple.
        If more than 1 ac_type is found then the ac_type will be a string
        of the ac_types with '+' in between.  e.g. 'active+apparent'.
        """
        # Pull data through preprocessing pipeline
        physical_quantity = load_kwargs['physical_quantity']
        generator = self.load(**load_kwargs)
        for chunk in generator:
            if chunk.empty:
                yield chunk
                continue
            chunk_to_yield = chunk[physical_quantity].sum(axis=1)
            value_types = '+'.join(chunk[physical_quantity].columns)
            chunk_to_yield.name = (physical_quantity, value_types) #value_types has been ac_types in the past
            chunk_to_yield.timeframe = getattr(chunk, 'timeframe', None)
            chunk_to_yield.look_ahead = getattr(chunk, 'look_ahead', None)
            yield chunk_to_yield


             
    
           
    def load(self, **load_kwargs):
        '''
        This load function has to be overwritten by inheriting classes.
        '''
        raise Exception('Each non-abstract class, which inherits meterseries needs '
                         'to implement the load function.')