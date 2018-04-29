from __future__ import print_function, division
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
from ..results import Results
from nilmtk.timeframe import TimeFrame, convert_none_to_nat, convert_nat_to_none
from nilmtk.utils import get_tz, tz_localize_naive
from nilmtk.timeframegroup import TimeFrameGroup
import numpy as np

class OverBasepowerSectionsResults(Results):
    """ The result of the Non zero section statistic.
    Attributes
    ----------
    _data : pd.DataFrame
        index is start date for the whole chunk
        `end` is end date for the whole chunk
        `sections` is a TimeFrameGroups object (a list of nilmtk.TimeFrame objects)
    """
    
    name = "overbasepower_sections"

    def __init__(self, max_sample_rate):
        # Used to know when to combine
        self.max_sample_rate = max_sample_rate
        super(OverBasepowerSectionsResults, self).__init__()

    def append(self, timeframe, new_results):
        """Append a single result.

        Parameters
        ----------
        timeframe : nilmtk.TimeFrame
        new_results : {'sections': list of TimeFrame objects}
        """
        super(OverBasepowerSectionsResults, self).append(timeframe, new_results)

    def finalize(self):
        """ Merges together any nonzero sections which span multiple segments.
        Whether there are gaps in between does not matter.

        Returns
        -------
        sections : TimeFrameGroup (a subclass of Python's list class)
        """

        # Merge the results of all chunks
        starts = []
        ends = []
        for index, row in self._data.iterrows():
            starts.append(row['sections']['start'])
            ends.append(row['sections']['end'])

        if len(starts) == 0 == len(ends):
            self._data = TimeFrameGroup()
            return

        starts = pd.concat(starts)
        ends = pd.concat(ends)
        
        rate = pd.Timedelta(seconds=self.max_sample_rate)
        self._data = TimeFrameGroup(starts_and_ends={'starts': starts, 'ends': ends})#.merge_shorter_gaps_than(rate) TODO: Merge needed?


    def unify(self, other):
        raise Exception("Did not try this yet for the new nonzeroresults")
        super(OverBasepowerSectionsResults, self).unify(other)
        for start, row in self._data.iterrows():
            other_sections = other._data['sections'].loc[start]
            intersection = row['sections'].intersection(other_sections)
            self._data['sections'].loc[start] = intersection


    def to_dict(self):
        overbasepower_sections = self._data
        overbasepower_sections_list_of_dicts = [timeframe.to_dict() 
                                       for timeframe in overbasepower_sections]
        return {'statistics': {'overbasepower_sections': overbasepower_sections_list_of_dicts}}


    def plot(self, **plot_kwargs):
        timeframes = self
        return timeframes.plot(**plot_kwargs)

        
    def import_from_cache(self, cached_stat, sections):   
        ''' Stores the statistic into the cache of the nilmtk.DataStore

        Note
        ----
        I do not know whether this is still an issue: 
        HIER IST DAS PROBLEM BEIM STATISTIKEN LESEN! 
        DIE WERDEN CHUNK Weise GESPEICHERT, aber hier wird auf 
        das Vorhandensein der gesamten Section als ganzes vertraut.
        '''
        self._data = TimeFrameGroup(cached_stat)


    def export_to_cache(self):
        """
        Returns the DataFrame to be written into cache.

        Returns
        -------
        df: pd.DataFrame
            With three columns: 'end', 'section_end', 'section_start.      
        """
        return self._data._df