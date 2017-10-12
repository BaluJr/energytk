from __future__ import print_function, division
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
from ..results import Results
from nilmtk.timeframe import TimeFrame, convert_none_to_nat, convert_nat_to_none
from nilmtk.utils import get_tz, tz_localize_naive
from nilmtk.timeframegroup import TimeFrameGroup
import numpy as np

class NonZeroSectionsResults(Results):
    """
    Attributes
    ----------
    _data : pd.DataFrame
        index is start date for the whole chunk
        `end` is end date for the whole chunk
        `sections` is a TimeFrameGroups object (a list of nilmtk.TimeFrame objects)
    """
    
    name = "nonzero_sections"

    def __init__(self, max_sample_rate):
        # Used to know when to combine
        self.max_sample_rate = max_sample_rate
        super(NonZeroSectionsResults, self).__init__()

    def append(self, timeframe, new_results):
        """Append a single result.

        Parameters
        ----------
        timeframe : nilmtk.TimeFrame
        new_results : {'sections': list of TimeFrame objects}
        """
        #new_results['sections'] = [TimeFrameGroup(new_results['sections'][0])]
        super(NonZeroSectionsResults, self).append(timeframe, new_results)

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
        
        # Check whether something has to be added in between or before
        # if len(starts) == 0 == len(ends):
        #     self._data = TimeFrameGroup()
        #     return
        # elif len(starts) == 0:
        #     starts = np.array([self._data.head(1)['start'][0]])
        # elif len(ends) == 0:
        #     ends = np.array([self._data.tail(1)['end'][0]])
        # else:
        #     if starts[0] > ends[0]:
        #         starts = np.append(np.datetime64(self._data.index[0]), starts)
        #     if ends[-1] < starts[-1]:
        #         ends = np.append(ends, np.datetime64(self._data.tail(1)['end'][0]))
        rate = pd.Timedelta(seconds=self.max_sample_rate)
        self._data = TimeFrameGroup(starts_and_ends={'starts': starts, 'ends': ends}).merge_shorter_gaps_than(rate)


    def unify(self, other):
        raise Exception("Did not try this yet for the new nonzeroresults")
        super(NonZeroSectionsResults, self).unify(other)
        for start, row in self._data.iterrows():
            other_sections = other._data['sections'].loc[start]
            intersection = row['sections'].intersection(other_sections)
            self._data['sections'].loc[start] = intersection


    def to_dict(self):
        nonzero_sections = self._data #.combined()
        nonzero_sections_list_of_dicts = [timeframe.to_dict() 
                                       for timeframe in nonzero_sections]
        return {'statistics': {'nonzero_sections': nonzero_sections_list_of_dicts}}


    def plot(self, **plot_kwargs):
        timeframes = self#.combined()
        return timeframes.plot(**plot_kwargs)

        
    def import_from_cache(self, cached_stat, sections):   
        # HIER IST DAS PROBLEM BEIM STATISTIKEN LESEN! DIE WERDEN CHUNK Weise GESPEICHERT, aber hier wird auf das Vorhandensein der gesamten Section als ganzes vertraut
        '''
        As explained in 'export_to_cache' the sections have to be stored 
        rowwise. This function parses the lines and rearranges them as a 
        proper NonZeroSectionsResult again.
        '''
        self._data = TimeFrameGroup(cached_stat)

        # we (deliberately) use duplicate indices to cache NonZeroSectionResults
        #grouped_by_index = cached_stat.groupby(level=0)
        #tz = get_tz(cached_stat)
        #for tf_start, df_grouped_by_index in grouped_by_index:
        #    grouped_by_end = df_grouped_by_index.groupby('end')
        #    for tf_end, sections_df in grouped_by_end:
        #        end = tz_localize_naive(tf_end, tz)
        #        timeframe = TimeFrame(tf_start, end)
        #        if any([section.contains(timeframe) for section in sections]): # Had to adapt this, because otherwise no cache use when loaded in chunks
        #            timeframes = []
        #            for _, row in sections_df.iterrows():
        #                section_start = tz_localize_naive(row['section_start'], tz)
        #                section_end = tz_localize_naive(row['section_end'], tz)
        #                timeframes.append(TimeFrame(section_start, section_end))
        #            self.append(timeframe, {'sections': [timeframes]})

    def export_to_cache(self):
        """
        Returns the DataFrame to be written into cache.

        Returns
        -------
        DataFrame with three columns: 'end', 'section_end', 'section_start'.
            Instead of storing a list of TimeFrames on each row,
            we store one TimeFrame per row.  This is because pd.HDFStore cannot
            save a DataFrame where one column is a list if using 'table' format'.
            We also need to strip the timezone information from the data columns.
            When we import from cache, we assume the timezone for the data 
            columns is the same as the tz for the index.
        """
        return self._data._df
        #index_for_cache = []
        #data_for_cache = [] # list of dicts with keys 'end', 'section_end', 'section_start'
        #for index, row in self._data.iterrows():
        #    for section in row['sections']:
        #        index_for_cache.append(index)
        #        data_for_cache.append(
        #            {'end': row['end'], 
        #             'section_start': convert_none_to_nat(section.start),
        #             'section_end': convert_none_to_nat(section.end)})
        #df = pd.DataFrame(data_for_cache, index=index_for_cache)
        #return df.convert_objects()
