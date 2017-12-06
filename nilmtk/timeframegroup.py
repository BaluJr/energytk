from __future__ import print_function, division
#from nilmtk.stats import intersect_many_fast
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
import matplotlib.dates as mdates
from copy import deepcopy
import numpy as np

# NILMTK imports
from nilmtk.consts import SECS_PER_DAY
from nilmtk.timeframe import TimeFrame, convert_none_to_nat


class TimeFrameGroup():
    """A collection of nilmtk.TimeFrame objects.
    The timeframegroup is used to store TimeFrames of a certain
    type (eg. good sections) for a whole load profile together.
    It then allows intersection functionality between multiple
    load profiles to eg. find the good timeframes in all 
    the TimeFrameGroups.

    The TimeFrameGroup has been rewritten using pandas DataFrames 
    because the previous implementation was far to slow

    _df: The dataframe with the sec_start sec_end [start_time, end_time]
    """

    def __init__(self, timeframes=None, starts_and_ends = None):
        if isinstance(timeframes, TimeFrameGroup):
            self._df = timeframes._df
        if isinstance(timeframes, pd.core.indexes.datetimes.DatetimeIndex):
            self._df = timeframes
            #periods = timeframes
            #timeframes = [TimeFrame(period.start_time, period.end_time)
            #              for period in periods]
        elif isinstance(timeframes, pd.DataFrame):
            self._df = timeframes.copy() 
        elif not starts_and_ends is None:
            self._df = pd.DataFrame({'section_start': starts_and_ends['starts'],  'section_end': starts_and_ends['ends']})
        elif not timeframes is None:
            self._df = pd.DataFrame([(frame.start, frame.end) for frame in timeframes], columns = ['section_start', 'section_end'])
        else:
            self._df = pd.DataFrame(columns = ['section_start', 'section_end'])
        
        #self._df.sort_values('section_start', inplace = True)
        #args = [timeframes] if timeframes else []
        #super(TimeFrameGroup, self).__init__(*args)

    def plot(self, ax=None, y=0, height=1, gap=0.05, color='b', **plot_kwargs):
        if ax is None:
            ax = plt.gca()
        ax.xaxis.axis_date()
        height -= gap * 2
        for _, row in self._df.iterrows():
            length = (row['section_end'] - row['section_start']).total_seconds() / SECS_PER_DAY
            bottom_left_corner = (mdates.date2num(row['section_start']), y + gap)
            rect = plt.Rectangle(bottom_left_corner, length, height,
                                 color=color, **plot_kwargs)
            ax.add_patch(rect)

        ax.autoscale_view()
        return ax

    
    def plot_simple(self, ax=None, gap=0.05, **plot_kwargs):
        for _, row in self._df.iterrows():
            length = (row['section_end'] - row['section_start']).total_seconds() / SECS_PER_DAY
            bottom_left_corner = (mdates.date2num(row['section_start']), 0)
            rect = plt.Rectangle(bottom_left_corner, length, 1,
                                 color='b', **plot_kwargs)
            ax.add_patch(rect)
        return ax

    
    def plot_deltahistogram(self, bins = 10):
        (self._df['section_end'] - self._df['section_start']).apply(lambda e: e.total_seconds()).hist(bins=bins)



    def get_timeframe(self):
        '''
        Returns the outer timeframe of this TimeFrameGroup
        '''
        return TimeFrame(start = self._df.iloc[0, 'section_start'], end = self._df.iloc[0, 'section_end'])


    def union(self, other):
        '''
        self.good_sections():   |######----#####-----######-#|
        other.good_sections():  |---##---####----##-----###-#|
        diff():                 |######--#######-##--######-#|
        '''
        assert isinstance(other, (TimeFrameGroup, list))
        return TimeFrameGroup.union_many([self, other])

    def union_many(groups):
        ''' 
        Function to do a do a fast intersection between many timeframes
        '''
        all_events = pd.Series()
        for group in groups:
            all_events = all_events.append(pd.Series(1, index=pd.DatetimeIndex(group._df['section_start'])))
            all_events = all_events.append(pd.Series(-1, index=pd.DatetimeIndex(group._df['section_end'])))
        all_events.sort_index(inplace=True)
        any_active = (all_events.cumsum()>0).astype(int)

        switches = (any_active - any_active.shift(1).fillna(0))
        starts = all_events[switches == 1].index
        ends = all_events[switches == -1].index
        result = pd.DataFrame({'section_start': starts, 'section_end':ends})
        return TimeFrameGroup(result)


    def diff(self, other):
        '''
        Difference between this and the other TimeFrameGroup.

        self.good_sections():   |######----#####-----######-#|
        other.good_sections():  |---##---####----##-----###-#|
        diff():                 |###--#------###-----###-----|
        '''
        assert isinstance(other, (TimeFrameGroup, list))
        
        all_events = pd.Series()
        all_events = all_events.append(pd.Series(1, index=pd.DatetimeIndex(self._df['section_start'])))
        all_events = all_events.append(pd.Series(-1, index=pd.DatetimeIndex(self._df['section_end'])))
        all_events = all_events.append(pd.Series(-1, index=pd.DatetimeIndex(other._df['section_start'])))
        all_events = all_events.append(pd.Series(+1, index=pd.DatetimeIndex(other._df['section_end'])))
        all_events.sort_index(inplace=True)

        all_active = (all_events.cumsum()>0)
        starts = all_events.index[all_active]
        ends = all_active.shift(1)
        if len(ends > 0):
            ends[0] = False
        ends = all_events[ends].index
        result = pd.DataFrame({'section_start': starts, 'section_end':ends})
        return TimeFrameGroup(result)



    def intersection(self, other):
        """Returns a new TimeFrameGroup of self masked by other.

        Illustrated example:

        self.good_sections():   |######----#####-----######-#|
        other.good_sections():  |---##---####----##-----###-#|
               intersection():  |---##-----##-----------###-#|
        """
        assert isinstance(other, (TimeFrameGroup, list))
        return TimeFrameGroup.intersect_many([self, other])

    
    def intersect_many(groups):
        ''' 
        Function to do a do a fast intersection between many timeframes
        '''
        all_events = pd.Series()
        for group in groups:
            all_events = all_events.append(pd.Series(1, index=pd.DatetimeIndex(group._df['section_start'])))
            all_events = all_events.append(pd.Series(-1, index=pd.DatetimeIndex(group._df['section_end'])))
        all_events.sort_index(inplace=True)
        all_active = (all_events.cumsum()==len(groups))
        starts = all_events.index[all_active]
        ends = all_active.shift(1)
        if len(ends > 0):
            ends[0] = False
        ends = all_events[ends].index
        result = pd.DataFrame({'section_start': starts, 'section_end':ends})
        return TimeFrameGroup(result)



    def matching(self, other):  #, valid_timeframes = None, in_percent = False):
        '''
        Calculates the matching of two timeframegroups.
        These are the timeframes where both are on or off.
        If given, excluded timeframes are calculated out. This is usefull when there 
        are eg. notgood sections.
        
        self.good_sections():   |######----#####-----######-#|
        other.good_sections():  |---##---####----##-----###-#|
        matching():             |---##-##--##---#--##---#####|

        Paramters:
        other: The other timeframe to match with
        valid_timeframes: TimeFrameGroup which defines the area for which to do the calculation
        in_percent: Whether the amount of matched time shall be returned as fraction of whole valid timespan.
                    (takes into account the "excluded_timeframes")
        '''
        
        assert isinstance(other, (TimeFrameGroup, list))
        return TimeFrameGroup.matching_many([self, other])

        #common_on = self.intersection(other)
        #self_off = self.invert(start=first, end=last)
        #other_off = other.invert(start=first, end=last)
        #common_off = self_off.intersection(other_off)

        #if in_percent:
        #    valid_time = self.full_time() - excluded_timeframes.uptime()
        #    percent = common.uptime() / valid_time
        #    return percent
        #else:
        #    return common

    
    def matching_many(groups):
        ''' 
        Function to do a do a fast matching between many timeframes
        '''
        all_events = pd.Series()
        for group in groups:
            all_events = all_events.append(pd.Series(1, index=pd.DatetimeIndex(group._df['section_start'])))
            all_events = all_events.append(pd.Series(-1, index=pd.DatetimeIndex(group._df['section_end'])))
        all_events.sort_index(inplace=True)
        all_events_sum = all_events.cumsum()
        all_active = ((all_events_sum==len(groups)) | (all_events_sum == 0))
        # Remove last, which is always created after end of all sections
        starts = all_events.index[all_active][:-1] 
        ends = all_active.shift(1)
        if len(ends > 0):
            ends[0] = False
        ends = all_events[ends].index
        result = pd.DataFrame({'section_start': starts, 'section_end':ends})
        return TimeFrameGroup(result).simplify()




    def uptime(self):
        """
        Returns total timedelta of all timeframes joined together.
        """
        return (self._df['section_end'] - self._df['section_start']).sum()



    def remove_shorter_than(self, threshold):
        """
        Removes TimeFrames shorter than `threshold` seconds.
        """
        return TimeFrameGroup(self._df[self._df['section_end']-self._df['section_start'] > threshold])



    def merge_shorter_gaps_than(self, threshold):
        if self._df.empty:
            return TimeFrameGroup()

        if isinstance(threshold, str):
            threshold = pd.Timedelta(threshold)

        gap_larger = ((self._df["section_start"].shift(-1) - self._df["section_end"]) > threshold)
        gap_larger.iloc[-1] = True # keep last
        relevant_starts = self._df[["section_start"]][gap_larger.shift(1).fillna(True)].reset_index(drop=True)
        relevant_ends = self._df[["section_end"]][gap_larger].reset_index(drop=True)
        return TimeFrameGroup(pd.concat([relevant_starts, relevant_ends], axis=1))


    def simplify(self):
        '''
        Merges all sections, which are next to each other.
        Another view: Removes gaps of zero length.
        '''
        to_keep = (self._df["section_start"].shift(-1) != self._df["section_end"])
        #to_keep.iloc[-1] = True # keep last
        relevant_starts = self._df[["section_start"]][to_keep.shift(1).fillna(True)].reset_index(drop=True)
        relevant_ends = self._df[["section_end"]][to_keep].reset_index(drop=True)
        return TimeFrameGroup(pd.concat([relevant_starts, relevant_ends], axis=1))




    def remove_before(self, start):
        self._df  = self._df[self._df["section_end"] > start]
        self._df[self._df["section_start"] < start]["section_start"] = start


    def invert(self, start = None, end = None, outer_timeframe = None):
        ''' 
        Returns a TimeFrameGroup with inverted rectangles.
        That means where there was a gap before is now a 
        TimeFrame and vice versa.

        Paramter:
        start, end: Timestamps defining the start and end of the region to invert
        outer_timeframe: 
        '''
        if self._df.empty:
            if not start is None and not end is None:
                return TimeFrameGroup([TimeFrame(start=start, end=end)])
            return TimeFrameGroup()

        inversion = self._df.copy()
        if self._df.iloc[-1,:]["section_end"] < end:
            val_to_append = self._df.iloc[-1,:]["section_start"]
            inversion['section_end'] = inversion['section_end'].shift(1)
            row = len(inversion)
            inversion.loc[row, :] = [start, start]
            inversion.loc[row, 'section_start'] = end
            inversion.loc[row, 'section_end'] = val_to_append

        else:
            inversion['section_end'] = inversion['section_end'].shift(1)
        if not start is None and start < self._df.iloc[-1,:]['section_start']:
            inversion.loc[0, 'section_end'] = start

        inversion = inversion.dropna().rename(columns={"section_end":"section_start", "section_start": "section_end"})
        if not start is None and inversion.loc[0, 'section_start'] < start:
            inversion.loc[0, 'section_start'] = start
        if not end is None and inversion.loc[inversion.index[-1], 'section_end'] > end:
            inversion.loc[inversion.index[-1], 'section_end'] = end

        return TimeFrameGroup(inversion)



    def __iter__(self):
        if len(self._df) == 0:
            return iter([])
        else:
            return iter(list(self._df.apply(lambda row: TimeFrame(start=row['section_start'], end=row['section_end']), axis=1)))

    def extend(self, new_timeframes):
        ''' new_timeframes muss auch TimeFrameGroup sein '''

        if not new_timeframes._df.empty:
            if len(self._df) == 0:
                self._df = new_timeframes._df.copy()
            else:
                self._df = self._df.append(new_timeframes._df)
        #data_for_cache = {'section_start':[], 'section_end':[]}
        #for section in new_timeframes:
        #    data_for_cache['section_start'].append(section.start)
        #    data_for_cache['section_end'].append(section.end)
        
        #if len(data_for_cache['section_start']) > 0:
        #    if len(self._df) == 0:
        #        self._df = pd.DataFrame(data_for_cache, dtype=np.datetime64)
        #        self._df['section_start'] = self._df['section_start'].dt.tz_localize('utc')
        #        self._df['section_end'] = self._df['section_end'].dt.tz_localize('utc')
        #    else:
        #        self._df = self._df.append(pd.DataFrame(data_for_cache))

    def count(self):
        return len(self._df)

    def __getitem__(self, i):
        elements = self._df.iloc[i,:]
        return TimeFrame(elements['section_start'], elements['section_end'])
        
    def pop(self, i):
        if i is None:
            i = -1
        last = self._df.iloc[i,:]
        self._df.drop(self._df.index[i], inplace = True)
        return TimeFrame(last['section_start'], last['section_end'])

    def drop_all_but(self,i):
        return TimeFrameGroup(self._df[i:i+1].reset_index(drop=True))


#class TimeFrameGroup(list):
#    """A collection of nilmtk.TimeFrame objects.
#    The timeframegroup is used to store TimeFrames of a certain
#    type (eg. good sections) for a whole load profile together.
#    It then allows intersection functionality between multiple
#    load profiles to eg. find the good timeframes in all 
#    the TimeFrameGroups.

#    The TimeFrameGroup has been rewritten using pandas DataFrames 
#    because the previous implementation was far to slow
#    """

#    def __init__(self, timeframes=None):
#        if isinstance(timeframes, pd.core.indexes.datetimes.DatetimeIndex):
#            periods = timeframes
#            timeframes = [TimeFrame(period.start_time, period.end_time)
#                          for period in periods]
#        args = [timeframes] if timeframes else []
#        super(TimeFrameGroup, self).__init__(*args)

#    def plot(self, ax=None, y=0, height=1, gap=0.05, color='b', **plot_kwargs):
#        if ax is None:
#            ax = plt.gca()
#        ax.xaxis.axis_date()
#        height -= gap * 2
#        for timeframe in self:
#            length = timeframe.timedelta.total_seconds() / SECS_PER_DAY
#            bottom_left_corner = (mdates.date2num(timeframe.start), y + gap)
#            rect = plt.Rectangle(bottom_left_corner, length, height,
#                                 color=color, **plot_kwargs)
#            ax.add_patch(rect)

#        ax.autoscale_view()
#        return ax

#    def intersection(self, other):
#        """Returns a new TimeFrameGroup of self masked by other.

#        Illustrated example:

#         self.good_sections():  |######----#####-----######|
#        other.good_sections():  |---##---####----##-----###|
#               intersection():  |---##-----##-----------###|
#        """
#        assert isinstance(other, (TimeFrameGroup, list))
#        new_tfg = TimeFrameGroup()
#        for self_timeframe in self:
#            for other_timeframe in other:
#                intersect = self_timeframe.intersection(other_timeframe)
#                if not intersect.empty:
#                    new_tfg.append(intersect)
#        return new_tfg

#    def uptime(self):
#        """Returns total timedelta of all timeframes joined together."""
#        uptime = timedelta(0)
#        for timeframe in self:
#            uptime += timeframe.timedelta
#        return uptime

#    def remove_shorter_than(self, threshold):
#        """Removes TimeFrames shorter than `threshold` seconds."""
#        new_tfg = TimeFrameGroup()
#        for timeframe in self:
#            if timeframe.timedelta.total_seconds() >= threshold:
#                new_tfg.append(timeframe)

#        return new_tfg

#    def invert(self, start = None, end = None):
#        ''' 
#        Returns a TimeFrameGroup with inverted rectangles.
#        That means where there was a gap before is now a 
#        TimeFrame and vice versa.
#        '''

#        if not(start is None and end is None):
#            raise Exception("Not implemented yet")

#        if len(self) < 2:
#            return TimeFrameGroup()

#        new_tfg = TimeFrameGroup()
#        prevEnd = self[0].end
#        for timeframe in self[1:]:
#            if prevEnd != timeframe.start:
#                new_tfg.append(TimeFrame(start = prevEnd, end = timeframe.start))
#            prevEnd = timeframe.end            
#        return new_tfg


#    def diff(self, other):
#        '''
#        self.good_sections():   |######----#####-----######|
#        other.good_sections():  |---##---####----##-----###|
#        diff():                 |###--#------###-----###---|
#        '''
#        assert isinstance(other, (TimeFrameGroup, list))
#        if len(self) == 0:
#            return TimeFrameGroup(self[0].start)
#        elif len(other) == 0:
#            return deepcopy(self)
                
#        other_cpy = deepcopy(other)
#        other_cpy.insert(0,TimeFrame(start=self[0].start - pd.Timedelta(1, unit='s'), end=self[0].start))
#        other_cpy.append(TimeFrame(start=self[0].end, end=self[0].end + pd.Timedelta(1, unit='s')))
#        other = other_cpy.invert()
#        return self.intersection(other)
        
            

#        #assert isinstance(other, (TimeFrameGroup, list))
#        #total_result = TimeFrameGroup()
#        #for self_timeframe in self:
#        #    starts = { self_timeframe.start }
#        #    ends = { self_timeframe.end }

#        #    # Finde die, wo start vor dem ende und ende hinter dem Start
#        #    of_interest = []
#        #    for other_timeframe in other:
#        #        if self_timeframe.contains(other_timeframe.start): #> self_timeframe.start and other_timeframe.end < self_timeframe.end:
#        #            starts.add(other_timeframe.end)
#        #        if self_timeframe.contains(other_timeframe.end):# other_timeframe.start > self_timeframe.start and other_timeframe.start < self_timeframe.end:
#        #            ends.add(other_timeframe.start)
            
#        #    for element in 
#        #    for i in of_interest:
#        #        if other_timeframe.end < self_timeframe.end:
#        #            if  other_timeframe.start < self_timeframe.end:

#            #for 
#            #    intersect = self_timeframe.intersection(other_timeframe)
#            #    if not intersect.empty:
#            #        new_tfg.append(intersect)

#            #    if other_timeframe.end > self_timeframe.end:
#            #        append(TimeFrame(start=start, end = self_timeframe.end)
#            #        continue aussen
#            #    else:
#            #        new_tfg.append(start=start, end=other_timeframe.end)
#            #        start = other_timeframe.end
#            #total_result.union(cur_result)
#        return total_result

