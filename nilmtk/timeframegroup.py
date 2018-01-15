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
    """ A collection of nilmtk.TimeFrame objects.
    The timeframegroup is used to store TimeFrames of a certain
    type (eg. good sections) for a whole load profile together.
    It then allows intersection functionality between multiple
    load profiles to eg. find the good timeframes in all 
    the TimeFrameGroups.

    The TimeFrameGroup has been rewritten using pandas DataFrames 
    because the previous implementation was far to slow

    Attributes:
    ----------
    _df: [start_time, end_time]
         The dataframe with the sec_start sec_end 
    """

    def __init__(self, timeframes=None, starts_and_ends = None):
        if isinstance(timeframes, TimeFrameGroup):
            self._df = timeframes._df.copy()
        if isinstance(timeframes, pd.core.indexes.datetimes.DatetimeIndex):
            self._df = timeframes
        elif isinstance(timeframes, pd.DataFrame):
            self._df = timeframes.copy() 
        elif not starts_and_ends is None:
            self._df = pd.DataFrame({'section_start': starts_and_ends['starts'],  'section_end': starts_and_ends['ends']})
        elif not timeframes is None:
            self._df = pd.DataFrame([(frame.start, frame.end) for frame in timeframes], columns = ['section_start', 'section_end'])
        else:
            self._df = pd.DataFrame(columns = ['section_start', 'section_end'])


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
        ''' Returns the timeframe from start of first section to end of last section.

        Returns:
            timeframe: outer timeframe of this TimeFrameGroup
        '''

        if self._df.empty:
            return TimeFrame(start = None, end = None)

        idx = self._df.index
        return TimeFrame(start = self._df.loc[idx[0], 'section_start'], end = self._df.loc[idx[-1], 'section_end'])


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
        
        Paramters
        ---------
        groups: [nilmtk.TimeFrameGroup]
            The group of timeframegroups to calculate the union for.
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

        # Hier hat es geknallt als ich Accuracy als Error Metric berechnen wollte. Bei der Assertion

        assert isinstance(other, (TimeFrameGroup, list))
        return TimeFrameGroup.intersect_many([self, other])


    def intersect_many(groups):
        ''' 
        Function to do a do a fast intersection between many timeframes
        
        Paramters
        ---------
        groups: [nilmtk.TimeFrameGroup]
            The group of timeframegroups to calculate the intersection for.
        '''
        if any(map(lambda grp: len(grp._df) == 0, groups)):
            return TimeFrameGroup()

        all_events = pd.Series()
        for group in groups:
            all_events = all_events.append(pd.Series(1, index=pd.DatetimeIndex(group._df['section_start'])))
            all_events = all_events.append(pd.Series(-1, index=pd.DatetimeIndex(group._df['section_end'])))
        all_events.sort_index(inplace=True)
        all_active = (all_events.cumsum()==len(groups))
        starts = all_events.index[all_active]
        ends = all_active.shift(1).fillna(False)
        #if len(ends > 0):
        #    ends[0] = False
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


    def matching_many(groups):
        ''' 
        Function to do a do a fast matching between many timeframes
        If the groups are two TimeFrameGroups as binary estimator, this 
        is the accuracy.

        Paramters
        ---------
        groups: [nilmtk.TimeFrameGroup]
            The group of timeframegroups to calculate the matching for.
        '''
        if any(map(lambda grp: len(grp._df) == 0, groups)):
            return TimeFrameGroup()

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


    def get_TP_TN_FP_FN(self, ground_truth):
        ''' Returns all the basic descriptors of binary classifier.

        Paramters
        ---------
        ground_truth: TimeFrameGroup
            The ground truth this timeframegroup is compared with.

        Returns
        -------
        TP: TimeFrameGroup
            TimeFrameGroup designating sections where the state is True Positive
        TN: TimeFrameGroup
            TimeFrameGroup designating sections where the state is True Negative
        FP: TimeFrameGroup
            TimeFrameGroup designating sections where the state is False Positive
        FN: TimeFrameGroup
            TimeFrameGroup designating sections where the state is False Negative
        '''
        all_events = pd.Series()
        for i, cur in enumerate([self, ground_truth]):
            i += 1 # The i is used to distinguish FP and FN
            all_events = all_events.append(pd.Series(1*i, index=pd.DatetimeIndex(cur._df['section_start'])))
            all_events = all_events.append(pd.Series(-1*i, index=pd.DatetimeIndex(cur._df['section_end'])))
        all_events.sort_index(inplace=True)
        all_events_sum = all_events.cumsum()

        TP = (all_events_sum == 3) # both on
        TN = (all_events_sum == 0) # both off
        FP = (all_events_sum == 1) # gt == 0 but self == 1
        FN = (all_events_sum == 2) # gt == 2 but self == 0

        # Remove last, which is always created after end of all sections
        results = []
        for cur in [TP, TN, FP, FN]:
            starts = all_events.index[cur]#[:-1]
            ends = cur.shift(1)
            if len(ends > 0):
                ends[0] = False
            ends = all_events[ends].index

            if len(starts) == 0 or len(ends) == 0:
                results.append(TimeFrameGroup())
                continue

            if starts[-1] > ends[-1]:
                starts = starts[:-1]
            result = pd.DataFrame({'section_start': starts, 'section_end':ends})
            results.append(TimeFrameGroup(result).simplify())
        return results

    def uptime(self):
        """
        Calculates total timedelta of all timeframes joined together.

        Returns
        -------
        uptime: int
            total timedelta of all timeframes joined together.
        """
        if self._df.empty:
            return pd.Timedelta(0)

        return (self._df['section_end'] - self._df['section_start']).sum()



    def remove_shorter_than(self, threshold):
        """
        Removes TimeFrames shorter than `threshold` seconds.

        Parameters
        ----------
        threshold: int
            Only keep segments, with a duration longer than threshold.

        Returns
        -------
        simplified: nilmtk.TimeFrameGroup
            A timeframegroup with the targeted segments removed.
        """
        return TimeFrameGroup(self._df[self._df['section_end']-self._df['section_start'] > threshold])


    def merge_shorter_gaps_than(self, threshold):
        """
        Merges TimeFrames which are separated by a timespan that is shorter than 
        `threshold` seconds.

        Parameters
        ----------
        threshold: int
            Only keep gaps, with a duration longer than threshold.

        Returns
        -------
        simplified: nilmtk.TimeFrameGroup
            A timeframegroup with the targeted segments removed.
        """
        if len(self._df) < 2:
            return TimeFrameGroup(self._df.copy())

        if isinstance(threshold, str):
            threshold = pd.Timedelta(threshold)
        if isinstance(threshold, int):
            threshold = pd.Timedelta(str(threshold) + "s")

        gap_larger = ((self._df["section_start"].shift(-1).ffill() - self._df["section_end"]) > threshold)
        gap_larger.iloc[-1] = True # keep last
        relevant_starts = self._df[["section_start"]][gap_larger.shift(1).fillna(True)].reset_index(drop=True)
        relevant_ends = self._df[["section_end"]][gap_larger].reset_index(drop=True)
        return TimeFrameGroup(pd.concat([relevant_starts, relevant_ends], axis=1))


    def simplify(self):
        '''
        Merges all sections, which are next to each other.
        In other words: Removes gaps of zero length.

        Returns
        -------
        simplified: TimeFrameGroup:
            The simplified timeframegroup.
        '''
        to_keep = (self._df["section_start"].shift(-1) != self._df["section_end"])
        #to_keep.iloc[-1] = True # keep last
        relevant_starts = self._df[["section_start"]][to_keep.shift(1).fillna(True)].reset_index(drop=True)
        relevant_ends = self._df[["section_end"]][to_keep].reset_index(drop=True)
        return TimeFrameGroup(pd.concat([relevant_starts, relevant_ends], axis=1))




    def truncate(self, timeframe = None, start = None, end = None, ):
        ''' Removes all sections outside the given section.
        The input can be either a timeframe of start and end.

        Paramter
        --------
        timeframe: nilmtk.TimeFrame
            If set, the timeframegroup is limited to this timeframe
        start: pd.TimeStamp
            The timestamp before which all segments are removed.
        end: pd.TimeStamp
            The timestamp after which all segments are removed.
        '''
        mystart, myend = start, end
        if start == None and not timeframe is None:
            mystart = timeframe.start
        if end == None and not timeframe is None:
            myend = timeframe.end

        # Cut front
        self._df  = self._df[self._df["section_end"] > mystart]
        self._df.loc[self._df["section_start"] < mystart,"section_start"] = mystart

        # Cut end
        self._df  = self._df[self._df["section_start"] < myend]
        self._df.loc[self._df["section_end"] > myend,"section_end"] = myend


    def invert(self, start = None, end = None):
        ''' 
        Returns a TimeFrameGroup with inverted rectangles.
        That means where there was a gap before is now a 
        TimeFrame and vice versa.

        Paramter
        --------
        start, end: pd.TimeStamp
            Defining the start and end of the region to invert.

        Returns
        -------
        Inversion: pd.TimeFrameGroup
            The inverted timeframegroup, with the section beeing the 
            gaps and the other ways arround.
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
        ''' Enabled an iterator to iterate the TimeframeGroup
        '''
        if len(self._df) == 0:
            return iter([])
        else:
            for i, row in self._df.iterrows():
                yield TimeFrame(start=row['section_start'], end=row['section_end'])
            #return iter(list(self._df.apply(lambda row: TimeFrame(start=row['section_start'], end=row['section_end']), axis=1)))
        

    def __getitem__(self, i):
        ''' Enabled to access TimeFrameGroup as a list

        Parameters
        ----------
        i:  int
            Position to return

        Results
        -------
        elements: nilmtk.TimeFrame
            The element at position i
        '''
        elements = self._df.iloc[i,:]
        return TimeFrame(elements['section_start'], elements['section_end'])
        


    def extend(self, new_timeframes):
        ''' Extends a new_timeframes 

        Paramters
        ---------
        new_timeframes: TimeFrameGroup
            Another timeframggroup which shall be added to the current one.
        '''

        if not new_timeframes._df.empty:
            if len(self._df) == 0:
                self._df = new_timeframes._df.copy()
            else:
                self._df = self._df.append(new_timeframes._df)


    def count(self):
        ''' Returns number of conained TimeFrames '''
        return len(self._df)


    def pop(self, i):
        ''' Pops a certain TimeFrame from the TimeFrameGroup
        The TimeFrame at position i is removed from the Group and returned
        
        Paramters
        ---------
        i: int
            The location of the event to remove.
        '''
        if i is None:
            i = -1
        last = self._df.iloc[i,:]
        self._df.drop(self._df.index[i], inplace = True)
        return TimeFrame(last['section_start'], last['section_end'])


    def drop_all_but(self,i):
        ''' Removes all but a single TimeFrame from the TimeFrameGroup

        Paramters
        ---------
        i: int
            The location of the event to keep.
        
        '''
        return TimeFrameGroup(self._df[i:i+1].reset_index(drop=True))



    def calculate_upcounts(timeframegroups):
        '''
        This function takes a list of timeframegroups ans calulates a Series which 
        contain the counts of activates for each section.
        
        Parameters
        ----------
        timframegroups: [nilmtk.TimeFrameGroup]

        Returns
        -------
        counts : pd.Series
            Each entry contains the count active at that point in time.
            The count can be considered constant to the next entry point sothat 
            the values can be afterwards read by using get_loc with the method set 
            to pad or ffill or by resampling to needed amount.
        '''

        start, end = [], []
        for tfg in timeframegroups:
            start.append(tfg._df['section_start'])
            end.append(tfg._df['section_end'])
        start = pd.Series(1, index = pd.concat(start))
        end = pd.Series(-1, index = pd.concat(end))
        counts = start.append(end).sort_index().cumsum()
        counts = counts.groupby(level=0).sum()
        return counts
