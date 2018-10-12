# distutils: language=c++

import pandas as pd
DEBUG = False
import time 
import sys
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from cython.parallel import parallel, prange
from ..timeframe import TimeFrame
from numpy import diff, concatenate
from ..utils import timedelta64_to_secs
import gc
from nilmtk.timeframegroup import TimeFrameGroup


def _free_enumerable_fast(element):
    if isinstance(element, (list, np.ndarray, pd.DatetimeIndex)):
        last_index = element[-1]
    elif isinstance(element, pd.DataFrame):
        last_index = element.index[-1]
    else:
        raise("Function not working for this type.")
    del element
    gc.collect()
    return last_index


def get_good_sections_fast(df, max_sample_period, look_ahead=None,
                      previous_chunk_ended_with_open_ended_good_section=False):
    """
    Parameters
    ----------
    df : pd.DataFrame
    look_ahead : pd.DataFrame
    max_sample_period : number

    Returns
    -------
    sections : list of TimeFrame objects
        Each good section in `df` is marked with a TimeFrame.
        If this df ends with an open-ended good section (assessed by
        examining `look_ahead`) then the last TimeFrame will have
        `end=None`.  If this df starts with an open-ended good section
        then the first TimeFrame will have `start=None`.
    """
    index = df.dropna().sort_index().index
    _free_enumerable_fast(df)

    if len(index) < 2:
        return []

    # Determine where there are missing samples
    timedeltas_sec = timedelta64_to_secs(diff(index.values))
    timedeltas_check = timedeltas_sec <= max_sample_period
    _free_enumerable_fast(timedeltas_sec)
    
    # Determine start and end of good sections (after/before missing samples regions)
    timedeltas_check = concatenate(
        [[previous_chunk_ended_with_open_ended_good_section],
         timedeltas_check])
    transitions = diff(timedeltas_check.astype(np.int))
    last_timedeltas_check  = _free_enumerable_fast(timedeltas_check)
    good_sect_starts = list(index[:-1][transitions ==  1])
    good_sect_ends   = list(index[:-1][transitions == -1])
    last_index  = _free_enumerable_fast(index)

    # Use look_ahead to see if we need to append a 
    # good section start or good section end.
    look_ahead_valid = look_ahead is not None and not look_ahead.empty
    if look_ahead_valid:
        look_ahead_timedelta = look_ahead.dropna().index[0] - last_index
        look_ahead_gap = look_ahead_timedelta.total_seconds()
    if last_timedeltas_check: # current chunk ends with a good section
        if not look_ahead_valid or look_ahead_gap > max_sample_period:
            # current chunk ends with a good section which needs to 
            # be closed because next chunk either does not exist
            # or starts with a sample which is more than max_sample_period
            # away from df.index[-1]
            good_sect_ends += [last_index]
    elif look_ahead_valid and look_ahead_gap <= max_sample_period:
        # Current chunk appears to end with a bad section
        # but last sample is the start of a good section
        good_sect_starts += [last_index]

    # Work out if this chunk ends with an open ended good section
    all_sections_closed = (
        len(good_sect_ends) > len(good_sect_starts) or 
        len(good_sect_ends) == len(good_sect_starts) and not previous_chunk_ended_with_open_ended_good_section)
    ends_with_open_ended_good_section = not all_sections_closed

    # If this chunk starts or ends with an open-ended good 
    # section then the missing edge is remembered by a None
    # at the begging/end. (later in the overallresult this
    # can then be stacked together above multiple chunks )
    if previous_chunk_ended_with_open_ended_good_section:
        good_sect_starts = [None] + good_sect_starts
    if ends_with_open_ended_good_section:
        good_sect_ends += [None]

    # Merge together starts and ends and return sections 
    # as result for timeframe of this chunk
    assert len(good_sect_starts) == len(good_sect_ends)
    sections = [TimeFrame(start, end)
                for start, end in zip(good_sect_starts, good_sect_ends)
                if not (start == end and start is not None)]
    _free_enumerable_fast([good_sect_starts, good_sect_ends])
    return sections







def get_nonzero_sections_fast(df):
    """
    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    sections : list of TimeFrame objects
        Each nonzero section in `df` is marked with a TimeFrame.
        If this df ends with an open-ended nonzero section (assessed by
        examining `look_ahead`) then the last TimeFrame will have
        `end=None`.  If this df starts with an open-ended nonzero section
        then the first TimeFrame will have `start=None`.
    """

    # Find the switching actions, which stay constant for minimal_zerotime times
    minimal_zerotime = 3
    look_ahead = getattr(df, 'look_ahead', None)
    df = df > 0    

    tmp = df.astype(np.int).diff()
    nonzero_sect_starts = (tmp == 1)
    nonzero_sect_ends = (tmp == 0)
    for i in range(2,minimal_zerotime):
        tmp = df.astype(np.int).diff(i)
        nonzero_sect_starts *= tmp == 1
        nonzero_sect_ends *= tmp == 0
    tmp = df.astype(np.int).diff(minimal_zerotime)
    nonzero_sect_starts *=  tmp == 1
    nonzero_sect_ends *= tmp == -1
    del tmp
    nonzero_sect_starts = list(df[nonzero_sect_starts].dropna().index)
    nonzero_sect_ends   = list(df[nonzero_sect_ends.shift(-minimal_zerotime).fillna(False)].dropna().index)

    # If this chunk starts or ends with an open-ended
    # nonzero section then the relevant TimeFrame needs to have
    # a None as the start or end.
    for i in range(minimal_zerotime):
        if df.iloc[i, 0] == True:
            nonzero_sect_starts = [df.index[i]] + nonzero_sect_starts
            break

    if df.iloc[-1,0] == True:
        nonzero_sect_ends += [None]
    else:
        # Only start new zerosection when long enough, need look_ahead
        for i in range(1,minimal_zerotime+1):
            if df.iloc[-i, 0] != False:
                break

        if i < (minimal_zerotime):
            if look_ahead.head(minimal_zerotime-i).sum()[0] == 0:
                nonzero_sect_ends += [df.index[-i]] #, 0]]
            else:
                nonzero_sect_ends += [None]


    # Merge together ends and starts
    assert len(nonzero_sect_starts) == len(nonzero_sect_ends)
    sections = [TimeFrame(start, end)
                for start, end in zip(nonzero_sect_starts, nonzero_sect_ends)
                if not (start == end and start is not None)]

    # Memory management
    del nonzero_sect_starts
    del nonzero_sect_ends
    gc.collect()

    return sections



def get_nonzero_sections_fast(df):
    """
    The input are always good_sections

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    sections : list of TimeFrame objects
        Each nonzero section in `df` is marked with a TimeFrame.
        If this df ends with an open-ended nonzero section (assessed by
        examining `look_ahead`) then the last TimeFrame will have
        `end=None`.  If this df starts with an open-ended nonzero section
        then the first TimeFrame will have `start=None`.
    """

    df = df > 0    
    tmp = df.astype(np.int).diff()
    nonzero_sect_starts = df[(tmp == 1).values].index.values
    nonzero_sect_ends = df[(tmp == -1).values].index.values
    return nonzero_sect_starts, nonzero_sect_ends



def intersect_many_fast(groups):
    ''' 
    Function to do a do a fast intersection between many timeframes
    '''
    all_events = pd.Series()
    for group in groups:
        all_events = all_events.append(pd.Series(1, index=group._df['section_start']))
        all_events = all_events.append(pd.Series(-1, index=group._df['section_end']))
    all_events.sort_index(inplace=True)
    all_active = (all_events.cumsum()==len(groups))
    starts = all_events.index[all_active]
    ends = all_active.shift(1)
    ends[0] = False
    ends = all_events[ends].index
    result = pd.DataFrame({"section_start": starts, "section_end":ends})
    return TimeFrameGroup(result)