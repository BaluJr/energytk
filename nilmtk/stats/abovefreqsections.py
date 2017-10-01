from __future__ import print_function, division
import numpy as np
from numpy import diff, concatenate
import gc
from .AboveFreqsectionsresults import AboveFreqSectionsResults
from ..timeframe import TimeFrame
from ..utils import timedelta64_to_secs
from ..node import Node
from ..timeframe import list_of_timeframes_from_list_of_dicts, timeframe_from_dict


class AboveFreqSections(Node):
    """Locate sections of data where the sampling rate is above a certain limit.
    TODO

    Attributes
    ----------
    previous_chunk_ended_with_open_ended_AboveFreq_section : bool
    """

    postconditions =  {'statistics': {'AboveFreq_sections': []}}
    results_class = AboveFreqSectionsResults
        
    def reset(self):
        self.previous_chunk_ended_with_open_ended_AboveFreq_section = False

    def process(self):
        metadata = self.upstream.get_metadata()
        self.check_requirements()
        self.results = AboveFreqSectionsResults()
        for chunk in self.upstream.process():
            self._process_chunk(chunk)
            yield chunk

    def _process_chunk(self, df):
        """
        Only checks where the chunk has AboveFreq values.

        Parameters
        ----------
        df : pd.DataFrame
            with attributes:
            - look_ahead : pd.DataFrame
            - timeframe : nilmtk.TimeFrame

        Returns
        -------
        None

        Notes
        -----
        Updates `self.results`
            Each AboveFreq section in `df` is marked with a TimeFrame.
            If this df ends with an open-ended AboveFreq section (assessed by
            examining df.look_ahead) then the last TimeFrame will have
            `end=None`. If this df starts with an open-ended AboveFreq section
            then the first TimeFrame will have `start=None`.
        """
        # Retrieve relevant metadata
        timeframe = df.timeframe

        # Process dataframe
        AboveFreq_sections = get_AboveFreq_sections(
            df, self.previous_chunk_ended_with_open_ended_AboveFreq_section)

        # Set self.previous_chunk_ended_with_open_ended_AboveFreq_section
        if AboveFreq_sections:
            self.previous_chunk_ended_with_open_ended_AboveFreq_section = (
                AboveFreq_sections[-1].end is None)

            # Update self.results
            self.results.append(timeframe, {'sections': [AboveFreq_sections]})




def _free_memory_dataframe(df):
    last_index = df[-1]
    del index
    gc.collect()
    return last_index

def get_AboveFreq_sections(df, previous_chunk_ended_with_zero=False):
    """
    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    sections : list of TimeFrame objects
        Each AboveFreq section in `df` is marked with a TimeFrame.
        If this df ends with an open-ended AboveFreq section (assessed by
        examining `look_ahead`) then the last TimeFrame will have
        `end=None`.  If this df starts with an open-ended AboveFreq section
        then the first TimeFrame will have `start=None`.
    """
    df = df.dropna()
    df = df[df > 0]
    index = df.index
    
    switches = diff(timedeltas_check.astype(np.int))
    AboveFreq_sect_starts = list(index[:-1][transitions ==  1])
    AboveFreq_sect_ends   = list(index[:-1][transitions == -1])
    last_index = _free_memory_dataframe(index)

    # Use look_ahead to see if we need to append a 
    # AboveFreq sect start or AboveFreq sect end.
    look_ahead_valid = look_ahead is not None and not look_ahead.empty
    if look_ahead_valid:
        look_ahead_timedelta = look_ahead.dropna().index[0] - last_index
        look_ahead_gap = look_ahead_timedelta.total_seconds()
    if last_timedeltas_check: # current chunk ends with a AboveFreq section
        if not look_ahead_valid or look_ahead_gap > max_sample_period:
            # current chunk ends with a AboveFreq section which needs to 
            # be closed because next chunk either does not exist
            # or starts with a sample which is more than max_sample_period
            # away from df.index[-1]
            AboveFreq_sect_ends += [last_index]
    elif look_ahead_valid and look_ahead_gap <= max_sample_period:
        # Current chunk appears to end with a bad section
        # but last sample is the start of a AboveFreq section
        AboveFreq_sect_starts += [last_index]

    # Work out if this chunk ends with an open ended AboveFreq section
    if len(AboveFreq_sect_ends) == 0:
        ends_with_open_ended_AboveFreq_section = (
            len(AboveFreq_sect_starts) > 0 or 
            previous_chunk_ended_with_open_ended_AboveFreq_section)
    elif len(AboveFreq_sect_starts) > 0:
        # We have AboveFreq_sect_ends and AboveFreq_sect_starts
        ends_with_open_ended_AboveFreq_section = (
            AboveFreq_sect_ends[-1] < AboveFreq_sect_starts[-1])
    else:
        # We have AboveFreq_sect_ends but no AboveFreq_sect_starts
        ends_with_open_ended_AboveFreq_section = False

    # If this chunk starts or ends with an open-ended
    # AboveFreq section then the relevant TimeFrame needs to have
    # a None as the start or end.
    if previous_chunk_ended_with_open_ended_AboveFreq_section:
        AboveFreq_sect_starts = [None] + AboveFreq_sect_starts
    if ends_with_open_ended_AboveFreq_section:
        AboveFreq_sect_ends += [None]

    assert len(AboveFreq_sect_starts) == len(AboveFreq_sect_ends)

    sections = [TimeFrame(start, end)
                for start, end in zip(AboveFreq_sect_starts, AboveFreq_sect_ends)
                if not (start == end and start is not None)]

    # Memory management
    del AboveFreq_sect_starts
    del AboveFreq_sect_ends
    gc.collect()

    return sections
