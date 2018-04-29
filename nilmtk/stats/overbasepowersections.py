from __future__ import print_function, division
#from .accelerators_stat import get_overbasepower_sections_fast
import numpy as np
from numpy import diff, concatenate
import gc
from .overbasepowersectionsresults import OverBasepowerSectionsResults
from ..timeframe import TimeFrame
from ..utils import timedelta64_to_secs
from ..node import Node
from ..timeframe import list_of_timeframes_from_list_of_dicts, timeframe_from_dict
import pandas as pd

class OverBasepowerSections(Node):
    """Locate sections of data where the samples are bigger 
    larger 0. This is mostly used for disaggregated powerflows
    where there is really a power of 0 when the appliance 
    is not overbasepower. Do not confuse it with the 'get_activations' 
    function of elecmeter. That function is not cached and returns 
    the real dataframe, while this stat only defines the borders.

    !!!!!!!!! DAS MUSS ICH IMPLEMENTIEREN!!!!!!!!!!!
    Only regards sections longer than 1 step. Because otherwise to many      
    problems.

    Attributes
    ----------
    previous_chunk_ended_with_open_ended_overbasepower_section : bool
    """

    postconditions =  {'statistics': {'overbasepower_sections': []}}
    results_class = OverBasepowerSectionsResults
        
    def reset(self):
        ''' nothing to do here '''
        pass

    def process(self):
        metadata = self.upstream.get_metadata()
        self.check_requirements()
        self.results = OverBasepowerSectionsResults(2.3) #TODO
        for chunk in self.upstream.process():
            self._process_chunk(chunk)
            yield chunk

    def _process_chunk(self, df):
        """
        Only checks where the chunk has overbasepower values.

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
            Each overbasepower section in `df` is marked with a TimeFrame.
            If this df ends with an open-ended overbasepower section (assessed by
            examining df.look_ahead) then the last TimeFrame will have
            `end=None`. If this df starts with an open-ended overbasepower section
            then the first TimeFrame will have `start=None`.
        """
        # Retrieve relevant metadata
        timeframe = df.timeframe

        # Process dataframe
        overbasepower_sections_starts, overbasepower_sections_ends = get_overbasepower_sections(df)#_fast(df)

        # Update self.results
        #if overbasepower_sections:
        self.results.append(timeframe, {'sections' : [{'start': overbasepower_sections_starts, 'end': overbasepower_sections_ends}]})


def get_overbasepower_sections(df):
    """
    The input are always good_sections

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    sections : list of TimeFrame objects
        Each overbasepower section in `df` is marked with a TimeFrame.
        If this df ends with an open-ended overbasepower section (assessed by
        examining `look_ahead`) then the last TimeFrame will have
        `end=None`.  If this df starts with an open-ended overbasepower section
        then the first TimeFrame will have `start=None`.
    """

    # Find the switching actions, which stay constant for minimal_zerotime times
    #minimal_zerotime = 3
    #look_ahead = getattr(df, 'look_ahead', None)
    df = df > 15   
    tmp = df.astype(np.int).diff()
    overbasepower_sect_starts = (tmp == 1).values
    overbasepower_sect_ends = (tmp == -1).values
    #overbasepower_sect_starts = df[(tmp == 1).values].index
    if len(df) > 0:
        overbasepower_sect_starts[0] = df.iloc[0,0] # = np.append(df.index[0], overbasepower_sect_starts)
        overbasepower_sect_ends[-1] |= df.iloc[-1,0] #np.append(overbasepower_sect_ends, df.index[-1]) # FUCK THIS SHIT!!! Mal verliert er die Timezone
    overbasepower_sect_starts = pd.Series(df[overbasepower_sect_starts].index)
    overbasepower_sect_ends = pd.Series(df[overbasepower_sect_ends].index)
    return overbasepower_sect_starts, overbasepower_sect_ends

    #for i in range(2,minimal_zerotime):
    #    tmp = df.astype(np.int).diff(i)
    #    overbasepower_sect_starts *= tmp == 1
    #    overbasepower_sect_ends *= tmp == 0
    #tmp = df.astype(np.int).diff(minimal_zerotime)
    #overbasepower_sect_starts *=  tmp == 1
    #overbasepower_sect_ends *= tmp == -1
    #del tmp
    #overbasepower_sect_starts = list(df[overbasepower_sect_starts].dropna().index)
    #overbasepower_sect_ends   = list(df[overbasepower_sect_ends.shift(-minimal_zerotime).fillna(False)].dropna().index)

    ## If this chunk starts or ends with an open-ended
    ## overbasepower section then the relevant TimeFrame needs to have
    ## a None as the start or end.
    #for i in range(minimal_zerotime):
    #    if df.iloc[i, 0] == True:
    #        overbasepower_sect_starts = [df.index[i]] + overbasepower_sect_starts
    #        break

    #if df.iloc[-1,0] == True:
    #    overbasepower_sect_ends += [None]
    #else:
    #    # Only start new zerosection when long enough, need look_ahead
    #    for i in range(1,minimal_zerotime+1):
    #        if df.iloc[-i, 0] != False:
    #            break

    #    if i < (minimal_zerotime):
    #        if look_ahead.head(minimal_zerotime-i).sum()[0] == 0:
    #            overbasepower_sect_ends += [df.index[-i]] #, 0]]
    #        else:
    #            overbasepower_sect_ends += [None]


    ## Merge together ends and starts
    #assert len(overbasepower_sect_starts) == len(overbasepower_sect_ends)

    #if len(overbasepower_sect_ends) > len(overbasepower_sect_starts):
    #    sections = [TimeFrame(start, end)
    #else:
    #    sections = []
    #if len(overbasepower_sect_starts) == 0 == len(overbasepower_sect_ends):
    #    return [TimeFrameGroup()]
    #else:
    #    if overbasepower_sect_starts[0] > overbasepower_sect_ends[0]:
    #        overbasepower_sect_starts.append(None)
    #    if overbasepower_sect_ends[-1] < overbasepower_sect_starts[-1]:
    #        overbasepower_sect_ends.append(None)
    #sections = [TimeFrame(start, end)
    #            for start, end in zip(overbasepower_sect_starts, overbasepower_sect_ends)
    #            if not (start == end and start is not None)]

    # Memory management
    #del overbasepower_sect_starts
    #del overbasepower_sect_ends
    #gc.collect()

    #return sections
