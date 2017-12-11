from __future__ import print_function, division
import pandas as pd
import pytz
from datetime import timedelta
from copy import deepcopy
from warnings import warn
from six import iteritems
from functools import total_ordering
from nilmtk import DataSet
from datetime import datetime
from nilmtk import TimeFrameGroup
import matplotlib.pyplot as plt


class DatasetAnalysis(object):
    """ Offers diverse analysis functionality for whole datasets
    This includes the automated precalculation of the different
    statistics or the creation of diverse plots.

    Currently focuses on self.goodsections and nonzero sections.

    The support of multiple DataSets is reasonable because it is
    often advisable to separate large datasets into multiple smaller
    ones.
    (Eg. my 2TByte dataset with 3000 meters separated into 8
    subsets of about 250GB each)

    Attributes
    ----------
    datasets : [nilmtk.DataSet, ...]
        The datasets to do the calculation for.
    bad_meters: [str] (optional)
        Define some meters which shall be excluded as they are
        malicious.
    timeframe: pd.TimeFrame
        The region for which the analysis shall be performed.
    goodsections: [TimeFrameGroup]
        List of all the good sections.
        Is anonymous at the moment without storing the id.
    nonzerosections: [TimeFrameGroup]
        List of all the nonzero sections.
        Take care, that the nonzerosections can be different for the different
        phases within a single building. This nonetheless only contains a single
        TimeFrameGroup for all phases of the building. It is created by
        intersection. That means all have to be nonzero to form a nonzero section.
        Is anonymous at the moment without storing the id.
    zerosections: {str -> TimeFrameGroup}
        Dictionary from the building to the zerosection.
    phases: {str -> [bool, bool, bool]}
        For each phase whether it is active at all.
        Sometimes there is not a single measurement.

        Set containing the statistics already in memory.
        To allow the analysis the stats have to be preloaded.
        Once loaded subsequent analysis steps are faster since
        no reloading required. The loading happend automatically.
    """

    def __init__(self, paths, bad_meters=None, timeframe = None,
                 merge_shorter_gaps_then = None, remove_shorter_then = None, verbose = False):
        """ Creates an DatasetAnalysis object.
        
        Parameters
        ----------
        paths: str or [str]
            Paths to the datasets, which shall be analyzed.
        bad_meters: [str] (optional)
            Define some meters which shall be excluded as they are
            malicious.
        timeframe: pd.TimeFrame
            The region for which the analysis shall be performed.
            Todo: Should be made optional. Take whole timeframe then.
        merge_shorter_gaps_then: pd.Timedelta
            Merge sections which are separated by a gap smaller then this timedelta
        remove_shorter_then: pd.Timedelta
            Remove sections which are smaller then this timedelta
        verbose: bool
            Whether to return additional information.
        """
        if timeframe is None:
            raise Exception("TimeFrame has to be set. None timeframe not yet supported.")

        if not type(paths) is list:
            paths = [paths]

        self.datasets = []
        for path in paths:
            if verbose:
                print("Load Dataset {0}.".format(path))
            self.datasets.append(DataSet(path))

        self.timeframe = timeframe

        self.bad_meters = bad_meters

        self._load_all_stats(timeframe, verbose=verbose,
                             merge_shorter_gaps_then = merge_shorter_gaps_then,
                             remove_shorter_then = remove_shorter_then)



    def _load_all_stats(self, timeframe, verbose = True, repair = True,
                        merge_shorter_gaps_then = None, remove_shorter_then = None):
        """
        Loads all necessary stats from all meters of the DataSets

        Aftercalling this values the stats are in memory and the corresponding
        name is added to stats_loaded.

        Takes into account, that good_sections are currently always expected to be the same
        for a whole building while the nonzero sections might differe from meter to
        meter of the building.\

        The repairfunction is still adapted to my usecase. Has to be made general.

        Parameters
        ----------
        start: pd.TimeFrame
            The region for which the analysis shall be performed.
            Todo: Should be made optional. Take whole timeframe then.
        verbose: bool
            Whether to return additional information.
        repair: bool
            Sometimes somehow errors appear. With activated repair
            the script retries the process for these steps.
        merge_shorter_gaps_then: pd.Timedelta
            Merge sections which are separated by a gap smaller then this timedelta
        remove_shorter_then: pd.Timedelta
            Remove sections which are smaller then this timedelta
        """

        # Dictionaries to fill should be the ones of the object
        self.goodsections = []
        self.nonzerosections = []
        self.zerosections = {}
        self.phases = {}

        # Do the analysis
        for dataset in self.datasets:
            for building in list(dataset.buildings):

                original_name = dataset.buildings[building].metadata['original_name']
                if original_name in self.bad_meters:
                    continue

                if verbose:
                    print(building)
                try:
                    # Good sections same for the whole building
                    sec = dataset.buildings[building].elec.meters[0].good_sections()
                    sec.truncate(timeframe)
                    if not merge_shorter_gaps_then is None:
                        sec = sec.merge_shorter_gaps_than(pd.Timedelta(merge_shorter_gaps_then))
                    if not remove_shorter_then is None:
                        sec = sec.remove_shorter_than(pd.Timedelta(remove_shorter_then))
                    self.goodsections.append(sec)

                    # Calculate the nonzerosections for each meter
                    curnonzerosections = []
                    curphases = []
                    for i in range(3):
                        sec = dataset.buildings[building].elec.meters[i].nonzero_sections()
                        sec.truncate(timeframe)
                        curphases.append(sec.count() > 0)
                        if not merge_shorter_gaps_then is None:
                            sec = sec.merge_shorter_gaps_than(pd.Timedelta(merge_shorter_gaps_then))
                        curnonzerosections.append(sec)
                    curnonzerosections = TimeFrameGroup.intersect_many(curnonzerosections)
                    curzerosections = curnonzerosections.invert(timeframe.start, timeframe.end)

                    self.phases[building] = curphases
                    self.nonzerosections.append(curnonzerosections)
                    self.zerosections[building] = curzerosections
                except Exception as e:
                    if repair:
                        while (not "UTC" in str(self.goodsections[-1]._df.dtypes[0]) or not "UTC" in str(self.goodsections[-1]._df.dtypes[1])):
                            print("Repair")
                            dataset.buildings[building].elec.meters[0].clear_cache()
                            self.goodsections[-1] = dataset.buildings[building].elec.meters[0].good_sections().results._data
                    else:
                        print("Error in good_section calculation")
                        raise("Error in good_section calculation")


    def find_missing_phases(self):
        """ Finds self.phases which are completly missing.
        Sometimes it happens that not all parts of the multiphase smart meters
        are connected. This function tries to find these sections.

        Parameters
        ----------
        start: pd.Timestamp
            The beginning of the calculations
        verbose: bool
            Whether to return additional information.

        Returns
        -------
        phases : pd.DataFrame
            A dataframe with the columns equal to the phases and the
            value showing whether a certain phase is set at all.
        """
        dfphases = pd.DataFrame(columns=["phase1", "phase2", "phase3"])
        for cur in self.phases:
            original_name = self.datasets[int(cur / 1000)].buildings[cur].metadata["original_name"]
            dfphases.loc[original_name, :] = self.phases[cur]
        return dfphases



    def binning_good_nonzero_sections(self, bins = None):
        """ Creates bins of the lenghts of good nonzero sections.
        This function is helpfull when not all meters of the dataset are clean.
        Bins count the outage of data of a certain period for a certain length.
        These bins are a good possibility to analyse how well the dataquality is.
        Bins are not cumulative. That means an outage will only appear in one bin
        and does not also appear in the smaller bins if available.
        Eg. 6 minute outage appears in the 5-10min bin but not in the 0-1min and
        1-5min bin.

        Parameters
        ----------
        bins: [pd.Timedelta,...]
            A list of timedeltas in increasing order. These define the length of outage counted
            within this bin.
            Default: ["0s","5s","10s","20s","1min","5min","10min","1h","1d","7d","30d","100d","3650d"]

        Returns
        -------
        bins : pd.DataFrame
            A dataframe containing a column for each bin size and a row for each building
            in the dataset.
        """

        if bins is None:
            bins = ["0s", "5s", "10s", "20s", "1min", "5min", "10min", "1h", "1d", "7d", "30d", "100d", "3650d"]
        bin_names = ["bin_" + name for name in bins]
        bins = [pd.Timedelta(t) for t in bins]

        bin_df = None
        for building in self.zerosections:
            sects = self.zerosections[building]
            original_name = self.datasets[int(building / 1000)].buildings[building].metadata["original_name"]
            tst = sects._df['section_end'] - sects._df['section_start']
            tst2 = tst.groupby(pd.cut(tst, bins))
            cnt = tst2.count()
            cnt.index.name = ""
            cnt = cnt.rename(original_name)
            if not bin_df is None:
                bin_df[original_name] = cnt
            else:
                bin_df = pd.DataFrame(cnt)

        bin_df = bin_df.transpose()
        bin_df.columns = [bin_names]
        return bin_df



    def create_overviewplot_sections(self, buildings = None, nonzero_instead_good = False, verbose = True):
        """
        Creates an overview plot of all good sections in one figure.
        If nothing else specified does this for all meters.

        Parameters
        ----------
        buildings: [int]
            The buildings which shall be plotted
        nonzero_instead_good: bool
            If kept false good sections are plotted. Else nonzero sections are plotted.
        verbose: bool
            Whether to output additional information.

        Returns
        -------
        figure: matplotlib.figure.Figure
            The created figure
        """

        sections = self.nonzerosections if nonzero_instead_good else self.goodsections
        if not buildings is None:
            sections = list(map(sections, buildings))

        n = len(self.goodsections)
        fig = plt.figure(figsize=(50, 50)) #, tight_layout=True)
        for i in range(n):
            if verbose:
                print("Plot {0}/{1}".format(str(i),str(n)))

            ax = fig.add_subplot(n, 1, i + 1)
            sections.plot_simple(ax=ax) #self.nonzerosections[i][0]
            ax.set_xlim([self.timeframe.start, self.timeframe.end])

            if i != 0:
                plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            # ax1.set_xlabel('On-Off', fontsize=12)
            # ax1.set_title('Good Sections', fontsize=12)
            # ax.set_ylabel('{0}'.format(i), fontsize=12)
        fig.subplots_adjust(hspace=.0)
        return fig


    def create_plots_sections(self, folder, buildings = None,
                              nonzero_instead_good = False, verbose = True):
        """
        Creates an plot of the good sections of the given or all
        buildings. In addition plots one plot for the intersection.
        Stores all plots inside the given folder.
        Function mostly used to get an overview.

        Parameters
        ----------
        folder: str
            Path to a folder where the plots shall be stored.
        buildings: [int]
            The buildings which shall be plotted
        nonzero_instead_good: bool
            If kept false good sections are plotted. Else nonzero sections are plotted.

        Returns
        -------
        figure: matplotlib.figure.Figure
            The created figure
        """
        sections = self.nonzerosections if nonzero_instead_good else self.goodsections
        if not buildings is None:
            sections = list(map(sections, buildings))

        n = len(self.goodsections)
        for i, sec in enumerate(sections):
            if verbose:
                print("Plot {0}/{1}".format(str(i),str(n)))
            sec.plot()
            plt.savefig(folder + "{0}.png".format(i))

        intersection = TimeFrameGroup.intersect_many(sections)
        intersection.plot()
        plt.savefig(folder + "intersection.png".format(i))




def precalculate_all_stats(paths, bad_meters, verbose = True):
    """ Precalculates all statistics of the dataset.
    Afterwards the statistics are located in the caches.
    This script is perfect to run it overnight.

    Parameters
    ----------
    paths: str or [str]
        Paths to the datasets, which shall be analyzed.
    bad_meters: [str] (optional)
        Define some meters which shall be excluded as they are
        malicious.
    verbose: bool
        Whether to return additional information.
    """

    if not type(paths) is list:
        paths = [paths]

    datasets = []
    for path in paths:
        if verbose:
            print("Load Dataset {0}.".format(path))
        datasets.append(DataSet(path))

    total_sets = len(self.datasets)
    for i, dataset in enumerate(datasets):
        if verbose:
            print("##### Calculate {0}/{1}".format(str(i),str(total_sets)))
        dataset.calc_and_cache_stats(ignore_meters=bad_meters, verbose=True)
        if verbose:
            print("##### Finished {0} at {1}".format(str(i),datetime.now()))