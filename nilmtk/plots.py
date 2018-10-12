''' 
Plotter to collect all plotting functionality at one place.
If available, it uses simple plotting functionalities included into the different classes.
Merges them together to create more meaningfull plots.
'''
from __future__ import print_function, division
import numpy as np
import pandas as pd
import math
from warnings import warn
#from .metergroup import MeterGroup, iterate_through_submeters_of_two_metergroups
#from .electric import align_two_meters
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from nilmtk import TimeFrameGroup
import itertools
from nilmtk import TimeFrameGroup, TimeFrame
import matplotlib.dates as mdates

#############################################################
#region Nilm Plotting

def plot_overall_power_vs_disaggregation(main_meter, disaggregations, verbose = False):
    """ The plot for validating the NILM algorithm.
    Plots the disaggregation below the overall powerflow together with
    orientation lines.

    Parameters
    ----------
    predictions: nilmtk.Electrical
        Electrical with the disaggregation of the meters.
    ground_truth : nilmtk.MeterGroup
        MeterGroup with all the disaggregated meters.
    verbose:
        Whether additional ouput is printed.
    """

    # Create the main figure
    fig = plt.figure()  #, tight_layout=True)

    # Create one bigger subplot for the overall power
    timeframe = disaggregations.get_timeframe(intersection_instead_union = False)
    timeframe.start = timeframe.end - pd.Timedelta("48h")
    ax = fig.add_subplot(4,1,1)
    if not main_meter is None:
        main_meter.plot(ax, timeframe=timeframe, sample_period=2)
        ax.set_xlim([timeframe.start, timeframe.end])
        ax.set_xlabel('Time', fontsize=12)
        ax.set_title('Disaggregation', fontsize=14)
        #ax.set_ylabel('{0}'.format(i), fontsize=12)

    # Create multiple smaller ones for the disaggregated flows
    n = len(disaggregations.meters)
    sections = math.ceil(n / 2 * 3)
    size_main_figure = math.ceil(sections / 3)
    for i, dis in enumerate(disaggregations.meters):
        if verbose:
            print(str(i) + "/" + str(n))
        sub_ax = fig.add_subplot(sections, 1, size_main_figure+i+1)
        dis.plot(sub_ax,timeframe=timeframe, legend = False, sample_period = 2)
        ax.get_shared_x_axes().join(ax, sub_ax)
        ax.get_shared_y_axes().join(ax, sub_ax)
        sub_ax.set_ylim(ax.get_ylim())
        if i != 2:
            ax.set_ylabel("")
        #sub_ax.set_xlim([timeframe.start, timeframe.end])

    # Link the axis
    plt.setp(ax.get_xticklabels(), visible=True)
    #fig.subplots_adjust(hspace=0.0)
    return fig


def plot_phases(building, interval = pd.Timedelta("1d"), verbose = False):
    ''' Simply plots all three phases to see the output.
    This is equal to plotting the different sitemeters of the building.

    Parameters
    ----------
    building: nilmtk.building
        The building for which the different phases are plottet.
    interval: pd.Timedelta
        The timedelta to plot.
    verbose: bool
        Whether to plot additional output.
    '''
    fig = plt.figure()
    start = building.elec.sitemeters()[1].get_timeframe().start
    new_timeframe = TimeFrameGroup([TimeFrame(start=start, end = start + interval)])
    flows = []
    for i in range(1,4):
        if verbose:
            print("Load {0}/{1}".format(i,3))
        flows.append(building.elec.sitemeters()[i].power_series_all_data(sections=new_timeframe))
    all = pd.concat(flows, axis = 1)
    all.columns = ['Phase 1', 'Phase 2', 'Phase 3']
    all.plot(colors=['r', 'g', 'b'], ax = fig.add_subplot(111))
    return fig



def plot_stackplot(disaggregations, total_power = None, stacked = True, verbose = True):
    """ Plots a stackplot, which stacks all disaggregation results on top of each other.

    Parameters
    ----------
    disaggregations: nilmtk.MeterGroup
        Remember appliance 0 is the rest powerflow
    plot_total_power:  nilmtk.Electric (optional)
        Just for comparison an additional plot with the whole powerflow.
        Should be the same as all the diaggregated meters stacked together.
    verbose: bool
        Whether to print additional information

    Returns
    -------
    fig: matplotlib.figure.Figure
        The newly plot figure
    """

    timeframe = disaggregations.get_timeframe(intersection_instead_union = False)
    timeframe.start = timeframe.end - pd.Timedelta("48h")

    # Additional total power plot if demanded
    fig = plt.figure()
    if not total_power is None:
        ax = fig.add_subplot(211)
        total_power.power_series_all_data(sections=[timeframe], sample_period=2).plot(ax = ax)
        ax = fig.add_subplot(212)
    else:
        ax = fig.add_subplot(111)

    # The stacked plot
    all = pd.DataFrame(disaggregations.meters[0].power_series_all_data(sections=[timeframe], sample_period=2).rename('Rest'))
    for i, dis in enumerate(disaggregations.meters):
        if i == 0:
            continue
        name = "Appliance " + str(i)
        if verbose:
            print(name)
        all[name] = dis.power_series_all_data(sections=[timeframe], sample_period=2)
    all = all.fillna(0)
    all.plot.area(ax = ax, stacked = stacked)
    ax.set_xscale("log", nonposx='clip')
    ax.set_xlim([timeframe.start, timeframe.end])
    return fig


def plot_segments(transitions, steady_states, ax = None):
    '''
    This function takes the events and plots the segments.

    Paramters
    ---------
    transitions:
        The transitions with the 'segment' field set
    steady_states:
        The transitions with the 'segment' field set
    ax: matplotlib.axes.Axes
        An axis object to print to.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The newly plot figure
    '''
    # Prepare plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if ax is None:
        ax = plt.gca()
    #ax.xaxis.axis_date()

    # Sort segments to always plot lower segment on top
    steady_states['segment'] = transitions.set_index('starts')['segment']
    steady_states.sort_index(ascending = True, inplace = True)
    steady_states['starts'] = steady_states.index
    firsts = steady_states.groupby('segment').first()
    firsts = firsts.sort_values('starts', ascending = False).index

    # Fill_between does the trick
    for cur in firsts:
        rows = steady_states[steady_states['segment'] == cur]
        ax.fill_between(rows.index.to_pydatetime(), rows['active average'].values, 0, step='post')
    ax.set_xlabel("Time", fontsize = "12")
    ax.set_ylabel("Power [W]", fontsize = "12")
    ax.autoscale_view()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    return fig



def plot_evaluation_assignments(sec_ground_truth, sec_disaggregations, assignments,
                                gt_meters = None, timeframe = None, verbose = False):
    '''
    This function plots the assignments of the preassignment during the NILM evaluation.
    The plot has three columns:
        - The original disaggregated meters
        - The ground_truth meters
        - the combination of the meters assigned to the ground truth meters.

    Paramters
    ---------
    sec_ground_truth: [nilmtk.TimeFrameGroup]
        The on-sections of the ground truth.
    sec_disaggregations: [nilmtk.TimeFrameGroup]
        The on sections of the disaggregated meters. Some of these purely
        disaggregated meters might belong to the same ground truth appliance.
    assignments: dict(int -> [int])
        A dictionary with its entries mapping from a number of the ground_truth meters to a
        list of disaggregation meters. This enables the combination of the disaggregation meters.
    gt_meters: nilmtk.Electric
        If set, the meters are used to get the captions for the plots
    timeframe: nilmtk.Timeframe
        A timeframe for which the plot shall be drawn. If kept None, the whole timeframe
        of the ground_truth is plotted.
    verbose: bool
        If additional output is generated

    Returns
    -------
    fig: matplotlib.figure.Figure
        The newly plotted figure
    '''

    fig = plt.figure(figsize=(50,50)) #, tight_layout=True)
    if timeframe is None:
        timeframe = TimeFrameGroup(map(lambda cur: cur.get_timeframe(), sec_ground_truth)).get_timeframe()
    limit = TimeFrameGroup([timeframe])
    overall_length = max([len(sec_ground_truth), len(sec_disaggregations)])

    # Plot before assignment
    for i, cur_nonzero in enumerate(sec_disaggregations):
        ax = fig.add_subplot(overall_length,3,1+i*3)
        limited = cur_nonzero.intersection(limit)
        if verbose:
            print(str(i) + ": " + str(len(limited._df)))
        limited.plot(ax=ax)
        ax.set_xlim([timeframe.start, timeframe.end])
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_xlabel("Time")
        ax.set_ylabel("Activation")

    # Plot the original load
    for i, cur_nonzero in enumerate(sec_ground_truth):
        ax = fig.add_subplot(overall_length,3,2+i*3)
        limited = cur_nonzero.intersection(limit)
        if verbose:
            print(str(i) + ": " + str(len(limited._df)))
        limited.plot(ax=ax)
        if not gt_meters is None:
            ax.set_title(gt_meters.meters[i].appliances[0].metadata['type'])
        ax.set_xlim([timeframe.start, timeframe.end])
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_xlabel("Time")
        ax.set_ylabel("Activation")

    # Plot assigned disaggregations right
    for i in range(len(sec_ground_truth)):
        cur_nonzero = TimeFrameGroup.union_many(map(lambda a: sec_disaggregations[a], assignments[i]))
        ax = fig.add_subplot(overall_length,3,3+i*3)
        limited = cur_nonzero.intersection(limit)
        if verbose:
            print(str(i) + ": " + str(len(limited._df)))
        limited.plot(ax=ax)
        ax.set_title(str(assignments[i]))
        ax.set_xlim([timeframe.start, timeframe.end])
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_xlabel("Time")
        ax.set_ylabel("Activation")

    return fig



def plot_multiphase_event(original_powerflows, original_adapted, multiphase_events, section,
                          surrounding = 30, col = "active transition", plot_freq = "2s", verbose = False):
    ''' This function is used to plot multiphase events.
    It shows how the multiphase events are cut out and put inside separate poweflows.

    Parameters
    ----------
    original_powerflows: [pd.DataFrame]
        The original transients as DataFrame one per phase
    original_adapted: [pd.DataFrame]
        The new original phases where the multiphaseevents
        are removed.
    multiphase_events:
        The separated transients appearing in multiple phases.
    section: nilmtk.TimeFrame
        The section which shall be plotted.
    surrounding: int
        Minutes in the original power flows plottet
        arround the interesting section.
    col: index
        Which is the power transient index
    plot_freq: str
        The frequency with which the powerflows are resampled before being plotted.
    verbose: bool
        Whether to print additional information
        
    Returns
    -------
    fig: matplotlib.figure.Figure
        The newly plotted figure
    '''

    if not type(surrounding) is pd.Timedelta:
        surrounding = pd.Timedelta(minutes=surrounding)

    fig = plt.figure(figsize=(50,50)) #, tight_layout=True)

    plots_per_column = 3
    all_plots = [original_powerflows, original_adapted, multiphase_events]

    for i, cur_plot in enumerate(all_plots):
        for j, powerflow in enumerate(cur_plot):
            ax = fig.add_subplot(plots_per_column,3,i+j*3+1)
            limited = powerflow.loc[section.start-surrounding:section.end+surrounding][col]
            if verbose:
                print("Plot {0}:{1}".format(i,j))
            limited.loc[section.start-surrounding] = 0
            limited.loc[section.end+surrounding] = 0
            limited = limited.cumsum().resample(plot_freq).ffill()
            limited.plot(ax=ax)
            ax.set_xlim([section.start-surrounding, section.end+surrounding])
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)

    return fig

#endregion



################################################################
#region Cluster plotting

def plot_clustering(clusterers, elements, columns_to_project,
                    subtype_column="subtype", appliance_column="appliance", confidence_column = "confident",
                    print_confidence=True, filter=False, **plot_args):
    '''
    Plotting of points in 2d space. For K-means and gmm the bordes are also plotted.

    Paramters
    ---------
    clusterers: {str -> scikit.GaussianMixture}
        The dictionary of available clusterers as built
        within the eventbased_combination clusterer.
    elements: pd.DataFrame
        The dataframe containing the elements to plot.
    columns_to_project: [index,...]
        The indices to project to. The length of this function
        defines automatically defines the way of plotting.
    subtype_column: index
        The column defining the entry in the clusterers.
    appliance_column: index
        The column defining the appliance.
    confidence_column: index
        The column defining if the prediction was condident
    print_confidence: int
        If not zero, the confidence interval which will be plotted.
        (Currently not yet supported for 3d plots.)
    filter: bool
        Whether only the confident points shall be plotted.
    plot_args: dict
        Additional arguments forwarded to the plot function.
        Eg point size: s=0.1

    Returns
    -------
    fig: matplotlib.figure.Figure
        The newly plot figure
    '''

    # Create the input for the concrete plotting functions
    data = elements[columns_to_project].values
    _, labels = np.unique(elements[subtype_column].values, return_inverse=True)
    labels = labels * 100 + elements[appliance_column].astype(int).values
    confidence = elements[confidence_column].values

    # Call the plotting
    if len(columns_to_project) == 2:
        fig = plot_clustering_2d(clusterers, data, labels, confidence, columns_to_project, print_confidence, filter, **plot_args)
    elif len(columns_to_project) == 3:
        fig = plot_clustering_3d(clusterers, data, labels, confidence, columns_to_project, print_confidence, filter, **plot_args)
    else:
        raise Exception("Only 2d or 3d plot possible.")
    return fig


def plot_clustering_2d(clusterers, data, labels, confidence, columns, print_confidence = 1, filter = False, **plot_kwargs):
    '''
    Plotting of points in 2d space. For K-means and gmm the bordes are also plotted.

    Parameters
    ----------
    clusterers: {str -> scikit.GaussianMixture}
        The dictionary of relevant clusterers as built
        within the eventbased_combination clusterer.
    data: np.ndarray(,2)
        The data points to plot
    labels: np.ndarray(int)
        The labels the datapoints belong to
    confidence: np.ndarray(bool)
        Bool whether the point is seen as confident
    columns: str
        The columns which are printed as label of the axis.
    print_confidence: int
        If not zero, the confidence interval which will be plotted.
    filter: bool
        Whether only the confident points shall be plotted.
    plot_kwargs: dict
        Additional arguments forwarded to the plot function.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The newly plot figure
    '''

    fig = plt.figure()

    plot_kwargs.setdefault('cmap', plt.cm.Set1)
    color = plot_kwargs["cmap"](labels)
    plot_kwargs["s"] = 5

    # Print the datapoints
    plt.scatter(data[confidence][:,0], data[confidence][:,1], c=color[confidence], alpha = 1, **plot_kwargs)
    plt.xlabel("Power Transition [W]")  #columns[0]
    plt.ylabel("Power Peak [W]")        #columns[1]
    if not filter:
        plt.scatter(data[~confidence][:,0], data[~confidence][:,1], c=color[~confidence], alpha = 0.5, **plot_kwargs)

    if not print_confidence:
        return fig

    # Print the confidence intervals if demanded
    for key in clusterers:
        for mean, covar in zip(clusterers[key].means_, clusterers[key].covariances_):
            v, w = np.linalg.eigh(covar)
            v = print_confidence * np.sqrt(2.) * np.sqrt(v)
            #u = w[0] / linalg.norm(w[0]) # already normalized
            w = w[0,:] / np.linalg.norm(w[0,:])
            angle = np.arctan(w[1] / w[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color='black', fill = False, linewidth = 3)
            ell.set_clip_box(fig.bbox)
            ell.set_alpha(0.5)
            fig.axes[0].add_artist(ell)
    plt.show()
    return fig


def plot_clustering_3d(clusterers, data, labels, confidence, columns, print_confidence = True, filter = False, **plot_kwargs):
    '''
    Plotting of points in 3d space with optional colouring after assignments.

    clusterers: {str -> scikit.GaussianMixture}
        The dictionary of available clusterers as built
        within the eventbased_combination clusterer.
    data: np.ndarray(,2)
        The data points to plot
    labels: np.ndarray(int)
        The labels the datapoints belong to
    confidence: np.ndarray(bool)
        Bool whether the point is seen as confident
    columns: str
        The columns which are printed as label of the axis.
        Not yet used as plot labels.
    print_confidence: int
        If not zero, the confidence interval which will be plotted.
        Not yet supported for 3D!!!
    filter: bool
        Whether only the confident points shall be plotted.
    plot_args: dict
        Additional arguments forwarded to the plot function.

    Returns
    -------
    ax: matplotlib.figure.Axes
        The newly plot axes. One has to have a look how to make it a figure.
    '''

    plot_kwargs.setdefault('cmap', plt.cm.Set1)
    color = plot_kwargs["cmap"](labels)

    ax = plt.axes(projection='3d');
    ax.scatter(data[confidence][:,0], data[confidence][:,1], data[confidence][:,2], c=color[confidence],
               s=plot+plot_kwargs["s"])
    if not filter:
        ax.scatter(data[~confidence][:,0], data[~confidence][:,1], data[~confidence][:,2], c=color[~confidence],
                   alpha=0.5 ,s=0.1)
    return ax


def plot_correlation_matrix(corr_base, corr_disag, corr_base_clustered, corr_disag_clustered, corr_all):

    #  native implementation
    corrs = [corr_base, corr_disag, corr_base_clustered, corr_disag_clustered, corr_all]
    names = ["corr_households", "corr_disag", "corr_households_clustered", "corr_disag_clustered", "corr_all"]
    
    for cur in range(len(corrs)):
        if 'cluster' in corrs[cur].columns:
            corrs[cur] = corrs[cur].sort_values("cluster").drop(columns=['cluster'])

    columns =[]
    for cur in corr_base.columns:
        if cur[0] == 'hour' or cur[0] == 'weekday':
            columns.append(cur[1])
        else:
            columns.append(cur[0])

    for names, corr in zip(names, corrs):
        columns =[]
        for cur in corr.columns:
            if cur[0] == 'hour' or cur[0] == 'weekday':
                columns.append(cur[1])
            else:
                columns.append(cur[0])
    
        sns.set_style("white")
        plt.figure()
        cax = plt.matshow(corr.values.astype(float), cmap = plt.cm.seismic, vmin=-1, vmax=1,  aspect='auto')
        plt.colorbar(cax)
        plt.xticks(range(len(columns)), columns);
        #plt.yticks(range(len(corr.index)), range(len(corr.index)));
        #plt.tight_layout()
        plt.savefig("F:/" + names + ".svg", bbox_inches='tight')

def plot_correlations(orig_corrs, disag_corrs, cluster_corrs):
    '''
    Returns three columns of plots with the correlations of dimensions as Bar Diagram.
    For a single household!
    It taks place in three columns:
        1. The correlation of the original 
        2. The correlation of the disaggregations
        3. The correlation of the clusters
    This plot shall visualize the quality of correlation within each diagram.

    Parameters
    ----------
    orig_corrs: pd.DataFrame
        The correlations for the different clusters
        Row per correlation dimension
    disag_corrs: pd.DataFrame
        The metergroup of disaggregations
        Row per correlation dimension
    cluster_corrs: pd.DataFrame
        The correlations of each of the clustered powerflows.

    Results
    -------
    error_report: pd.Df
        The correlations mixed together and calculated.
    '''
    fig = plt.figure()
    plots_per_column = len(disag_corrs)
    # Plot before separation
    corrs = [orig_corrs, disag_corrs, cluster_corrs]
    for i, corr in enumerate(all_plots):
        for j, cur_corr in corr.iterrows():
            ax = fig.add_subplot(plots_per_column,3,i+j*3+1)
            cur_corr.plot(ax = ax)

            #limited = powerflow.loc[section.start-surrounding:section.end+surrounding][col]
            #if verbose:
            #    print("Plot {0}:{1}".format(i,j))
            #limited.loc[section.start-surrounding] = 0
            #limited.loc[section.end+surrounding] = 0
            #limited = limited.cumsum().resample(plot_freq).ffill()
            #limited.plot(ax=ax)
            #ax.set_xlim([section.start-surrounding, section.end+surrounding])
            #plt.setp(ax.get_xticklabels(), visible=False)
            #plt.setp(ax.get_yticklabels(), visible=False)

#endregion



################################################################
#region Forecast plotting

def plot_ext_data_relation(load, external_data, interval = None, smoothing = None, ):
    '''
    Plots a two scale plot for the load and the external data.
    Like this one can compare influence of external data to the powerflow.

    Paramters
    ---------
    load: pd.Series
        The load profile in KW
    external_data: pd.Series
        The external data one wants to compare the load to.
    interval: pd.Timeframe
        Optional definition of the reagion to plot.
    smoothing: int
        If set, the data is smoothed to the rolling average across the 
        given dates. Reasonable since the correlation sometimes gets 
        onlz visible within long term periods.
    '''
    if ax == None:
        fig, ax1 = plt.subplots()
       
    if not smoothing is None:
        load = load.rolling_mean(smoothing)
    load.plot(ax=ax1)
    #ax1.plot(t, s1, 'b-')
    #ax1.set_xlabel('time (s)')
    ## Make the y-axis label, ticks and tick labels match the line color.
    #ax1.set_ylabel('exp', color='b')
    #ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    if not smoothing is None:
        external_data = external_data.rolling_mean(smoothing)
    external_data.plot(ax=ax2)
    #s2 = np.sin(2 * np.pi * t)
    #ax2.plot(t, s2, 'r.')
    #ax2.set_ylabel('sin', color='r')
    #ax2.tick_params('y', colors='r')
    
    return fig



def plot_forecast(forecasts, original_load, interval = None, ax = None, additional_data =  {}):
    ''' Plots the forecast along the real powerflow
    This is the main function to be used for visualization of forecasting quality.

    Paramters
    ---------
    forecast: pd.Series
        The forecasted values
    original_load: pd.Series
        The original load. Series contains at least interval of forecast.
    interval: pd.Timeframe
        Optional definition of the reagion to plot. If nothing given 
        the timeframe of the forecasts dataframe is used - double the interval
    additional_data: [pd.Dataframe] or [pd.Series]
        Additional data, which can be plotted. For example the residuals of the
        ARIMA model or the external power.

    Returns
    -------
    ax: matplotlib.axes.Axes
        The ax object with the forecasts.
    '''

    if interval is None:
        load = original_load[forecasts.index[0]-pd.Timedelta("24h"):forecasts.index[-1]+pd.Timedelta("24h")]
    else:
        load = original_load[interval.start:interval.end]

    if ax is None:
        fig, ax = plt.subplots()

    # One main plot and distinct plot for each forecast
    load.plot(ax=ax, linewidth=2)
    marker = itertools.cycle((',', '+', '.', 'o', '*')) 
    for forecast in forecasts:
        forecasts[forecast].plot(ax=ax, marker = next(marker))

    # Finally plot additiona data if available
    for additional in additional_data:
        residuals =pd.DataFrame(model_fit.resid)
        additional.plot(ax = ax, kind='kde')
        pyplot.show()
        residuals.plot()
        pyplot.show()

    return ax
#endregion



################################################################
#region Elaborate Powerflow plotting
def plot_powerflow_from_events(events_list=[], column = 'active transition'):
    """
    Currently not in use.
    """
    fig, ax = plt.subplots(figsize=(8,6))#grps.plot(kind='kde', ax=ax, legend = None)
    for events in events_list:
        events[column].cumsum().plot(ax=ax)
    #fig, ax = plt.subplots(figsize=(8,6))#grps.plot(kind='kde', ax=ax, legend = None)
            #transients[a][firsts.values[1]:last.values[1]]['active transition'].cumsum().plot(ax=ax)
            #transients[a].loc[common_transients.index][firsts.values[1]:last.values[1]]['active transition'].cumsum().plot(ax=ax)
#endregion



################################################################
#region Originally available plots

def plot_series(series, ax=None, fig=None, date_format='%d/%m/%y %H:%M:%S', tz_localize=True, **plot_kwargs):
    """Faster plot function
    Function is about 5 times faster than pd.Series.plot().

    Parameters
    ----------
    series : pd.Series
        Data to plot
    ax : matplotlib Axes, optional
        If not provided then will generate our own axes.
    fig : matplotlib.figure.Figure
    date_format : str, optional, default='%d/%m/%y %H:%M:%S'
    tz_localize : boolean, optional, default is True
        if False then display UTC times.
    plot_kwargs:
        Can also use all **plot_kwargs expected by `ax.plot`
    """
    if series is None or len(series) == 0:
        return ax

    if ax is None:
        ax = plt.gca()

    if fig is None:
        fig = plt.gcf()

    x = _to_ordinalf_np_vectorized(series.index.to_pydatetime())
    ax.plot(x, series, **plot_kwargs)
    tz = series.index.tzinfo if tz_localize else None
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(date_format, tz=tz))
    ax.set_ylabel('watts')
    fig.autofmt_xdate()
    return ax


def plot_pairwise_heatmap(df, labels, edgecolors='w', cmap=mpl.cm.RdYlBu_r, log=False):
    """
    Plots a heatmap of a 'square' df
    Rows and columns are same and the values in this dataframe
    correspond to the computation b/w row,column.
    This plot can be used for plotting pairwise_correlation
    or pairwise_mutual_information or any method which works
    similarly
    """
    width = len(df.columns) / 4
    height = len(df.index) / 4

    fig, ax = plt.subplots(figsize=(width, height))

    heatmap = ax.pcolor(
        df,
        edgecolors=edgecolors,  # put white lines between squares in heatmap
        cmap=cmap,
        norm=mpl.colors.LogNorm() if log else None)

    ax.autoscale(tight=True)  # get rid of whitespace in margins of heatmap
    ax.set_aspect('equal')  # ensure heatmap cells are square
    ax.xaxis.set_ticks_position('top')  # put column labels at the top
    # turn off ticks:
    ax.tick_params(bottom='off', top='off', left='off', right='off')

    plt.yticks(np.arange(len(df.index)) + 0.5, labels)
    plt.xticks(np.arange(len(df.columns)) + 0.5, labels, rotation=90)

    # ugliness from http://matplotlib.org/users/tight_layout_guide.html
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "3%", pad="1%")
    plt.colorbar(heatmap, cax=cax)

#endregion



################################################################
# region Configurations

def latexify(fig_width=None, fig_height=None, columns=1, fontsize=10):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Din A4: 8.267 x 11.692 inches

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert columns in [1, 2]

    if fig_width is None:
        fig_width = 3.39 if columns == 2 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (math.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:", fig_height,
              "so will reduce to", MAX_HEIGHT_INCHES, "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {
        #'backend': 'ps',
        #'text.latex.preamble': ['\\usepackage{gensymb}'],
        'axes.labelsize': fontsize,  # fontsize for x and y labels (was 10)
        'axes.titlesize': fontsize,
        'font.size': fontsize,
        'legend.fontsize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        #'text.usetex': True,
        'figure.figsize': [fig_width, fig_height],
        'font.family': 'serif'
    }
    mpl.rcParams.update(params)


def format_axes(ax, spine_color='gray'):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(spine_color)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=spine_color)

#    matplotlib.pyplot.tight_layout()

    return ax

#endregion
