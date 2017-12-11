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
from .metergroup import MeterGroup
from .metergroup import iterate_through_submeters_of_two_metergroups
from .electric import align_two_meters
import matplotlib.pyplot as plt
import seaborn as sns
from nilmtk import TimeFrameGroup



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
    fig = plt.figure(figsize=(50,50))#, tight_layout=True)

    # Create one bigger subplot for the overall power
    timeframe = disaggregations.get_timeframe()
    timeframe.start = timeframe.end - pd.Timedelta("20d")
    ax = fig.add_subplot(4,1,1)
    main_meter.plot(ax, timeframe=timeframe )
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
        dis.plot(sub_ax,timeframe=timeframe, plot_legend = False)
        ax.get_shared_x_axes().join(ax, sub_ax)
        ax.get_shared_y_axes().join(ax, sub_ax)
        sub_ax.set_ylim(ax.get_ylim())

    # Link the axis
    plt.setp(ax.get_xticklabels(), visible=True)
    fig.subplots_adjust(hspace=0.0)
    return fig


def plot_phases():
    '''
    Plot all three phases to see the output.
    Was included in the Nilm test.
    Don't know whether I still need it.
    '''
    new_timeframe = TimeFrameGroup([TimeFrame(start=building.elec.sitemeters()[1].get_timeframe().start, end = building.elec.sitemeters()[1].get_timeframe().start + pd.Timedelta("1d"))])
    flows = []
    for i in range(1,4):
        print(i)
        flows.append(building.elec.sitemeters()[i].power_series_all_data(sections=new_timeframe))
    all = pd.concat(flows, axis = 1)
    all.columns = ['A', 'B', 'C']
    print('Plot')
    all.plot(colors=['r', 'g', 'b'])
    print('Show')
    plt.show()




def plot_stackplot(disaggregations, total_power = None, verbose = True):
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

    timeframe = disaggregations.get_timeframe()

    # Additional total power plot if demanded
    fig = plt.figure()
    if not total_power is None:
        ax = fig.add_subplot(211)
        total_power.power_series_all_data(timeframe=timeframe, sample_period=300).plot(ax = ax)
        ax = fig.add_subplot(212)
    else:
        ax = fig.add_subplot(111)

    # The stacked plot
    powerflows = []
    all = pd.DataFrame(disaggregations.meters[0].power_series_all_data(timeframe=timeframe).rename('Rest'))
    for i, dis in enumerate(disaggregations.meters):
        if i == 0:
            continue
        name = "Appliance " + str(i)
        if verbose:
            print(name)
        all[name] = dis.power_series_all_data(timeframe=timeframe)
    all.fillna(0)
    all.iloc[:,1:].plot.area(ax = ax)
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
    '''
    # Prepare plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if ax is None:
        ax = plt.gca()
    ax.xaxis.axis_date()

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
    ax.autoscale_view()
    i = 1



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

    # Plot the original load
    for i, cur_nonzero in enumerate(sec_ground_truth):
        ax = fig.add_subplot(overall_length,3,2+i*3)
        limited = cur_nonzero.intersection(limit)
        if verbose:
            print(str(i) + ": " + str(len(limited._df)))
        limited.plot(ax=ax)
        if not gt_meters is None:
            ax.set_title(gt_meters.meters[i].appliances[0].type['type'])
        ax.set_xlim([timeframe.start, timeframe.end])
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)

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

    return fig



def plot_multiphase_event(original_powerflows, separated_powerflows, amount_of_examples = 1, surrounding = 10):
    ''' This function is used to plot multiphase events.
    It shows how the multiphase events are cut out and put inside separate poweflows.

    Parameters
    ----------
    original_powerflows: [pd
        The original transients
    separated_powerflows:
        The separated transients
    amount_of_examples:
        How many examples shall be arranged together.
    surrounding: int
        Number of events in the original power flows
        which are incorporated into the load.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The newly plotted figure
    '''
    raise Exception("Not yet fully implemented!")

    fig, ax = plt.subplots(figsize=(50,50)) #, tight_layout=True)

    plot.plot_powerflow_from_events(
        transients[a][firsts.values[1]:last.values[1]], transients[a].loc[common_transients.index][firsts.values[1]:last.values[1]])
#endregion



################################################################
#region Cluster plotting
def plot_clustering(clusterers, elements, columns_to_project):
    '''
    Plotting of points in 2d space. For K-means and gmm the bordes are also plotted.

    Paramters
    ---------

    Returns
    -------
    '''

    if len(columns_to_project) == 2:
        fig = plot_clustering_2d(clusterers, elements, columns_to_project)
    elif len(columns_to_project) == 3:
        fig = plot_clustering_3d(clusterers, elements, columns_to_project)
    else:
        raise Exception("Only 2d or 3d plot possible.")
    return fig


def plot_clustering_2d(clusterers, elements, columns_to_project):
    '''
    Plotting of points in 2d space. For K-means and gmm the bordes are also plotted.
    '''
    print_ellipse = False
    print_ellipse = True
    filter = True
    filter = False
    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold', 'darkorange'])

    # Transform the points into the coordinate system of the covar ellipsoid
    cur_X = cur_X - mean
    v, w = np.linalg.eigh(covar)
    std_dev = np.sqrt(v) # * np.sqrt(2)
    transformed_points = cur_X.dot(w.T)

    tst = spatial.distance.mahalanobis(cur_X, mean, linalg.inv(covar))
    tst = spatial.distance.cdist(cur_X+mean, np.expand_dims(mean, 0), 'mahalanobis', VI=linalg.inv(covar))
    # If not 90% sure take at least the ones inside the one sigma environment
    confident_tmp = (np.sum(transformed_points**2 / (1*std_dev)**2 , axis = 1) < 1)
    for confidence_intervall in range(1.1,2.6,0.1):
        confident_tmp.append(np.sum(transformed_points**2 / (confidence_intervall*std_dev)**2 , axis = 1) < 1)
    confident[cur] |= (np.sum(transformed_points**2 / (1*std_dev)**2 , axis = 1) < 1)
    tst = spatial.distance.mahalanobis(cur_X, mean, linalg.inv(covar))

    # Plot an ellipse to show the Gaussian component
    plt.scatter((cur_X+mean)[:,2], (cur_X+mean)[:, 3], 5, c=confident[cur])
    if print_ellipse:
        v = 2. * np.sqrt(2.) * np.sqrt(v)      #Wurzeln, weil StdDev statt varianz, *2 da Diameter statt Radius, *sqrt(2)??
        #u = w[0] / linalg.norm(w[0])          Normalisiert ist es eigentlich schon!
        angle = np.arctan(w[0][1] / w[0][0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color='gold')
        ell.set_clip_box(fig.bbox)
        ell.set_alpha(0.5)
        fig.axes[0].add_artist(ell)


def plot_clustering_3d(clusterers, elements, columns_to_project, appliances = False):
    '''
    Plotting of points in 3d space with optional colouring after assignments.
    '''
    ax = plt.axes(projection='3d');
    ax.scatter(tst[('active transition', 'avg')].values,tst[('duration', 'max')].values, tst[('spike', 'max')].values, s=0.1)
    events.plot.scatter(('active transition', 'avg'),('duration', 'log'), c=(events['color']), s=1, colormap='gist_rainbow')
    events.plot.scatter(('active transition', 'avg'),('duration', 'max'), c=events['color'], s=1)



def plot_results(X, Y_, means, covariances, index, title):
    '''
    Function to plot the results of a GMM fitting.
    I do not know exactly where it comes from.
    '''

    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

        plt.xlim(-9., 5.)
        plt.ylim(-3., 6.)
        plt.xticks(())
        plt.yticks(())
        plt.title(title)
#endregion



################################################################
#region Forecast plotting
def plot_forecasting(trained, learnspan,  forecast, forecastspan):
    '''
    Plots the forecast and the real powerflow next to each other.
    trained: Das gemachte Training
    l
    '''
    metric_names = [metrics_label_dictionary[metric] for metric in metrics]
    result = pd.DataFrame(columns = metric_names)
    for appliance in ground_truth:
        cur_metric_results = []
        for metric in metrics:
            fn = metrics_func_dictionary[metric]
            cur_metric_results = fn(prediction, appliance)
        result.loc[appliance.name] = cur_metric_results


def plot_forecast(gt, forecaster):
    # Aus dem SARIMAX Forecaster genommen

    # Plot the forecast
    series_to_plot = pd.concat([powerflow, forecast], axis = 1).fillna(0)
    series_to_plot.plot()
    pyplot.show()
    i = abs['tst']

    # Plot residual errors
    residuals =pd.DataFrame(model_fit.resid)
    residuals.plot()
    pyplot.show()
    residuals.plot(kind='kde')
    pyplot.show()
    print(residuals.describe())

    # Print the summary
    print(model_fit.summary())
    print("############ ############ ############ ############ ############")
    print(model_fit2.summary())


def plot_forecast(forecast, original_load, interval = None, ax = None, additional_data =  {}):
    ''' Plots the forecast along the real powerflow
    Paramters
    ---------
    forecast: pd.Series
        The forecasted values
    original_load: pd.Series
        The original load. Series contains at least interval of forecast.
    interval: pd.Timeframe
        Optional definition of the reagion to plot.
    additional_data: [pd.Dataframe] or [pd.Series]
        Additional data, which can be plotted. For example the residuals of the
        ARIMA model.
    '''

    if interval is None:
        load = original_load[forecast.index[0]-pd.Timedelta("24h"):forecast.index[-1]+pd.Timedelta("24h")]
    else:
        load = original_load[interval.start:interval.end]

    if ax is None:
        fig, ax = plt.subplots()

    forecast.plot(ax=ax)
    load.plot(ax=ax)


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
    fig, ax = plt.subplots(figsize=(8,6))#grps.plot(kind='kde', ax=ax, legend = None)
    for events in events_list:
        events[column].cumsum().plot(ax=ax)
    #fig, ax = plt.subplots(figsize=(8,6))#grps.plot(kind='kde', ax=ax, legend = None)
            #transients[a][firsts.values[1]:last.values[1]]['active transition'].cumsum().plot(ax=ax)
            #transients[a].loc[common_transients.index][firsts.values[1]:last.values[1]]['active transition'].cumsum().plot(ax=ax)
#endregion



################################################################
#region Originally available plots

def plot_series(series, ax=None, fig=None,
                date_format='%d/%m/%y %H:%M:%S', tz_localize=True, **plot_kwargs):
    """Plot function for series which is about 5 times faster than
    pd.Series.plot().

    Parameters
    ----------
    series : pd.Series
    ax : matplotlib Axes, optional
        If not provided then will generate our own axes.
    fig : matplotlib Figure
    date_format : str, optional, default='%d/%m/%y %H:%M:%S'
    tz_localize : boolean, optional, default is True
        if False then display UTC times.

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


def plot_pairwise_heatmap(df, labels, edgecolors='w',
                          cmap=matplotlib.cm.RdYlBu_r, log=False):
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
        norm=matplotlib.colors.LogNorm() if log else None)

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

def latexify(fig_width=None, fig_height=None, columns=1, fontsize=8):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

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
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:", fig_height,
              "so will reduce to", MAX_HEIGHT_INCHES, "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {
        'backend': 'ps',
        'text.latex.preamble': ['\\usepackage{gensymb}'],
        'axes.labelsize': fontsize,  # fontsize for x and y labels (was 10)
        'axes.titlesize': fontsize,
        'font.size': fontsize,
        'legend.fontsize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'text.usetex': True,
        'figure.figsize': [fig_width, fig_height],
        'font.family': 'serif'
    }
    matplotlib.rcParams.update(params)


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
