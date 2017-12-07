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


def plot_overall_power_vs_disaggregation(main_meter, disaggregations):
    """ The plot for validating the NILM algorithm. 
    Plots the disaggregation below the overall powerflow together with
    orientation lines.

    Parameters
    ----------
    predictions, ground_truth : nilmtk.MeterGroup
    """
    # def plot(self, ax=None, timeframe=None, plot_legend=True, unit='W', plot_kwargs=None, **load_kwargs):
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
        print(str(i) + "/" + str(n))
        sub_ax = fig.add_subplot(sections, 1, size_main_figure+i+1)
        dis.plot(sub_ax,timeframe=timeframe, plot_legend = False)
        # Have to do it manually as series does not allow immediate shareing
        ax.get_shared_x_axes().join(ax, sub_ax) 
        ax.get_shared_y_axes().join(ax, sub_ax) 
        #sub_ax.set_autoscale_on(False)
        sub_ax.set_ylim(ax.get_ylim())
    # Linke the axis
    plt.setp(ax.get_xticklabels(), visible=True)
    fig.subplots_adjust(hspace=0.0)
    plt.show()
    i = 5



def plot_stackplot(disaggregations, total_power = None):
    """ Plots a stackplot, which stacks all disaggregation results on top of each other.

    Parameters
    ----------
    disaggregations: Remember appliance 0 is the rest powerflow
    plot_total_power: Just for comparison an additional plot with the whole powerflow.
    Should be the same as the overall powerflow in the end
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
        all[name] = dis.power_series_all_data(timeframe=timeframe)
    all.fillna(0)
    all.iloc[:,1:].plot.area(ax = ax)


def plot_segments(transitions, steady_states, ax = None):
    '''
    This function takes the events and plots the segments.

    Paramters:
    transitions: The transitions with the 'segment' field set 
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

    for cur in firsts:
        rows = steady_states[steady_states['segment'] == cur]
        ax.fill_between(rows.index.to_pydatetime(), rows['active average'].values, 0, step='post')
    ax.autoscale_view()
    i = 1
            

    ##transitions = transitions.set_index('starts')
    ##final_frame = pd.DataFrame(index = transitions.index)
    ##for seg, events in transitions[['segment','active transition']].groupby('segment'):
    ##    final_frame[seg] = events['active transition'].cumsum()
    ##    final_frame[seg].interpolate()
    ##final_frame = final_frame.fillna(0)
    ##final_frame[final_frame  < 0] = 0
    ##final_frame.plot.area(legend=None)
    ##i = 1
    ##i = 2
    
    #    if ax is None:
    #        ax = plt.gca()
    #    ax.xaxis.axis_date()
    #    height -= gap * 2
    #    for _, row in self._df.iterrows():
    #        length = (row['section_end'] - row['section_start']).total_seconds() / SECS_PER_DAY
    #        bottom_left_corner = (mdates.date2num(row['section_start']), y + gap)
    #        rect = plt.Rectangle(bottom_left_corner, length, height,
    #                             color=color, **plot_kwargs)
    #        ax.add_patch(rect)

    #    ax.autoscale_view()
    #    return ax


    
################################################################
# Forecast plotting
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


def plot_forecast(powerflow, modelfolder, horizons):
    ''' Plots the forecast along the real powerflow
    Paramters
    ---------
    powerflow: pd.DataFrame
        The load profile for which the forecast shall be done.
    modelfolder:
        The modelfolder, in which all the trained models are placed
    models:
        The modelobjects under which the training has been performed.
    '''

    fig, ax = plt.subplots()
    powerflow.plot(ax=ax)

    # Load all the trained models
    models = []
    for i, folder in enumerate(os.walk(modelfolder)):
        models.append(C.load_model(folder + "model.cnn"))



################################################################
# Elaborate Powerflow plotting
def plot_powerflow_from_events(events_list=[], column = 'active transition'):
    fig, ax = plt.subplots(figsize=(8,6))#grps.plot(kind='kde', ax=ax, legend = None)
    for events in events_list:
        events[column].cumsum().plot(ax=ax)
    #fig, ax = plt.subplots(figsize=(8,6))#grps.plot(kind='kde', ax=ax, legend = None)
            #transients[a][firsts.values[1]:last.values[1]]['active transition'].cumsum().plot(ax=ax)
            #transients[a].loc[common_transients.index][firsts.values[1]:last.values[1]]['active transition'].cumsum().plot(ax=ax)
            
          


################################################################
# Cluster plotting
def plot_clustering(clusterers, elements, columns_to_project):
    if len(columns_to_project) == 2:
        plot_clustering_2d(clusterers, elements, columns_to_project)
    elif len(columns_to_project) == 3:
        plot_clustering_3d(clusterers, elements, columns_to_project)
    else:
        raise Exception("Only 2d or 3d plot possible.")


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


def plot_multiphase_event():
    '''
    This function is used to plot multiphase events.
    '''
    plot.plot_powerflow_from_events(transients[a][firsts.values[1]:last.values[1]], transients[a].loc[common_transients.index][firsts.values[1]:last.values[1]])