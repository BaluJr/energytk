''' 
Plotter to collect all plotting functionality at one place.
It uses the simple plotting functionalities of the different classes and merges 
them together.
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
        total_power.power_series_all_data(timeframe=timeframe).plot(ax = ax)
        ax = fig.add_subplot(212)
    else:
        ax = fig.add_subplot(111)
    
    # The stacked plot
    powerflows = []
    for i, dis in enumerate(disaggregations.meters):
        powerflows.append(dis.power_series_all_data(timeframe=timeframe))
    all = pd.DataFrame(data=np.array(powerflows).T)
    all.plot.area()



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

