'''Metrics to compare disaggregation performance against ground truth
data.

All metrics functions have the same interface.  Each function takes
`predictions` and `ground_truth` parameters.  Both of which are
nilmtk.MeterGroup objects.  Each function returns one of two types:
either a pd.Series or a single float.  Most functions return a
pd.Series where each index element is a meter instance int or a tuple
of ints for MeterGroups.

Notation
--------

Below is the notation used to mathematically define each metric. 

:math:`T` - number of time slices.

:math:`t` - a time slice.

:math:`N` - number of appliances.

:math:`n` - an appliance.

:math:`y^{(n)}_t` -  ground truth power of appliance :math:`n` in time slice :math:`t`.

:math:`\\hat{y}^{(n)}_t` -  estimated power of appliance :math:`n` in time slice :math:`t`.

:math:`x^{(n)}_t` - ground truth state of appliance :math:`n` in time slice :math:`t`.

:math:`\\hat{x}^{(n)}_t` - estimated state of appliance :math:`n` in time slice :math:`t`.

Functions
---------

'''

from __future__ import print_function, division
import numpy as np
import pandas as pd
import math
from warnings import warn
from .metergroup import MeterGroup
from .metergroup import iterate_through_submeters_of_two_metergroups
from .electric import align_two_meters
from .elecmeter import ElecMeterID
from nilmtk.timeframegroup import TimeFrame, TimeFrameGroup
from nilmtk import Electric
import itertools
import gc
import pickle as pckl
import matplotlib.pyplot as plt
import sklearn.metrics



##################################
# Error metrics from disaggregation point of view
# How we implemented it, it is possible to hand in ALL meters, but it is
# also possible to only hand in a single appliance. Perfect!
def error_in_assigned_energy(pred_meter, ground_truth_meter, etype = ("power","active")):
    """ ORIGINAL
    A) Compute error in assigned energy. OK
    The difference between the energy within the original energy and the current one.

    .. math::
        error^{(n)} = 
        \\left | \\sum_t y^{(n)}_t - \\sum_t \\hat{y}^{(n)}_t \\right |

    Parameters
    ----------
    predictions: nilmtk.MeterGroup
        Prediction meters
    ground_truth : nilmtk.MeterGroup
        Ground Truth meters
    etype: index
        The measure to base the error calculation on

    Returns
    -------
    errors : float
        Absolute error in assigned energy in KWh between prediction and the ground truth.
    """
    # Sections extra removed. Total Energy calculation incorporates it intrinsically
    #sections = pred_meter.good_sections().intersection(ground_truth_meter.good_sections())
    ground_truth_energy = ground_truth_meter.total_energy() #sections=sections)
    predicted_energy = pred_meter.total_energy()#sections=sections)
    return np.abs(ground_truth_energy - predicted_energy)


def percetage_of_assigned_energy(pred_meter, ground_truth_meter, etype = ("power","active")):
    """ B) Compute percentage of the total energy, that is assigned. OK

    Parameters
    ----------
    predictions: nilmtk.MeterGroup
        Prediction meters
    ground_truth : nilmtk.MeterGroup
        Ground Truth meters
    type: index
        The measure to base the error calculation on

    Returns
    -------
    errors : float
        Float defining error in the assigned energy in percent.
        1 means exactly same value. 2 means, that the ground truth would have been double of the predicted energy.
    """
    # Sections extra removed. Total Energy calculation incorporates it intrinsically
    #sections = pred_meter.good_sections().intersection(ground_truth_meter.good_sections())
    ground_truth_energy = ground_truth_meter.total_energy()#sections=sections)
    predicted_energy = pred_meter.total_energy()#sections=sections)
    return ground_truth_energy / predicted_energy


def deviation_of_assigned_energy(pred_meter, ground_truth_meter, etype = ("power","active")):
    """ C) Error metric

    Parameters
    ----------
    predictions: nilmtk.MeterGroup
        Prediction meters
    ground_truth : nilmtk.MeterGroup
        Ground Truth meters
    type: index
        The measure to base the error calculation on

    Returns
    -------
    errors : float
        Float defining error in the assigned energy in percent.
        1 means exactly same value. 2 means, that the ground truth would have been double of the predicted energy.
    """
    # Sections extra removed. Total Energy calculation incorporates it intrinsically
    # sections = pred_meter.good_sections().intersection(ground_truth_meter.good_sections())
    ground_truth_energy = ground_truth_meter.total_energy() #sections=sections)
    predicted_energy = pred_meter.total_energy()            #sections=sections)
    return np.abs(ground_truth_energy - predicted_energy) / ground_truth_energy


def rms_error_power(pred_meter, ground_truth_meter, etype = ("power","active")):
    ''' ORIGINAL
    D) RMSE, RMSD
    Compute RMS error in assigned power
    
    .. math::
            error^{(n)} = \\sqrt{ \\frac{1}{T} \\sum_t{ \\left ( y_t - \\hat{y}_t \\right )^2 } }

    Parameters
    ----------
    predictions: nilmtk.MeterGroup
        Prediction meters
        Can be high resolution meters with a sampling rate of 0.
        These load profiles are stored by only storing the flags of the powerflow.
    ground_truth : nilmtk.MeterGroup
        Ground Truth meters.
        The GroundTruth Meter has to be a real sampled Meter, and not a high resolution meter
        with a sampling_rate of 0. These meters are created during disaggregation.
    etype: index
        The measure to base the error calculation on

    Returns
    -------
    error : pd.Series
        Each index is an meter instance int (or tuple for MeterGroups).
        Each value is the RMS error in predicted power for that appliance.
    '''

    if len(pred_meter.meters) == 0:
        return ground_truth_meter.total_energy() / ground_truth_meter.get_timeframe().timedelta.total_seconds()

    sum_of_squared_diff = 0.0
    n_samples = 0
    for aligned_meters_chunk in align_two_meters(ground_truth_meter, pred_meter, sample_period = 10): #gt is master
        diff = aligned_meters_chunk['master'] - aligned_meters_chunk['slave']
        diff.dropna(inplace=True)
        sum_of_squared_diff += (diff ** 2).sum()
        n_samples += len(diff)
    return math.sqrt(sum_of_squared_diff / n_samples)



def mae(pred_meter, ground_truth_meter, etype = ("power","active")):
    ''' E) This function calculates the mean average error    Parameters

    Parameters
    ----------
    predictions: nilmtk.MeterGroup
        Prediction meters
        Can be high resolution meters with a sampling rate of 0.
        These load profiles are stored by only storing the flags of the powerflow.
    ground_truth : nilmtk.MeterGroup
        Ground Truth meters.
        The GroundTruth Meter has to be a real sampled Meter, and not a high resolution meter
    '''

    #If no errors assigned, all power of the ground truth belongs to error
    if len(pred_meter.meters) == 0:
        return ground_truth_meter.total_energy() / ground_truth_meter.get_timeframe().timedelta.total_seconds()

    sum_of_absolute_error = 0.0
    n_samples = 0
    for aligned_meters_chunk in align_two_meters(ground_truth_meter, pred_meter, sample_period = 10):
        diff = aligned_meters_chunk['master'] - aligned_meters_chunk['slave']
        diff.dropna(inplace=True)
        sum_of_absolute_error += diff.abs().sum()
        n_samples += len(diff)
    return sum_of_absolute_error / n_samples


def mean_normalized_error_power(pred_meter, ground_truth_meter, etype = ("power","active")):
    ''' ORIGINAL
    F) Compute mean normalized error in assigned power
        
    .. math::
        error^{(n)} = 
        \\frac
        { \\sum_t {\\left | y_t^{(n)} - \\hat{y}_t^{(n)} \\right |} }
        { \\sum_t y_t^{(n)} }

    Parameters
    ----------
    predictions: nilmtk.MeterGroup
        Prediction meters
        Can be high resolution meters with a sampling rate of 0.
        These load profiles are stored by only storing the flags of the powerflow.
    ground_truth : nilmtk.MeterGroup
        Ground Truth meters.
        The GroundTruth Meter has to be a real sampled Meter, and not a high resolution meter

    Returns
    -------
    mne : pd.Series
        Each index is an meter instance int (or tuple for MeterGroups).
        Each value is the MNE for that appliance.
    '''

    if len(pred_meter.meters) == 0:
        return 1

    total_abs_diff = 0.0
    sum_of_ground_truth_power = 0.0
    for aligned_meters_chunk in align_two_meters(ground_truth_meter, pred_meter, sample_period = 10):
        diff = aligned_meters_chunk['master'] - aligned_meters_chunk['slave']
        total_abs_diff += sum(abs(diff.dropna()))
        sum_of_ground_truth_power += aligned_meters_chunk.icol(1).sum()
    return total_abs_diff / sum_of_ground_truth_power



##################################
# Errors in assignment of sources

def precision(pred_meter, ground_truth_meter, etype = ("power","active")):
    '''
    Precision defines the ration of the identified true elements over the overall 
    amount of true elements. A precision of 1 means, that all true elements have 
    been identified.

    This can be efficiently calculated by using the overbasepower stats.

    Paramters
    ---------
    pred_meter: nilmtk.Elec or TimeFrameGroup
        The prediction meter from which the on-sections are extracted. 
        Can be also immediatly the TimeFrameGroup of on-sections.
    ground_truth_meter: nilmtk.Elec or TimeFrameGroup
        The prediction meter from which the on-sections are extracted. 
        Can be also immediatly the TimeFrameGroup of on-sections.

    Returns
    -------
    precision: float
        The precision as a value between 0 and 1.
        Or Nan if now correct prediction at all.
    '''
    if not type(pred_meter) is TimeFrameGroup:
        pred_meter = pred_meter.overbasepower_sections()
    if not type(ground_truth_meter) is TimeFrameGroup:
        ground_truth_meter = ground_truth_meter.overbasepower_sections()
    
    true_positives = ground_truth_meter.intersection(pred_meter).uptime()
    selected_elements = pred_meter.uptime()
    if selected_elements.total_seconds() > 0:
        return true_positives / selected_elements
    else:
        return float('nan')


def recall(pred_meter, ground_truth_meter, etype = ("power","active")):
    '''
    Recall defines the ratio how many elements have been chosen as true over the 
    amount of elements which would really have been true. A precision of 1 would mean,
    that there are no elements, which ahve been accidentally chosen as true.
    
    This can be efficiently calculated by using the overbasepower stats.
    
    Paramters
    ---------
    pred_meter: nilmtk.Elec or TimeFrameGroup
        The prediction meter from which the on-sections are extracted. 
        Can be also immediatly the TimeFrameGroup of on-sections.
    ground_truth_meter: nilmtk.Elec or TimeFrameGroup
        The prediction meter from which the on-sections are extracted. 
        Can be also immediatly the TimeFrameGroup of on-sections.

    Returns
    -------
    recall: float
        The recall as a value between 0 and 1.
    '''
    if not type(pred_meter) is TimeFrameGroup:
        pred_meter = pred_meter.overbasepower_sections()
    if not type(ground_truth_meter) is TimeFrameGroup:
        ground_truth_meter = ground_truth_meter.overbasepower_sections()
    
    true_positives = ground_truth_meter.intersection(pred_meter).uptime()
    relevant_elements = ground_truth_meter.uptime()
    return true_positives / relevant_elements


def f1_score(pred_meter, ground_truth_meter, etype = ("power","active")):
    ''' ORIGINAL
    I Compute F1 scores.
    Harmonic mean of precision and recall.

    .. math::
        F_{score}^{(n)} = \\frac
            {2 * Precision * Recall}
            {Precision + Recall}

    Paramters
    ---------
    pred_meter: nilmtk.Elec or TimeFrameGroup
        The prediction meter from which the on-sections are extracted. 
        Can be also immediatly the TimeFrameGroup of on-sections.
    ground_truth_meter: nilmtk.Elec or TimeFrameGroup
        The prediction meter from which the on-sections are extracted. 
        Can be also immediatly the TimeFrameGroup of on-sections.

    Returns
    -------
    recall: float
        The recall as a value between 0 and 1.
    '''
    if not type(pred_meter) is TimeFrameGroup:
        pred_meter = pred_meter.overbasepower_sections()
    if not type(ground_truth_meter) is TimeFrameGroup:
        ground_truth_meter = ground_truth_meter.overbasepower_sections()
        
    true_positives = ground_truth_meter.intersection(pred_meter).uptime()
    relevant_elements = ground_truth_meter.uptime()
    selected_elements = pred_meter.uptime()

    precision = true_positives / relevant_elements
    recall = true_positives / relevant_elements

    if (precision + recall) > 0:
        return (2 * precision * recall / (precision + recall))
    else:
        return float('nan')



def accuracy(pred_meter, ground_truth_meter, good_sections = None, etype = ("power","active")):
    ''' The accuracy of the prediction.

    
    .. math::
        ACC = \\frac {TP + TN}{P+N}} = \frac {TP +TN}{TP + TN + FP + FN }}
        
    Paramters
    ---------
    pred_meter: nilmtk.Elec or TimeFrameGroup
        The prediction meter from which the on-sections are extracted. 
        Can be also immediatly the TimeFrameGroup of on-sections.
        IMPORTANT: I considered to have good_sections everywhere or 
        at the same places as ground_truth.
    ground_truth_meter: nilmtk.Elec or TimeFrameGroup
        The prediction meter from which the on-sections are extracted. 
        Can be also immediatly the TimeFrameGroup of on-sections.
    good_sections: optional
        As it cannot be identified whether the meters are 0 because 
        they are really 0 or because it is outside of good_sections, 
        this has to be given as additonal parameter. 
        If it is None and the other parameters are really meters, 
        the gound_truths good_sections are used. Otherwise all is considered 
        a good_section.
    Returns
    -------
    recall: float
        The recall as a value between 0 and 1.
    '''
    if good_sections is None and issubclass(type(ground_truth_meter), Electric):
        good_sections = ground_truth_meter.good_sections()
    if not type(pred_meter) is TimeFrameGroup:
        pred_meter = pred_meter.overbasepower_sections()
    if not type(ground_truth_meter) is TimeFrameGroup:
        ground_truth_meter = ground_truth_meter.overbasepower_sections()
        
    cur_matches = pred_meter.matching(ground_truth_meter)
    # Throw away part in invalid area:
    cur_matches = cur_matches.intersection(good_sections)
    matching_frac = cur_matches.uptime() / good_sections.uptime()
    return matching_frac


def mcc(pred_meter, ground_truth_meter, good_sections = None, etype = ("power","active")):
    ''' Calculates Matthews correlation coefficient.
    
    This is a good measurement to define the assignment of a 
    disaggregation to ground_truth element. It balances the error 
    for False Positives and False Negatives also when the groups are 
    different in size. This avoids the problem, that a ground_truth
    element which is always off, will be the assignment target for 


    
    .. math::
        MCC =\\frac  {TP \\times TN-FP \\times FN}{{\\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}}
        
    Paramters
    ---------
    pred_meter: nilmtk.Elec or TimeFrameGroup
        The prediction meter from which the on-sections are extracted. 
        Can be also immediatly the TimeFrameGroup of on-sections.
        IMPORTANT: I considered to have good_sections everywhere or 
        at the same places as ground_truth.
    ground_truth_meter: nilmtk.Elec or TimeFrameGroup
        The prediction meter from which the on-sections are extracted. 
        Can be also immediatly the TimeFrameGroup of on-sections.
    good_sections: optional
        As it cannot be identified whether the meters are 0 because 
        they are really 0 or because it is outside of good_sections, 
        this has to be given as additonal parameter. 
        If it is None and the other parameters are really meters, 
        the gound_truths good_sections are used. Otherwise all is considered 
        a good_section.
    Returns
    -------
    recall: float
        The recall as a value between 0 and 1.
    '''
    if good_sections is None and issubclass(type(ground_truth_meter), Electric):
        good_sections = ground_truth_meter.good_sections()
    if not type(pred_meter) is TimeFrameGroup:
        pred_meter = pred_meter.overbasepower_sections()
    if not type(ground_truth_meter) is TimeFrameGroup:
        ground_truth_meter = ground_truth_meter.overbasepower_sections()

    TP, TN, FP, FN = pred_meter.get_TP_TN_FP_FN(ground_truth_meter)
    TP, TN = TP.uptime().total_seconds(), TN.uptime().total_seconds()
    FP, FN = FP.uptime().total_seconds(), FN .uptime().total_seconds()

    mcc  = (TP*TN - FP*FN) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

    return mcc


##################################
# From here on there are only metrics which give one single value for the
# overall disaggregation.

def _pre_matching(prediction, ground_truth, metric, verbose = False):
    """
    This help function matches the disaggregated appliances together using the given metric.
    The chosen metric should be a fast one like MCC which operates on the whole powerflows
    at once by using the TimeFrameGroups.

    Parameters
    ----------
    prediction: nilmtk.elec
        The prediction, which has been made. Be it from forecasting or disaggregation.
        Then a pd.Series or pd.DataFrame is given it is expected to be a good_section
        without holes.
    ground_truth: nilmtk.elec
        The real powerflow, the prediction is compared to.
        Then a pd.Series or pd.DataFrame is given it is expected to be a good_section
        without holes.
    metric: str
        Has to be a metric, defined in the above dictionary 'metrics_dictionary'.
        When set, the appliances are first pre matched using the given metric.
        Using a fast metric here, avoids the combinatorical explosion during the
        calculation of the other metrics later on.
        Fast metrics are those based on stats instead of the powerflow. (eg. MCC)
        (When some appliances fit perfectly to a certain appliance,
        they are combined as a single instance.)

    Returns
    -------
    assigned_metegroups:
        A metergroup for each ground_truth element. The list contains the groups in the same order
        as the ground_truth elements in the list handed in.

    Note
    ----
        ToDo: Currently always mcc is used. Replace by the metric handed in. Should be VERY easy. The better
        flag shall define what is better for a certain metric: higher or lower.
    """
    # For acceleration preload all sections
    pred_abovebaseload_sec = []
    gt_abovebaseload_sec = []
    gt_good_sec = []
    for pred in prediction.meters:
        pred_abovebaseload_sec.append(pred.overbasepower_sections(verbose=verbose))
    assignments = []
    for gt in ground_truth.meters:
        gt_abovebaseload_sec.append(gt.overbasepower_sections(verbose=verbose))
        gt_good_sec.append(gt.good_sections(verbose=verbose))
        assignments.append([])

    # Select the combination, which maximizes the matching on times
    for i, pred_sec in enumerate(pred_abovebaseload_sec):
        best = 0
        assign_to = -1
        for j, gt_sec in enumerate(gt_abovebaseload_sec):
            if verbose:
                print(str(i) + ":" + str(j))
            score = mcc(pred_sec, gt_sec, gt_good_sec[j])
            if score > best:
                best = score
                assign_to = j
        if assign_to != -1:
            assignments[assign_to].append(i)

    # pckl.dump(assignments, open('tst_assignments.pckl','wb'))
    # plot.plot_evaluation_assignments(gt_abovebaseload_sec, pred_abovebaseload_sec, assignments, ground_truth,
    #                                 TimeFrame(timeframe.start, timeframe.start + pd.Timedelta("2d")))
    # stored_assignments = pckl.load(open('tst_assignments.pckl','rb'))

    assigned_metegroups = []
    for assignment in assignments:
        cur = []
        for a in assignment:
            cur.append(prediction.meters[a])
        assigned_metegroups.append(MeterGroup(cur))
    return assigned_metegroups



# Errors in both (assignment AND amount of energy)
def total_energy_correctly_assigned(prediction, ground_truth, tolerance = 0, etype = ("power","active")):
    """ J) TECA

    Parameters
    ----------
    prediction: MeterGroup
        The result of the disaggregation
    ground_truth: MeterGroup
        The original appliances recorded by plugs and included within the 
        source dataset.
    """

    assigned_metegroups = _pre_matching(prediction, ground_truth, "mcc")
    overall_error = 0
    overall_gt_power = 0
    for gt, pred in zip(ground_truth.meters, assigned_metegroups):
        for aligned_meters_chunk in align_two_meters(gt, pred, sample_period=10):  # gt is master
            diff = aligned_meters_chunk['master'] - aligned_meters_chunk['slave']
            diff.dropna(inplace=True)
            overall_error += diff.abs().sum()
            overall_gt_power += aligned_meters_chunk['master'].sum()
    return 1 - overall_error / (2 * overall_gt_power)



def disaggregation_error(prediction, ground_truth):
    """ K) DE

    Parameters
    ----------
    prediction: MeterGroup
        The result of the disaggregation
    ground_truth: MeterGroup
        The original appliances recorded by plugs and included within the
        source dataset.
    """

    assigned_metegroups = _pre_matching(prediction, ground_truth, "mcc")
    overall_error = 0
    for gt, pred in zip(ground_truth.meters, assigned_metegroups):
        for aligned_meters_chunk in align_two_meters(gt, pred, sample_period=10):  # gt is master
            diff = aligned_meters_chunk['master'] - aligned_meters_chunk['slave']
            diff.dropna(inplace=True)
            overall_error += (diff.abs()**2).sum()
    return overall_error / 2



# Meta functions

'''
This array contains all information about the available metrics.
This comprises:
- lbl:      Human readable name of the metric used as caption in plots
- fn:       Reference to the function implementing the metric
- better:   Whether a higher or lower value is better (1 = higher, -1 = lower)
'''
metrics_dictionary = {
    'A_AssignedEnergy':
        {'lbl': "Delta Assigned Energy", 'fn':error_in_assigned_energy, 'better':-1, "usecase":["all", "one"]},
    'B_PercentageAssignedEnergy':
        {'lbl': "Delta Assigned Energy [%]", 'fn':percetage_of_assigned_energy, 'better':-1, "usecase":["all", "one"]},
    'C_DeviationOfAssignedEnergy':
        {'lbl': "Deviation Assigned Energy", 'fn': deviation_of_assigned_energy, 'better':-1, "usecase":["all", "one"]},

    'D_RMSE':
        {'lbl': "RMSE", 'fn': rms_error_power, 'better':-1, "usecase":["all", "one"]},
    'E_MAE':
        {'lbl': "MAE", 'fn': mae, 'better':-1, "usecase":["all", "one"]},
    'F_MNE':
        {'lbl': "Mean Normalized Error", 'fn': mean_normalized_error_power, 'better':-1, "usecase":["all", "one"]},

    'G_Precision':
        {'lbl': "Precision", 'fn': precision, 'better': 1, "usecase":["all", "one"]},
    'H_Recall':
        {'lbl': "Recall", 'fn': recall, 'better': 1, "usecase":["all", "one"]},
    'I_F1':
        {'lbl': "F1-Score", 'fn': f1_score, 'better': 1, "usecase":["all", "one"]},
    'Accuracy':
        {'lbl': "Accuracy", 'fn': accuracy, 'better': 1},
    'MCC':
        {'lbl': "MCC", 'fn': mcc, 'better': 1, "usecase":["all", "one"]},

    'J_TotalEnergyCorrectlyAssigned':
        {'lbl': "Correctly Assigned Energy", 'fn': total_energy_correctly_assigned, 'better': 1, "usecase":["all"]},
    "K":
        {'lbl': "", 'fn': disaggregation_error, 'better': 1, "usecase":["all"]},

    # The following errors are currently not yet included into the
    #"MAPE":
    #    {'lbl': "MAPE", 'fn': mape},
    #"NRSME":
    #    {'lbl': "NRSME", 'fn' : nrmse},
    }



def calculate_metrics_per_appliances(metrics, prediction, ground_truth, prematching, timeframe = None,
                                           type = ('power', 'active'), verbose = False):
    ''' Calculates the metrics per ground truth appliance instead for the powerflow as a whole.

    Calculates the metrics for the given ground_truth and prediction.
    Matches the appliances by refering how good the on-off regions fi That means that multiple
    disaggregated events may be counted to the same target appliance.

    Not all metrics support the per appliance calculation. This is noted in the "usecase" field of
    the metrics_dictionary.

    Paramters
    ---------
    merics: [str]
        A list of the metrics, defined in the above dictionary 'metrics_dictionary'
    prediction: nilmtk.elec
        The prediction, which has been made. Be it from forecasting or disaggregation.
        Then a pd.Series or pd.DataFrame is given it is expected to be a good_section
        without holes.
    ground_truth: nilmtk.elec
        The real powerflow, the prediction is compared to.
        Then a pd.Series or pd.DataFrame is given it is expected to be a good_section
        without holes.
    prematching: str 
        Has to be a metric, defined in the above dictionary 'metrics_dictionary'.
        When set, the appliances are first pre matched using the given metric. 
        Using a fast metric here, avoids the combinatorical explosion during the 
        calculation of the other metrics later on.
        Fast metrics are those based on stats instead of the powerflow. (eg. MCC)
        (When some appliances fit perfectly to a certain appliance, 
        they are combined as a single instance.)
    timeframe: pd.Timeframe
        The timeframe which is taken into account from prediction and ground_truth 
        to determine the error. If kept None, the intersection of their timeframe 
        is used.
    type: index
        When meters or DataFrames are given, this function defines which dimension is 
        used for determining the error. Default ('power', 'active').    
    verbose: bool
        If additional output shall be given.
    '''
    for metric in metrics:
        if not "single" in metrics_dictionary[metric]["usecase"]:
            raise Exception("Metric " + metrics_dictionary[metric]["lbl"] +
                            "can only be used for the overall powerflow")

    if timeframe is None:
        timeframe = ground_truth.get_timeframe(intersection_instead_union=True)\
            .intersection(prediction.get_timeframe(intersection_instead_union=True))

    # Calculate all given metric based on the matching
    assigned_metegroups = _pre_matching(prediction, ground_truth, verbose)
    metric_names = [metrics_dictionary[metric]["lbl"] for metric in metrics]
    result = pd.DataFrame(index = metric_names)
    cur_metric_results = {}
    for metric in metrics:
        name = metrics_dictionary[metric]['lbl']
        fn = metrics_dictionary[metric]['fn']
        for gt, pred in zip(ground_truth.meters, assigned_metegroups):
            cur_metric_results = fn(pred, gt, etype = type)
            appliance_name = gt.appliances[0].type['type']
            result.loc[name, appliance_name] = cur_metric_results
    return result







def calculate_errors_combinatorical(predictions, ground_truth, error_func):
    """ Combinatorical calculation of the error.
    Should be quite slow and has not been tested to hard. Not in use.
    Also has to be altered sothat the iterate_through_submeters_of_two_metergroups
    function is called in here. The identifier are already adapted.

    In the completly unsupervised case it is not clear which disaggregated appliance 
    belongs to which object of the ground_truth.
    In a first step it just calculates the values by random mixing. 
    This can be achieved by temporarily changing the ids to different pairs.
    Combinatorical combination leads to explosion of calculation.

    Paramters:
    predictions: 
        The timelines which have been predicted
    ground_truth: 
        The real timelines of appliances. (Has to be aligned to predictions)
    error_func: 
        The error_function to evaluate as a lambda
    """

    def _create_temp_elecmeter_identifiers(n):
        """
        This function creates temporary Ids which are used to
        pair together all elements.

        Parameters
        ----------
        n : Amount of elecmeter identifiers

        Returns
        -------
        ElecMeterID or MeterGroupID with dataset replaced with `dataset`
        """
        ids = []
        for i in range(1, n + 1):
            ids.append(ElecMeterID(instance=i, building=0, dataset="temp"))
        return ids

    # Get the one with the smaller length
    if len(ground_truth) > len(predictions):
        fewer_meters = predictions.meters
        more_meters = ground_truth.meters
    else:
        fewer_meters = ground_truth.meters
        more_meters = predictions.meters
    identifiers = _create_temp_elecmeter_identifiers(len(fewer_meters))

    # Backup the identifiers
    backup_fewer = []
    for i in range(len(fewer_meters)):
        backup_fewer = fewer_meters[i].identifier
        fewer_meters[i].identifier = identifiers[i]
    backup_more = []
    for gt in more_meters:
        backup_more.append(gt.identifier)
        gt.identifier = ElecMeterID(instance=999, building=0, dataset="temp")

    # Assign new identifiers 
    least_error = np.inf
    least_error_perm = None
    for perm in itertools.permutations(more_meters,len(fewer_meters)):
        # Restore the original identifier
        for i in range(len(more_meters)):
            more_meters[i].identifier = ElecMeterID(instance=999, building=0, dataset="temp")
        # Set the new ones for the current permutation
        for j in range(len(perm)):
            perm[j].identifier = identifiers[j]
        # Calc the metric with the current permutation set up (idents changed since by-reference) 
        error = error_in_assigned_energy(predictions, ground_truth)
        if error < least_error:
            least_error = error
            least_error_perm = perm

    # Restore the original identifiers
    for i in range(len(fewer_meters)):
        fewer_meters[i].identifier = backup_fewer[i]
    for i in range(len(more_meters)):
        more_meters[i] = backup_more[i] 



def fraction_energy_assigned_correctly(predictions, ground_truth):
    ''' ORIGINAL
    ?) Kann ich nicht zuordnen:


    Compute fraction of energy assigned correctly
    
    .. math::
        fraction = 
        \\sum_n min \\left ( 
        \\frac{\\sum_n y}{\\sum_{n,t} y}, 
        \\frac{\\sum_n \\hat{y}}{\\sum_{n,t} \\hat{y}} 
        \\right )

    Ignores distinction between different AC types, instead if there are 
    multiple AC types for each meter then we just take the max value across
    the AC types.

    Parameters
    ----------
    predictions, ground_truth : nilmtk.MeterGroup

    Returns
    -------
    fraction : float in the range [0,1]
        Fraction of Energy Correctly Assigned.
    '''


    predictions_submeters = MeterGroup(meters=predictions.submeters().meters)
    ground_truth_submeters = MeterGroup(meters=ground_truth.submeters().meters)

    fraction_per_meter_predictions = predictions_submeters.fraction_per_meter()
    fraction_per_meter_ground_truth = ground_truth_submeters.fraction_per_meter()

    fraction_per_meter_ground_truth.index = fraction_per_meter_ground_truth.index.map(lambda meter: meter.instance)
    fraction_per_meter_predictions.index = fraction_per_meter_predictions.index.map(lambda meter: meter.instance)

    fraction = 0
    for meter_instance in predictions_submeters.instance():
        fraction += min(fraction_per_meter_ground_truth[meter_instance],
                        fraction_per_meter_predictions[meter_instance])
    return fraction







######################
# Scaled errors used for forecasting
# Separated from the others as calculated: Only forwards to scikit-learn metrics
# These are just temporary solutions as the forecasting should be also stored as a 
# elecmeter in the future.
def mae(pred, ground_truth):
    ''' 
    K) This function calculates the mean average error used for 
    forecasting. 

    Parameters
    ----------
    forecast: pd.Series
        The forecasted powerflow
    original: pd.Series
        The real powerflow

    Returns
    -------
        mae: float
    '''
    pred, ground_truth = np.array(pred), np.array(ground_truth)
    return sklearn.metrics.mean_absolute_error(pred, ground_truth)


def mape(forecast, original):
    ''' 
    K) This function calculates the mean average percentage error used for 
    forecasting. It takes to electric as input, whose powerflow function
    it uses.

    Parameters
    ----------
    forecast: pd.Series
        The forecasted powerflow
    original: pd.Series
        The real powerflow

    Returns
    -------
        mape: float
    '''
    forecast, original = np.array(forecast), np.array(original)
    return np.mean(np.abs((forecast - original) / forecast)) * 100


def nrmse(forecast, original):
    ''' M) Normalized Root-Mean-Square Error (NRMSE)

    Parameters
    ----------
    forecast: pd.Series
        The forecasted powerflow
    original: pd.Series
        The real powerflow

    Returns
    -------
        nrmse: float
    '''
    forecast, original = np.array(forecast), np.array(original)
    tst = sklearn.metrics.mean_squared_error(forecast, original)
    return np.sqrt(tst)


# This dictionary contains the metrics which can be appliend to normal pandas frames. 
# This is different to the metrics above, which target nilmtk.Elecs. 
metrics_forecasting_dictionary = {
    'MAE':
        {'lbl': "MAE", 'fn': mae, 'better':-1},
    'MAPE':
        {'lbl': "MAPE", 'fn':mape, 'better':-1},
    'NRMSE':
        {'lbl': "NRMSE", 'fn':nrmse, 'better':-1},
}


def calc_errors_forecasting(forecasts, original_load, metrics, null_handling = 'drop'):
    '''
    Calculates the metrics for the forecasting. Is different 
    from the error calculations above, which are used for 
    NILM as there is only one fixed ground truth and the 
    data is delivered in Memory as a pd.DataFrame
    
    Paramters
    ---------
    forecasts: pd.DataFrame
        Columns are the forecastes of the timestamps aligned to 
        the original_load.
    original: pd.Series
        The original powerflow
    merics: [str]
        A list of the metrics, defined in the above dictionary 'metrics_forecasting_dictionary'
    null_handling: str ('drop' or 'interpolate')
        What shalle be done with null values. Can be ignored or interpolated
        Not used at the moment and always dropped.

    Returns
    -------
    metrics: pd.DataFrame
        An overview of all calculated metrics. The columns are the 
        different methods of forecasts. Then there is one row per 
        metric.
    '''
    # Truncate ground truth profile to interval
    original_load = original_load.reindex(forecasts.index.tz_localize('UTC'))

    # Calc the metrics
    metric_names = [metrics_forecasting_dictionary[metric]['lbl'] for metric in metrics]
    result = pd.DataFrame(columns = forecasts.columns, index = metric_names)
    for cur in forecasts.columns:
        for metric in metrics:
            metric = metrics_forecasting_dictionary[metric]
            name = metric['lbl']
            forecast = forecasts[cur]
            
            # Drop nulls
            missing = forecast.isnull()
            forecast = forecasts[cur][~missing.values]
            cur_original = original_load[~missing.values]
            result.loc[name,cur] = metric['fn'](forecast, cur_original)
    return result



###################################
# Clustering metrics
def calc_errors_correlations(orig_corrs, disag_corrs, cluster_corrs, metrics):
    '''
    This is the main function calculating error metrics for 
    correlation. It calculated the average maximum correlation

    Parameters
    ----------
    orig_corrs:
        The correlations for the different clusters
    disag_corrs
        The metergroup of disaggregations
    cluster_corrs:
        The created clustering
    metrics:
        The metrics used to 

    Results
    -------
    error_report: pd.Df
        The correlations mixed together and calculated.
    '''

    pass



def calc_errors_clustering(orig_corrs, orig_clustering, disag_corrs, disag_clustering):
    '''
    This is the main function calculating error metrics for 
    clustering.
    It calculates the signature score for each group.

    Parameter
    ----------
    orig_corrs: pd.DataFrame
        The correlations of all the original loads in the dataset
    disag_corrs:
        The correlations of the disaggregated meters.
    '''
    pass




###################################
## Error metrics from disaggregation point of view (Also for forecasting)

#def error_in_assigned_energy(predictions, ground_truth):
#    """ ORIGINAL
#    A) Compute error in assigned energy. OK
#    The difference between the energy within the original energy and the current one.

#    .. math::
#        error^{(n)} = 
#        \\left | \\sum_t y^{(n)}_t - \\sum_t \\hat{y}^{(n)}_t \\right |

#    Parameters
#    ----------
#    predictions, ground_truth : nilmtk.MeterGroup

#    Returns
#    -------
#    errors : pd.Series
#        Each index is an meter instance int (or tuple for MeterGroups).
#        Each value is the absolute error in assigned energy for that appliance,
#        in kWh.
#    """
#    errors = {}
#    both_sets_of_meters = iterate_through_submeters_of_two_metergroups(predictions, ground_truth)
#    for pred_meter, ground_truth_meter in both_sets_of_meters:
#        sections = pred_meter.good_sections().intersection(ground_truth_meter.good_sections())        
#        ground_truth_energy = ground_truth_meter.total_energy(sections=sections)
#        predicted_energy = pred_meter.total_energy(sections=sections)
#        errors[pred_meter.instance()] = np.abs(ground_truth_energy - predicted_energy)
#    return pd.Series(errors)


#def percetage_of_assigned_energy(predictions, ground_truth):
#    """ B) Compute percentage of the total energy, that is assigned. OK
#    """
#    errors = {}
#    both_sets_of_meters = iterate_through_submeters_of_two_metergroups(
#        predictions, ground_truth)
#    for pred_meter, ground_truth_meter in both_sets_of_meters:
#        sections = TimeFrameGroup.intersect_many(pred_meter.good_sections(),  ground_truth.good_sections())
#        ground_truth_energy = ground_truth_meter.total_energy(sections=sections)
#        predicted_energy = pred_meter.total_energy(sections=sections)
#        errors[pred_meter.instance()] = ground_truth_energy / predicted_energy
#    return pd.Series(errors)


#def deviation_of_assigned_energy(predictions, ground_truth):
#    """ C) Error metric
#    """
#    errors = {}
#    both_sets_of_meters = iterate_through_submeters_of_two_metergroups(
#        predictions, ground_truth)
#    for pred_meter, ground_truth_meter in both_sets_of_meters:
#        sections = pred_meter.good_sections().intersection(ground_truth_meter.good_sections())        
#        ground_truth_energy = ground_truth_meter.total_energy(sections=sections)
#        predicted_energy = pred_meter.total_energy(sections=sections)
#        errors[pred_meter.instance()] = np.abs(ground_truth_energy - predicted_energy) / ground_truth_energy
#    return pd.Series(errors)


#def rms_error_power(predictions, ground_truth):
#    ''' ORIGINAL
#    D) RMSE, RMSD
#    Compute RMS error in assigned power
    
#    .. math::
#            error^{(n)} = \\sqrt{ \\frac{1}{T} \\sum_t{ \\left ( y_t - \\hat{y}_t \\right )^2 } }

#    Parameters
#    ----------
#    predictions, ground_truth : nilmtk.MeterGroup

#    Returns
#    -------
#    error : pd.Series
#        Each index is an meter instance int (or tuple for MeterGroups).
#        Each value is the RMS error in predicted power for that appliance.
#    '''

#    error = {}

#    both_sets_of_meters = iterate_through_submeters_of_two_metergroups(
#        predictions, ground_truth)
#    for pred_meter, ground_truth_meter in both_sets_of_meters:
#        sum_of_squared_diff = 0.0
#        n_samples = 0
#        for aligned_meters_chunk in align_two_meters(pred_meter, 
#                                                     ground_truth_meter):
#            diff = aligned_meters_chunk.icol(0) - aligned_meters_chunk.icol(1)
#            diff.dropna(inplace=True)
#            sum_of_squared_diff += (diff ** 2).sum()
#            n_samples += len(diff)

#        error[pred_meter.instance()] = math.sqrt(sum_of_squared_diff / n_samples)

#    return pd.Series(error)


#def mae(predictions, ground_truth):
#    ''' E) This function calculates the mean average percentage error used for 
#    forecasting. It takes to electric as input, whose powerflow function
#    it uses.
#    '''
#    error = {}
#    both_sets_of_meters = iterate_through_submeters_of_two_metergroups(
#        predictions, ground_truth)
#    for pred_meter, ground_truth_meter in both_sets_of_meters:
#        sum_of_absolute_error = 0.0
#        n_samples = 0
#        for aligned_meters_chunk in align_two_meters(pred_meter, 
#                                                     ground_truth_meter):
#            diff = aligned_meters_chunk.icol(0) - aligned_meters_chunk.icol(1)
#            diff.dropna(inplace=True)
#            sum_of_absolute_error += diff.abs().sum()
#            n_samples += len(diff)

#        error[pred_meter.instance()] = sum_of_absolute_error / n_samples

#    return pd.Series(error)


#def mean_normalized_error_power(predictions, ground_truth):
#    ''' ORIGINAL
#    F) Compute mean normalized error in assigned power
        
#    .. math::
#        error^{(n)} = 
#        \\frac
#        { \\sum_t {\\left | y_t^{(n)} - \\hat{y}_t^{(n)} \\right |} }
#        { \\sum_t y_t^{(n)} }

#    Parameters
#    ----------
#    predictions, ground_truth : nilmtk.MeterGroup

#    Returns
#    -------
#    mne : pd.Series
#        Each index is an meter instance int (or tuple for MeterGroups).
#        Each value is the MNE for that appliance.
#    '''

#    mne = {}
#    both_sets_of_meters = iterate_through_submeters_of_two_metergroups(
#        predictions, ground_truth)
#    for pred_meter, ground_truth_meter in both_sets_of_meters:
#        total_abs_diff = 0.0
#        sum_of_ground_truth_power = 0.0
#        for aligned_meters_chunk in align_two_meters(pred_meter, 
#                                                     ground_truth_meter):
#            diff = aligned_meters_chunk.icol(0) - aligned_meters_chunk.icol(1)
#            total_abs_diff += sum(abs(diff.dropna()))
#            sum_of_ground_truth_power += aligned_meters_chunk.icol(1).sum()

#        mne[pred_meter.instance()] = total_abs_diff / sum_of_ground_truth_power

#    return pd.Series(mne)



###################################
## Errors in assignment of sources

#def precision(predictions, ground_truth):
#    raise NotImplementedError("Can be very easily done in the exact same manner as f1 score")
#    pass

#def recall(predictions, ground_truth):
#    raise NotImplementedError("Can be very easily done in the exact same manner as f1 score")
#    pass


#def f1_score(predictions, ground_truth):
#    ''' ORIGINAL
#    I Compute F1 scores.

#    .. math::
#        F_{score}^{(n)} = \\frac
#            {2 * Precision * Recall}
#            {Precision + Recall}

#    Parameters
#    ----------
#    predictions, ground_truth : nilmtk.MeterGroup

#    Returns
#    -------
#    f1_scores : pd.Series
#        Each index is an meter instance int (or tuple for MeterGroups).
#        Each value is the F1 score for that appliance.  If there are multiple
#        chunks then the value is the weighted mean of the F1 score for
#        each chunk.
#    '''
#    # If we import sklearn at top of file then sphinx breaks.
#    from sklearn.metrics import f1_score as sklearn_f1_score

#    # sklearn produces lots of DepreciationWarnings with PyTables
#    import warnings
#    warnings.filterwarnings("ignore", category=DeprecationWarning)

#    f1_scores = {}
#    both_sets_of_meters = iterate_through_submeters_of_two_metergroups(
#        predictions, ground_truth)
#    for pred_meter, ground_truth_meter in both_sets_of_meters:
#        scores_for_meter = pd.DataFrame(columns=['score', 'num_samples'])
#        aligned_meters = align_two_meters(
#            pred_meter, ground_truth_meter, 'when_on')
#        for aligned_states_chunk in aligned_meters:
#            aligned_states_chunk.dropna(inplace=True)
#            aligned_states_chunk = aligned_states_chunk.astype(int)
#            score = sklearn_f1_score(aligned_states_chunk.icol(0),
#                                     aligned_states_chunk.icol(1))
#            scores_for_meter = scores_for_meter.append(
#                {'score': score, 'num_samples': len(aligned_states_chunk)},
#                ignore_index=True)

#        # Calculate weighted mean
#        num_samples = scores_for_meter['num_samples'].sum()
#        if num_samples > 0:
#            scores_for_meter['proportion'] = (
#                scores_for_meter['num_samples'] / num_samples)
#            avg_score = (
#                scores_for_meter['score'] * scores_for_meter['proportion']
#            ).sum()
#        else:
#            warn("No aligned samples when calculating F1-score for prediction"
#                 " meter {} and ground truth meter {}."
#                 .format(pred_meter, ground_truth_meter))
#            avg_score = np.NaN
#        f1_scores[pred_meter.instance()] = avg_score

#    return pd.Series(f1_scores)



###################################
## Errors in both

#def total_energy_correctly_assigned(disaggregations, ground_truth, tolerance = 0):
#    """ J) 
#    This is the self made error function for calculating the error of the disaggregation.
#    Parameters:
#    predictions: MeterGroup
#        The result of the disaggregation
#    ground_truth: MeterGroup
#        The original appliances recorded by plugs and included within the 
#        source dataset.
#    """

#    # Start from second, becasue first is the disaggregated load
#    errors = {}
#    for disag in disaggregations:
#        predictedActiveSections = disag.good_sections()
#        for gtLoad in ground_truth:
#            gtActiveSections = gtLoad
#        both_sets_of_meters = iterate_through_submeters_of_two_metergroups(predictions, ground_truth)
#    for pred_meter, ground_truth_meter in both_sets_of_meters:
#        sections = pred_meter.good_sections()
#        ground_truth_energy = ground_truth_meter.total_energy(sections=sections)
#        predicted_energy = pred_meter.total_energy(sections=sections)
#        errors[pred_meter.instance()] = np.abs(ground_truth_energy - predicted_energy)
#    return pd.Series(errors)


#def disaggregation_error(disaggregations, ground_truth):
#    """ K
#    """



###################################
## Scaled errors mostly used for forecasting

#def mape(pred, ground_truth):
#    ''' 
#    K) This function calculates the mean average percentage error used for 
#    forecasting. It takes to electric as input, whose powerflow function
#    it uses.
#    '''
#    main_meter = 0
#    # 1. Rausfinden ob partitiell ausführbar, dh chunkwise
#    # 2. Schauen ob implementierung in Pandas
#    # 3. Berechnen
#    # 4. Disaggregation plotfunktion schreiben
#    # 5. Wrapper um die ganzen Plot Funktionen herum schreiben
#    # 6. Die versch. Correlations einbauen

#    pred, ground_truth = np.array(pred), np.array(ground_truth)
#    return np.mean(np.abs((pred - ground_truth) / pred)) * 100


#def nrmse(disaggregations, ground_truth, tolerance = 0):
#    ''' M) Normalized Root-Mean-Square Error (NRMSE)
#    '''
#    pass


#def fraction_energy_assigned_correctly(predictions, ground_truth):
#    ''' ORIGINAL
#    ?) Kann ich nicht zuordnen:


#    Compute fraction of energy assigned correctly
    
#    .. math::
#        fraction = 
#        \\sum_n min \\left ( 
#        \\frac{\\sum_n y}{\\sum_{n,t} y}, 
#        \\frac{\\sum_n \\hat{y}}{\\sum_{n,t} \\hat{y}} 
#        \\right )

#    Ignores distinction between different AC types, instead if there are 
#    multiple AC types for each meter then we just take the max value across
#    the AC types.

#    Parameters
#    ----------
#    predictions, ground_truth : nilmtk.MeterGroup

#    Returns
#    -------
#    fraction : float in the range [0,1]
#        Fraction of Energy Correctly Assigned.
#    '''


#    predictions_submeters = MeterGroup(meters=predictions.submeters().meters)
#    ground_truth_submeters = MeterGroup(meters=ground_truth.submeters().meters)

#    fraction_per_meter_predictions = predictions_submeters.fraction_per_meter()
#    fraction_per_meter_ground_truth = ground_truth_submeters.fraction_per_meter()

#    fraction_per_meter_ground_truth.index = fraction_per_meter_ground_truth.index.map(lambda meter: meter.instance)
#    fraction_per_meter_predictions.index = fraction_per_meter_predictions.index.map(lambda meter: meter.instance)

#    fraction = 0
#    for meter_instance in predictions_submeters.instance():
#        fraction += min(fraction_per_meter_ground_truth[meter_instance],
#                        fraction_per_meter_predictions[meter_instance])
#    return fraction






















##### FUNCTIONS BELOW THIS LINE HAVE NOT YET BEEN CONVERTED TO NILMTK v0.2 #####
"""
def confusion_matrices(predicted_states, ground_truth_states):
    '''Compute confusion matrix between appliance states for each appliance

    Parameters
    ----------

    predicted_state: Pandas DataFrame of type {appliance :
         [array of predicted states]}

    ground_truth_state: Pandas DataFrame of type {appliance :
        [array of ground truth states]}

    Returns
    -------
    dict of type {appliance : confusion matrix}
    '''

    re = {}

    for appliance in predicted_states:
        matrix = np.zeros([np.max(ground_truth_states[appliance]) + 1,
                           np.max(ground_truth_states[appliance]) + 1])
        for time in predicted_states[appliance]:
            matrix[predicted_states.values[time, appliance],
                   ground_truth_states.values[time, appliance]] += 1
        re[appliance] = matrix

    return re


def tp_fp_fn_tn(predicted_states, ground_truth_states):
    '''Compute counts of True Positives, False Positives, False Negatives, True Negatives
    
    .. math::
        TP^{(n)} = 
        \\sum_{t}
        and \\left ( x^{(n)}_t = on, \\hat{x}^{(n)}_t = on \\right )
        
        FP^{(n)} = 
        \\sum_{t}
        and \\left ( x^{(n)}_t = off, \\hat{x}^{(n)}_t = on \\right )
        
        FN^{(n)} = 
        \\sum_{t}
        and \\left ( x^{(n)}_t = on, \\hat{x}^{(n)}_t = off \\right )
        
        TN^{(n)} = 
        \\sum_{t}
        and \\left ( x^{(n)}_t = off, \\hat{x}^{(n)}_t = off \\right )

    Parameters
    ----------

    predicted_state: Pandas DataFrame of type {appliance :
         [array of predicted states]}

    ground_truth_state: Pandas DataFrame of type {appliance :
        [array of ground truth states]}

    Returns
    -------
    numpy array where columns represent appliances and rows represent: [TP, FP, FN, TN]
    '''

    # assumes state 0 = off, all other states = on
    predicted_states_on = predicted_states > 0
    ground_truth_states_on = ground_truth_states > 0

    tp = np.sum(np.logical_and(predicted_states_on.values == True,
                ground_truth_states_on.values == True), axis=0)
    fp = np.sum(np.logical_and(predicted_states_on.values == True,
                ground_truth_states_on.values == False), axis=0)
    fn = np.sum(np.logical_and(predicted_states_on.values == False,
                ground_truth_states_on.values == True), axis=0)
    tn = np.sum(np.logical_and(predicted_states_on.values == False,
                ground_truth_states_on.values == False), axis=0)

    return np.array([tp, fp, fn, tn]).astype(float)


def tpr_fpr(predicted_states, ground_truth_states):
    '''Compute True Positive Rate and False Negative Rate
    
    .. math::
        TPR^{(n)} = \\frac{TP}{\\left ( TP + FN \\right )}
        
        FPR^{(n)} = \\frac{FP}{\\left ( FP + TN \\right )}

    Parameters
    ----------

    predicted_state: Pandas DataFrame of type {appliance :
         [array of predicted states]}

    ground_truth_state: Pandas DataFrame of type {appliance :
        [array of ground truth states]}

    Returns
    -------
    numpy array where columns represent appliances and rows represent: [TPR, FPR]
    '''

    tfpn = tp_fp_fn_tn(predicted_states, ground_truth_states)

    tpr = tfpn[0, :] / (tfpn[0, :] + tfpn[2, :])
    fpr = tfpn[1, :] / (tfpn[1, :] + tfpn[3, :])

    return np.array([tpr, fpr])


def precision_recall(predicted_states, ground_truth_states):
    '''Compute Precision and Recall
    
    .. math::
        Precision^{(n)} = \\frac{TP}{\\left ( TP + FP \\right )}
        
        Recall^{(n)} = \\frac{TP}{\\left ( TP + FN \\right )}

    Parameters
    ----------

    predicted_state: Pandas DataFrame of type {appliance :
         [array of predicted states]}

    ground_truth_state: Pandas DataFrame of type {appliance :
        [array of ground truth states]}

    Returns
    -------
    numpy array where columns represent appliances and rows represent: [Precision, Recall]
    '''

    tfpn = tp_fp_fn_tn(predicted_states, ground_truth_states)

    prec = tfpn[0, :] / (tfpn[0, :] + tfpn[1, :])
    rec = tfpn[0, :] / (tfpn[0, :] + tfpn[2, :])

    return np.array([prec, rec])


def hamming_loss(predicted_state, ground_truth_state):
    '''Compute Hamming loss
    
    .. math::
        HammingLoss = 
        \\frac{1}{T} \\sum_{t}
        \\frac{1}{N} \\sum_{n}
        xor \\left ( x^{(n)}_t, \\hat{x}^{(n)}_t \\right )

    Parameters
    ----------

    predicted_state: Pandas DataFrame of type {appliance :
         [array of predicted states]}

    ground_truth_state: Pandas DataFrame of type {appliance :
        [array of ground truth states]}

    Returns
    -------
    float of hamming_loss
    '''

    num_appliances = np.size(ground_truth_state.values, axis=1)

    xors = np.sum((predicted_state.values != ground_truth_state.values),
                  axis=1) / num_appliances

    return np.mean(xors)
"""


#def plot_errors_as_latex_table(df_errors, caption = None):
#    '''
#    This function calculates the metrics and returns the code for a latextable.
#    '''
#    if caption == None:
#        caption = "NILM Metrics"

#    metricsstring = ""
#    metrics = 
#    for metric in metrics:
#        metricsstring += str(metric)  + "&"
#        metricsstring += str(metric)


#    tableheader = "\begin{table}[] \centering \caption{My caption} \
#    \label{my-label} \
#    \begin{tabular}{|l|l|} \
#    \hline \hline \
#    \textbf{Element} & \textbf{Metric}    \\ \hline"

#    for 
#    Electromechanical      & Digital                \\ \hline
#    One-way communication  & Two-way communicaton   \\ \hline
#    Centralized generation & Distributed generation \\ \hline
#    Few sensors            & Sensors throughout     \\ \hline
#    Manual monitoring      & Self-monitoring        \\ \hline
#    Manual restoration     & Self-healing           \\ \hline
#    Failures and blackouts & Adaptive and islanding \\ \hline
#    Limited control        & Pervasive Control      \\ \hline
#    Few customer choices   & Many customer choices  \\ \hline
#    \end{tabular}
#    \end{table}
