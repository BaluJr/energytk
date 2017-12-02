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
from nilmtk.timeframegroup import TimeFrameGroup
import itertools


metrics_label_dictionary = {
    'A_AssignedEnergy': "",
    'B_PercentageAssignedEnergy': "",
    'C_DeviationOfAssignedEnergy': "",
    'D_RMSE': "",
    'E_MAE': "",
    'F_': "",
    'G_Precision': "",
    'H_Recall': "",
    'i_F1':"",
    'J_TotalEnergyCorrectlyAssigned': "",
    "K": "",
    "MAPE":"",
    "NRSME":""
}
metrics_func_dictionary = {
    'A_AssignedEnergy': "",
    'B_PercentageAssignedEnergy': "",
    'C_DeviationOfAssignedEnergy': "",
    'D_RMSE': "",
    'E_MAE': "",
    'F_': "",
    'G_Precision': "",
    'H_Recall': "",
    'i_F1':"",
    'J_TotalEnergyCorrectlyAssigned': "",
    "K": "",
    "MAPE":"",
    "NRSME":""
}
#### 

def calculate_all_errors(metrics, prediction, groud_truth):
    '''
    Calculates the given metrics for the output
    '''
    metric_names = [metrics_label_dictionary[metric] for metric in metrics]
    result = pd.DataFrame(columns = metric_names)
    for appliance in ground_truth:
        cur_metric_results = []
        for metric in metrics:
            fn = metrics_func_dictionary[metric]
            cur_metric_results = fn(prediction, appliance)
        result.loc[appliance.name] = cur_metric_results

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


#### Meta funktion

# FUCK! Ich muss das doch noch mal alles anpassen, da die normale Def in der Arbeit eben nich 
# Anlagen genau funktioniert.

def create_temp_elecmeter_identifiers(n):
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
    for i in range(1, n+1):
        ids.append(ElecMeterID(instance=i, building=0, dataset="temp"))
    return ids


def calculate_error_for_unsupervised(predictions, ground_truth):
    """
    In the completly unsupervised case it is not clear which disaggregated appliance 
    belongs to which object of the ground_truth.
    In a first step it just calculates the values by random mixing. 
    This can be achieved by temporarily changing the ids to different pairs.
    """
    
    # Get the one with the smaller length
    if len(ground_truth) > len(predictions):
        fewer_meters = predictions.meters
        more_meters = ground_truth.meters
    else:
        fewer_meters = ground_truth.meters
        more_meters = predictions.meters
    identifiers = create_temp_elecmeter_identifiers(len(fewer_meters))

    # Backup the identifiers
    backup_fewer = []
    for i in range(len(fewer_meters)):
        backup_fewer = fewer_meters[i].identifier
        fewer_meters[i].identifier = identifiers[i]
    backup_more = []
    for gt in more_meters:
        backup_more.append(gt.identifier)
        gt.identifier = ElecMeterID(instance=999, building=0, dataset="temp")

    # Die jetzt assignen
    least_error = np.inf
    least_error_perm = None
    for perm in itertools.permutations(more_meters,len(fewer_meters)):
        # Restore the original identifiert
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



##################################
# Error metrics from disaggregation point of view (Also for forecasting)

# Original
def error_in_assigned_energy(predictions, ground_truth):
    """ A) Compute error in assigned energy. OK
    The difference between the energy within the original energy and the current one.

    .. math::
        error^{(n)} = 
        \\left | \\sum_t y^{(n)}_t - \\sum_t \\hat{y}^{(n)}_t \\right |

    Parameters
    ----------
    predictions, ground_truth : nilmtk.MeterGroup

    Returns
    -------
    errors : pd.Series
        Each index is an meter instance int (or tuple for MeterGroups).
        Each value is the absolute error in assigned energy for that appliance,
        in kWh.
    """
    errors = {}
    both_sets_of_meters = iterate_through_submeters_of_two_metergroups(
        predictions, ground_truth)
    for pred_meter, ground_truth_meter in both_sets_of_meters:
        sections = pred_meter.good_sections().intersection(ground_truth_meter.good_sections())        
        ground_truth_energy = ground_truth_meter.total_energy(sections=sections)
        predicted_energy = pred_meter.total_energy(sections=sections)
        errors[pred_meter.instance()] = np.abs(ground_truth_energy - predicted_energy)
    return pd.Series(errors)


def percetage_of_assigned_energy(predictions, ground_truth):
    """ B) Compute percentage of the total energy, that is assigned. OK
    """
    errors = {}
    both_sets_of_meters = iterate_through_submeters_of_two_metergroups(
        predictions, ground_truth)
    for pred_meter, ground_truth_meter in both_sets_of_meters:
        sections = TimeFrameGroup.intersect_many(pred_meter.good_sections(),  ground_truth.good_sections())
        ground_truth_energy = ground_truth_meter.total_energy(sections=sections)
        predicted_energy = pred_meter.total_energy(sections=sections)
        errors[pred_meter.instance()] = ground_truth_energy / predicted_energy
    return pd.Series(errors)


def deviation_of_assigned_energy(predictions, ground_truth):
    """ C) Error metric
    """
    errors = {}
    both_sets_of_meters = iterate_through_submeters_of_two_metergroups(
        predictions, ground_truth)
    for pred_meter, ground_truth_meter in both_sets_of_meters:
        sections = pred_meter.good_sections().intersection(ground_truth_meter.good_sections())        
        ground_truth_energy = ground_truth_meter.total_energy(sections=sections)
        predicted_energy = pred_meter.total_energy(sections=sections)
        errors[pred_meter.instance()] = np.abs(ground_truth_energy - predicted_energy) / ground_truth_energy
    return pd.Series(errors)

# Original
def rms_error_power(predictions, ground_truth):
    '''D) RMSE, RMSD
    Compute RMS error in assigned power
    
    .. math::
            error^{(n)} = \\sqrt{ \\frac{1}{T} \\sum_t{ \\left ( y_t - \\hat{y}_t \\right )^2 } }

    Parameters
    ----------
    predictions, ground_truth : nilmtk.MeterGroup

    Returns
    -------
    error : pd.Series
        Each index is an meter instance int (or tuple for MeterGroups).
        Each value is the RMS error in predicted power for that appliance.
    '''

    error = {}

    both_sets_of_meters = iterate_through_submeters_of_two_metergroups(
        predictions, ground_truth)
    for pred_meter, ground_truth_meter in both_sets_of_meters:
        sum_of_squared_diff = 0.0
        n_samples = 0
        for aligned_meters_chunk in align_two_meters(pred_meter, 
                                                     ground_truth_meter):
            diff = aligned_meters_chunk.icol(0) - aligned_meters_chunk.icol(1)
            diff.dropna(inplace=True)
            sum_of_squared_diff += (diff ** 2).sum()
            n_samples += len(diff)

        error[pred_meter.instance()] = math.sqrt(sum_of_squared_diff / n_samples)

    return pd.Series(error)

def mae(predictions, ground_truth):
    ''' E) This function calculates the mean average percentage error used for 
    forecasting. It takes to electric as input, whose powerflow function
    it uses.
    '''
    error = {}
    both_sets_of_meters = iterate_through_submeters_of_two_metergroups(
        predictions, ground_truth)
    for pred_meter, ground_truth_meter in both_sets_of_meters:
        sum_of_absolute_error = 0.0
        n_samples = 0
        for aligned_meters_chunk in align_two_meters(pred_meter, 
                                                     ground_truth_meter):
            diff = aligned_meters_chunk.icol(0) - aligned_meters_chunk.icol(1)
            diff.dropna(inplace=True)
            sum_of_absolute_error += diff.abs().sum()
            n_samples += len(diff)

        error[pred_meter.instance()] = sum_of_absolute_error / n_samples

    return pd.Series(error)




# Original
def mean_normalized_error_power(predictions, ground_truth):
    '''F) Compute mean normalized error in assigned power
        
    .. math::
        error^{(n)} = 
        \\frac
        { \\sum_t {\\left | y_t^{(n)} - \\hat{y}_t^{(n)} \\right |} }
        { \\sum_t y_t^{(n)} }

    Parameters
    ----------
    predictions, ground_truth : nilmtk.MeterGroup

    Returns
    -------
    mne : pd.Series
        Each index is an meter instance int (or tuple for MeterGroups).
        Each value is the MNE for that appliance.
    '''

    mne = {}
    both_sets_of_meters = iterate_through_submeters_of_two_metergroups(
        predictions, ground_truth)
    for pred_meter, ground_truth_meter in both_sets_of_meters:
        total_abs_diff = 0.0
        sum_of_ground_truth_power = 0.0
        for aligned_meters_chunk in align_two_meters(pred_meter, 
                                                     ground_truth_meter):
            diff = aligned_meters_chunk.icol(0) - aligned_meters_chunk.icol(1)
            total_abs_diff += sum(abs(diff.dropna()))
            sum_of_ground_truth_power += aligned_meters_chunk.icol(1).sum()

        mne[pred_meter.instance()] = total_abs_diff / sum_of_ground_truth_power

    return pd.Series(mne)



##################################
# Errors in assignment of sources

def precision(predictions, ground_truth):
    raise NotImplementedError("Can be very easily done in the exact same manner as f1 score")
    pass

def recall(predictions, ground_truth):
    raise NotImplementedError("Can be very easily done in the exact same manner as f1 score")
    pass

# Original
def f1_score(predictions, ground_truth):
    ''' I 
    Compute F1 scores.

    .. math::
        F_{score}^{(n)} = \\frac
            {2 * Precision * Recall}
            {Precision + Recall}

    Parameters
    ----------
    predictions, ground_truth : nilmtk.MeterGroup

    Returns
    -------
    f1_scores : pd.Series
        Each index is an meter instance int (or tuple for MeterGroups).
        Each value is the F1 score for that appliance.  If there are multiple
        chunks then the value is the weighted mean of the F1 score for
        each chunk.
    '''
    # If we import sklearn at top of file then sphinx breaks.
    from sklearn.metrics import f1_score as sklearn_f1_score

    # sklearn produces lots of DepreciationWarnings with PyTables
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    f1_scores = {}
    both_sets_of_meters = iterate_through_submeters_of_two_metergroups(
        predictions, ground_truth)
    for pred_meter, ground_truth_meter in both_sets_of_meters:
        scores_for_meter = pd.DataFrame(columns=['score', 'num_samples'])
        aligned_meters = align_two_meters(
            pred_meter, ground_truth_meter, 'when_on')
        for aligned_states_chunk in aligned_meters:
            aligned_states_chunk.dropna(inplace=True)
            aligned_states_chunk = aligned_states_chunk.astype(int)
            score = sklearn_f1_score(aligned_states_chunk.icol(0),
                                     aligned_states_chunk.icol(1))
            scores_for_meter = scores_for_meter.append(
                {'score': score, 'num_samples': len(aligned_states_chunk)},
                ignore_index=True)

        # Calculate weighted mean
        num_samples = scores_for_meter['num_samples'].sum()
        if num_samples > 0:
            scores_for_meter['proportion'] = (
                scores_for_meter['num_samples'] / num_samples)
            avg_score = (
                scores_for_meter['score'] * scores_for_meter['proportion']
            ).sum()
        else:
            warn("No aligned samples when calculating F1-score for prediction"
                 " meter {} and ground truth meter {}."
                 .format(pred_meter, ground_truth_meter))
            avg_score = np.NaN
        f1_scores[pred_meter.instance()] = avg_score

    return pd.Series(f1_scores)





##################################
# Errors in both

def total_energy_correctly_assigned(disaggregations, ground_truth, tolerance = 0):
    """ J) 
    This is the self made error function for calculating the error of the disaggregation.
    Parameters:
    predictions: MeterGroup
        The result of the disaggregation
    ground_truth: MeterGroup
        The original appliances recorded by plugs and included within the 
        source dataset.
    """

    # Start from second, becasue first is the disaggregated load
    errors = {}
    for disag in disaggregations:
        predictedActiveSections = disag.good_sections()
        for gtLoad in ground_truth:
            gtActiveSections = gtLoad
        both_sets_of_meters = iterate_through_submeters_of_two_metergroups(predictions, ground_truth)
    for pred_meter, ground_truth_meter in both_sets_of_meters:
        sections = pred_meter.good_sections()
        ground_truth_energy = ground_truth_meter.total_energy(sections=sections)
        predicted_energy = pred_meter.total_energy(sections=sections)
        errors[pred_meter.instance()] = np.abs(ground_truth_energy - predicted_energy)
    return pd.Series(errors)


def disaggregation_error(disaggregations, ground_truth):
    """ K
    """


##################################
# Scaled errors mostly used for forecasting

def mape(y_true, y_pred):
    ''' 
    K) This function calculates the mean average percentage error used for 
    forecasting. It takes to electric as input, whose powerflow function
    it uses.
    '''
    main_meter = 0
    # 1. Rausfinden ob partitiell ausführbar, dh chunkwise
    # 2. Schauen ob implementierung in Pandas
    # 3. Berechnen
    # 4. Disaggregation plotfunktion schreiben
    # 5. Wrapper um die ganzen Plot Funktionen herum schreiben
    # 6. Die versch. Correlations einbauen

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def nrmse(disaggregations, ground_truth, tolerance = 0):
    ''' M) Normalized Root-Mean-Square Error (NRMSE)
    '''
    pass


# Original
def fraction_energy_assigned_correctly(predictions, ground_truth):
    ''' ?) Kann ich nicht zuordnen:


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
