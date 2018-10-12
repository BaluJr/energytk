'''
Problem is, that the metadata is not defines. I therefore vote to use the following scheme:

DataSet can be kept with small adaptions:
- Number_of_building => number_of_locations which are immediatly the ElecMeters
- Schema: might be in the future adapted to what we have here

Building inspires locations (Here it becomes interesting as buildings represent an arbitrary grouping)
in my case this is the zip

ElecMeter inspires the content of one ExtDataSample (Has fewer attributes):
- device_model => data_type: The type of the dataset, that means, which values are included. 
                  I vote to have a specific model for each dataset one gathers. 
                  (Eg. one from the weather API, one derived from calendar etc...)
- name
- timeframe
- datalocation
- device_model

MeterDevice inspires the Dataset Metadata:
- model => Data_Source
- model_url => source_url
- sample_period
- measurements
- 
'''

class externaldata(object):
    """
    This class handles all external data sources which might be used in 
    relation to the buildings. This means information, which do not have
    a concrete meter of the house as a source.
    Examples: Weather data for regions or holidays for counties.
    """


