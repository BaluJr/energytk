import datetime
 
class VeniosMessage:
    '''
    This is the baseclass for all messages send to VEP.

    Static Attributes:
        ClassName : string
            The name of the class
        GUID : int
            The GUID of the message

    Attributes:
        PartitionKey : str
        RowKey : str
        Timestamp : str
        ETag : str
    '''
    ClassName = "VeniosMessage"
    GUID = 500000
    
    def __init__(self):
        self.PartitionKey = "pqr"
        self.RowKey = "def"
        self.Timestamp = datetime.datetime.now()
        self.ETag = "abc"
        
        self.MessageGuid = ""
        self.ReadBy = set()
        self.CreationTimestamp = datetime.datetime.now()
        ScenarioId = ""

        super(VeniosMessage, self).__init__()
 


class EnergyTkResultMessage(VeniosMessage):
    '''
    Baseclass for all results yielded to VEP.

    Parameters:
    duration: (datetime.timedelta)
        Amount of seconds, it took to do process the job.
    '''

    MessageId = 3300000
    EventType = 2

    def __init__(self, duration):

        self.Duration = duration
        super(EnergyTkResultMessage, self).__init__()



class EnergyTkForecastResultMessage(EnergyTkResultMessage):
    '''
    Class fpr all forecasting results yielded to VEP.
    '''
    MessageId = 3300002
    EventType = 3

    def __init__(self, modelId, duration):
        self.ModelId = modelId
        self.SimulationTimepoint = simulationTimepoint

        super(EnergyTkForecastResultMessage, self).__init__(duration)



class EnergyTkTrainForecasterResultMessage(EnergyTkResultMessage):
    '''
    Class for all results yielded by training processes.
    Remember, the messages do only signal, that the process has 
    terminated. The payload has been already written into the storage 
    at this point in time.
    '''

    MessageId = 3300001
    EventType = 3

    def __init__(self, modelId, duration):
        self.ModelId = modelId;        
        super(EnergyTkTrainForecasterResultMessage, self).__init__(duration)
 