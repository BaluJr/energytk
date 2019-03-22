from __future__ import print_function, division
import pandas as pd
import requests
from copy import deepcopy
import numpy as np
from os.path import isfile
from nilmtk.timeframe import TimeFrame
from nilmtk.timeframegroup import TimeFrameGroup
from .datastore import DataStore, MAX_MEM_ALLOWANCE_IN_BYTES
from nilmtk.docinherit import doc_inherit
from builtins import range
import json
import logging
from nilmtk.measurement import LEVEL_NAMES
import datetime
from azure.servicebus import ServiceBusService, Message, Queue, Topic

# do not edit! added by PythonBreakpoints
from pdb import set_trace as _breakpoint



class VEPServiceBusConnector:
    '''
    This class encapsulates the access to the Events of the ServiceBus of VEP.
    It uses the same RestAPI of VEP, which is used to retrieve and submit data via the VEPConnector class.
    The receiving of messages is currently initiated by pulling.
    
    Parameters
    ---------
    connection_string : string
        Connection string of the ServiceBus to connect to.
    pulling_interval:
                The amount of milliseconds before checking for new 
                messages.       

    Attributes
    ----------
        bus_service : ServiceBusService
            The service bus, the requests are sent to
        pulling_interval :
                The amount of milliseconds before checking for new 
                messages.        
        subscriptions : The subscriptions already set up. List is used to poll
                them regularly.
    '''

    topics = { 1000000, // InfoMessages
            2000000, // ControlMessages
            3100000, // ResultMessages - Topics
            3200000, // ResultMessages - Powerflow
            3300000, // ResultMessages - EnergyTk
            4000000, // Miscellaneous
            5000000, // WorkerRoleTasks
            6000000, // ProgressMessages
            9000000 //HistoricEvents
    }

    def _determine_topic(self, msg_type):
        '''
        Determines the correct service bus topic to use for 
        sending a message of type msg_type

        Paramters
        ---------
        msg_type : int
            Type of the message as the type code.
        '''
        id = msg_type
        i = 0
        while (not id in topics):
            id = msg_type / (10 ^ i)
            i += 1
        return id


    #def ServiceBusJobListener(connectionString):
    def __init__(self, connection_string, pulling_interval = 0):
        """
        This executer is one of the execution drivers. It takes care that incoming requests are processed.
        It can built upon multiple connectors. Azure service bus is implemented. A python flask WebAPI
        is also planned. based upon: azure-servicebus v0.21.x

        Parameters:
            connection_string : string
                The connection string of the serice bus
            pulling_interval : int
                The amount of milliseconds before checking for new messages. 
                Any value other than 0 results in an active polling pattern to gather 
                new events.
                If the interval is set to 0, websockets are used for the connection. This allows 
                VEP to push messages as they appear without any delay. 
        """

        if pulling_interval == 0:
            raise Exception("Websocket connection is not yet implemented in the connector.")

        self.subscriptions = []

        # Read variables from connection string
        subparts = connectionString.split(";")
        namespace = subparts[0].split(".")[0]
        namespace = namespace[namespace.rfind("/")+1:]
        keyName = subparts[1][subparts[1].find("=")+1:]
        keyValue = subparts[2][subparts[2].find("=")+1:]

        # Connect the Service Bus and subscribe to job topic
        self.bus_service = ServiceBusService(
            service_namespace=namespace,
            shared_access_key_name=keyName,
            shared_access_key_value=keyValue)

        self.pulling_interval = pulling_interval 
        
       # Execute https://stackoverflow.com/questions/36307767/how-to-remove-strin3http-schemas-microsoft-com-2003-10-serialization-receive
        def add_inputs():
            while True:
                # Receive messages from subscription (Nice -> Jobs appear as separate per role) 
                for (subscription in subscriptions):
                    msg = bus_service.receive_subscription_message(subscription[0], subscription[1], peek_lock=False)
                    if not msg.body is None:
                        msg = _decodeMessage(msg)
                        responseMessage = subscription[2](msg)
                        self._send_message(responseMessage)
                    time.sleep(self.pulling_interval)
        input_thread = threading.Thread(target=add_inputs)
        input_thread.start()


    def subscribe(self, message_type, subcription_name, event_handler):
        '''
        Subscribes to the subscription with the given event_handler.
        Parameters
        ----------
            message_type : int
                The type id which shall be subscribed
            subcription_name : str
                The name of the subscription to be used for registering
            event_handler : fn
                The function to be called when an event is retrieved.
                The parameter of the function has to be the message.
        '''

        bus_service.create_subscription(int(message_type), subscription_name, fail_on_exist = False)
        subscriptions.append([message_type, subscription_name, event_handler])
        #bus_service.create_subscription('5000000', 'EnergyTkSubscription', fail_on_exist = False)
        



    def _decodeMessage(self, msg):
        '''
        Decodes the message from the VEP API
        There is a bug that an annotation is put arround the core message.
        This is solved here.

        Paramters:
            msg : str
                The message coming from the service bus.
        '''
        start = msg.body.find(b'{');
        end = msg.body.rfind(b'}') + 1;
        interesting_part = msg.body[start:end]
        msg = json.loads(interesting_part)
        return msg


    def _send_message(self, msg):
        '''
        Create and send the response to the VEP system

        Parameters:
            msg : string
                The original message to send
        '''
        customProperties = {'CustomMessageId': str(msg.MessageId)}
        brokerProperties = {'Label': str(msg.MessageId)}
        msg = ModelJsonfyWithTypeAnnotations(msg)
        msg = Message(msg.encode("UTF8"), custom_properties=customProperties, broker_properties = brokerProperties)
        topic = self._determine_topic(msg.MessageId)
        self.bus_service.send_topic_message(topic, msg)
    