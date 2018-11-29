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

# do not edit! added by PythonBreakpoints
from pdb import set_trace as _breakpoint


class VEPConnector(DataStore):
    ''' 
    This DataStore sets up a connection to VEP and handles all 
    data over the REST API.

    For VEP the buildings are the different grids.
    '''

    @doc_inherit
    def __init__(self, api_address, datastore_kwargs = None, logging = False):
        '''
        Sets up the connector to the VEP plattform.
        
        Parameters
        ----------
        api_address : str
            The base url where the api can be accessed.
        username : str
            The username for login
        password : str
            The password belonging to the username.
        logging: bool
            It set to true, the sent packages are logged to 
            standard output. This is usefull to check out
            when API access is not working properly.            
        '''

        if not "username" in datastore_kwargs:
            raise Exception("Username missing for VEP API.")
        self.username = datastore_kwargs['username']

        if not "password" in datastore_kwargs:
            raise Exception("Username missing for VEP API.")
        self.password = datastore_kwargs['password']

        self.base_url = api_address
        if not self.base_url.endswith("/"):
             self.base_url =  self.base_url + "/"

        if logging:
            self._init_logging()

        #Get Token
        data = "grant_type=password&username=" + self.username +"&password=" + self.password
        r = requests.post(self.base_url + "gettoken", data = data)
        self.auth_token = json.loads(r.text)["access_token"]
        
        self.metadata = None
        super(VEPConnector, self).__init__()


    def _check_connections(self):
        '''
        Check if the connection is working.
        Has to be moved into tests later on.
        '''
       
        if logging:
            self._init_logging()

        # Test public function
        r = requests.get(self.base_url + "demo")
        data = json.loads(r.text)

        # Get Token
        data = "grant_type=password&username=" + self.username +"&password=" + self.password
        r = requests.post(self.base_url + "gettoken", data = data)
        self.token = json.loads(r.text)["access_token"]

        # Test secure function using authorization token
        header = {"Authorization": "Bearer " + self.token}
        r = requests.get(self.base_url + "demo3", headers = header)
        test = json.loads(r.text)


    def _execute_request(self, fn, parameters = {}, type = "GET", body_data = {}):
        '''
        This is the internal helper function to send a request to the VEP API.
        
        Parameters
        ----------
        fn : str
            The function to call (eg. load, save_metadata, ...)
        type: GET or POST
            The type of the http request.
        parameters:
            The parameters sent to the API. The order of the parameters is important.
            The names are currently not in use and only added for readability. Maybe in
            the future this will be added.
        body_data:
            Only used for POST requests. The payload sent within
            the body of the message.

        Returns
        -------
        The result of the server request.
        '''

        header = {"Authorization": "Bearer " + self.auth_token}
        target_url = self.base_url + "/EnergyTK/" + fn + "/" + "/".join(parameters.values())

        if type == "GET":
            r = requests.get(target_url, headers = header)
        else:
            r = requests.post(target_url, headers = header, data = body_data)

        result = json.loads(r.text)
        return result

    
    def test(self):
        ''' 
        Small helper function which is only used to initiate the
        testing on the serverside.
        '''
        self._execute_request("test")
        pass
        return


    def _init_logging(self):
        '''
        These two lines enable debugging at httplib level (requests->urllib3->http.client)
        You will see the REQUEST, including HEADERS and DATA, and RESPONSE with HEADERS but without DATA.
        The only thing missing will be the response.body which is not logged.
        '''
        try:
            import http.client as http_client
        except ImportError:
            # Python 2
            import httplib as http_client
        http_client.HTTPConnection.debuglevel = 1

        # You must initialize logging, otherwise you'll not see debug output.
        logging.basicConfig()
        logging.getLogger().setLevel(logging.DEBUG)
        requests_log = logging.getLogger("requests.packages.urllib3")
        requests_log.setLevel(logging.DEBUG)
        requests_log.propagate = True

    @doc_inherit
    def __getitem__(self, key):
        '''
        Get a certain measurement device
        Ich glaube dsa hier war einfach nur der ShortCut für DFStore.Get 
        Im Gegensatz hierzu hat DFStore.Select die möglichkeit zusätzliche Parameter anzugeben
        '''

        return self.store[key]

    def _jsonDataToPandasDF(self, columns, data):
        '''
        This is an internal helper function to translate the results of the Rest API 
        into pandas dataframes.
        
        Parameters
        ----------
        columns List<Measurement>:
            This is the list of requested measurement types. It is used to setup an
            index for the table rows.
        data : object
            The result returned from the Rest API call. From serverside description:

        Returns
        -------
        A pandas dataframe with indices as used by EnergyTk.
        '''
        timestamps = data["TimeStamps"]
        idx = pd.DatetimeIndex(timestamps, tz = "UTC")
        data = np.transpose(data["Values"])
        columns = pd.MultiIndex.from_tuples(columns, names = LEVEL_NAMES)
        df = pd.DataFrame(data=data, index=idx, columns = columns)
        return df
    
    
    
    def load(self, key, columns=None, sections=None, n_look_ahead_rows=0,
             chunksize=MAX_MEM_ALLOWANCE_IN_BYTES, verbose=False, **additionalLoaderKwargs):
        '''
        Load measurments over a certain period of time.
        '''
        # TODO: calculate chunksize default based on physical
        # memory installed and number of columns

        # Make sure key has a slash at the front but not at the end.
        if key[0] != '/':
            key = '/' + key
        if len(key) > 1 and key[-1] == '/':
            key = key[:-1]

        # Make sure chunksize is an int otherwise `range` complains later.
        chunksize = np.int64(chunksize)

        # Set `sections` variable
        sections = [TimeFrame()] if sections is None else sections
        sections = TimeFrameGroup(sections)

        # Replace any Nones with '' in cols:
        if columns is not None:
            columns = [('' if pq is None else pq, '' if ac is None else ac)
                    for pq, ac in columns]
            cols_idx = pd.MultiIndex.from_tuples(columns, names = ['physical_quantity', 'type'])
        
        columnsStr = []
        for i, val in enumerate(columns):
            columnsStr.append(str(columns[i]))
        columnsStr = str(columnsStr)

        if verbose:
            print("HDFDataStore.load(key='{}', columns='{}', sections='{}',"
                  " n_look_ahead_rows='{}', chunksize='{}')"
                  .format(key, columns, sections, n_look_ahead_rows, chunksize))

        self.all_sections_smaller_than_chunksize = True

        for section in sections:
            if verbose:
                print("   ", section)
            window_intersect = self.window.intersection(section)

            if window_intersect.empty: # Wenn der abgefragte Zeitabschnitt nicht in der Datenreihe enthalten ist
                data = pd.DataFrame(columns = cols_idx)
                data.timeframe = section
                yield data
                continue

            # The estimation of fitting slices is avoided
            delta = section.end -  section.start
            n_chunks = int(np.ceil((delta.total_seconds() / (15*60) / chunksize)))
            delta = delta/n_chunks

            slice_starts = []
            for i in range(n_chunks):
                slice_starts.append(section.start + delta * i)
            #n_chunks = int(np.ceil((section.end - section.start) / chunksize))
            #if n_chunks > 1:
                #self.all_sections_smaller_than_chunksize = False

            for chunk_i, chunk_start_i in enumerate(slice_starts):
                chunk_end_i = chunk_start_i + chunksize * pd.Timedelta('15 minutes')
                there_are_more_subchunks = (chunk_i < n_chunks-1)

                if chunk_end_i > section.end:
                    chunk_end_i = section.end

                # The required parameter form is: base={lat}/{lng}/{deviceKey}/{deviceType} + {start}/{end}/{columns}
                iso_chunk_start = chunk_start_i.isoformat()
                iso_chunk_end = chunk_end_i.isoformat()
                data = self._execute_request("load", type = "GET", parameters = {"url": key, "start": iso_chunk_start, "end": iso_chunk_end, "columns": columnsStr}) 
                data = self._jsonDataToPandasDF(columns, data)
                #data = self.store.select(key=key, columns=cols_idx, start=chunk_start_i, stop=chunk_end_i)

                if len(data) <= 2:
                    data = pd.DataFrame(columns=cols_idx)
                    data.timeframe = section
                    yield data

                # Load look ahead if necessary
                if n_look_ahead_rows > 0:
                    if len(data.index) > 0:
                        look_ahead_start_i = chunk_end_i
                        look_ahead_end_i = look_ahead_start_i + n_look_ahead_rows
                        try:
                            data.look_ahead = self.store.select(
                                key=key, columns=columns,
                                start=look_ahead_start_i,
                                stop=look_ahead_end_i)
                        except ValueError:
                            data.look_ahead = pd.DataFrame()
                    else:
                        data.look_ahead = pd.DataFrame()

                data.timeframe = _timeframe_for_chunk(there_are_more_subchunks, 
                                                      chunk_i, window_intersect,
                                                      data.index)
                yield data
                del data

    @doc_inherit
    def append(self, key, value):
        """
        Parameters
        ----------
        key : str
        value : pd.DataFrame

        Notes
        -----
        To quote the Pandas documentation for pandas.io.pytables.HDFStore.append:
        Append does *not* check if data being appended overlaps with existing
        data in the table, so be careful.
        """
        data = self._execute_request("load", type = "POST", parameters = {"id": key}, body_data = {"value": value}) 


    @doc_inherit
    def put(self, key, value, fixed = False):
        """
        Stores a new timeline inside the VEP system.

        Parameters
        ----------
        key : str
        value : pd.DataFrame
        """
        
        data = self._execute_request("put", type = "POST", parameters = {"id": key}, body_data = {"value": value}) 

    @doc_inherit
    def remove(self, key):
        """
        Removes a timeline from the tables of the VEP system.

        Parameters
        ----------
        key : str
        """
        raise NotImplementedError("Since the VEP Api has been used for reading data only. Remove data has not been implemented yet.")
    

    @doc_inherit
    def load_metadata(self, key='/'):
        """
        Tries to read metadata from VEP.

        Parameters
        ----------
        key : str
            Assumes that it is either empty or "/" to get the datasets main metadata. 
            Or it expects it to be of the form "/123" where 123 is the building number.
        """
        if self.metadata is None:
            self.metadata = self._execute_request("load_metadata", type = "GET", parameters = {"device": "key_is_not_used"})
            buildings = self.metadata["buildings"]
            self.metadata["buildings"] = {}
            for building in buildings:
                 self.metadata["buildings"][str(building["instance"])] = building
            del buildings

        if key == '/':
            return self.metadata["dataset"]
        else:
            return self.metadata["buildings"][key[1:]]


    @doc_inherit
    def save_metadata(self, key, metadata):
        """
        Tries to save metadata to VEP.

        Parameters
        ----------
        key : str
        """
        raise NotImplementedError("Since the VEP Api has been used for reading data only it is not supported to store metadata.")
    

    @doc_inherit
    def update_root_metadata(self, new_metadata):
        raise NotImplementedError("Since the VEP Api has been used for reading data only it is not supported to store metadata.")

    @doc_inherit
    def elements_below_key(self, key='/'):
        if key == "/":
            key = "-"
        metadata = self._execute_request("elements_below_key", type = "GET", parameters = {"device": key}) 
        return list(metadata)

    @doc_inherit
    def flush(self):
        """
        For the REST Connection to the API it is not necessary to 
        do a separate flush as everything is automatically sent.
        """
        pass

    @doc_inherit
    def close(self):
        """
        Not necessary to close the connection to the REST API.
        """
        pass

    @doc_inherit
    def open(self, mode='a'):
        """
        Not necessary to open the connection to the REST API.
        Automatically done when initializing the object.    
        """
        pass
        
    @doc_inherit
    def get_timeframe(self, key):
        """
        The key is immediatly in the correct form, required to do the request.
        It has the form: {lat}/{lng}/{deviceKey}/{deviceType}

        Returns
        -------
        nilmtk.TimeFrame of entire table after intersecting with self.window.
        """
        timeframe = self._execute_request("get_timeframe", type = "GET", parameters = {"url": key}) 
        start = pd.Timestamp(timeframe[0])
        end = pd.Timestamp(timeframe[1])
        timeframe = TimeFrame(start,end)
        return timeframe
        

    def store_model(self, key, row_values, blob):
        """
        The VEP API stores the row values within easily accessible table rows.
        The blob is stored at a separate location. 


        Parameters
        ----------
        key : string, optional
            The key of the measurment device
        row_values: string
            Serialized object in readable form. This part can be later 
            read without deserializing the blob.
        blob: string
            A binary object which is stored as a whole. 
        """
        metadata = self._execute_request("store_model", type = "POST", parameters = {"device": key}) 

    
    def get_model(self, key):
        """
        This function returns a model stored in the storage.


        Parameters
        ----------
        key : string, optional
            The key of the measurment device
        row_values: string
            Serialized object in readable form. This part can be later 
            read without deserializing the blob.
        blob: string
            A binary object which is stored as a whole. 
        """
        metadata = self._execute_request("get_model", type = "GET", parameters = {"device": key}) 




def _timeframe_for_chunk(there_are_more_subchunks, chunk_i, window_intersect, index):
    start = None
    end = None

    # Test if there are any more subchunks
    if there_are_more_subchunks:
        if chunk_i == 0:
            start = window_intersect.start
    elif chunk_i > 0:
        # This is the last subchunk
        end = window_intersect.end
    else:
        # Just a single 'subchunk'
        start = window_intersect.start
        end = window_intersect.end

    if start is None:
        start = index[0]
    if end is None:
        end = index[-1]

    return TimeFrame(start, end)
