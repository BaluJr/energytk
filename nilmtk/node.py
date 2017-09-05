from copy import deepcopy
from six import iteritems
from nilm_metadata import recursively_update_dict

class Node(object):
    """Abstract class defining interface for all Node subclasses,
    where a 'node' is a module which runs pre-processing or statistics
    (or, later, maybe NILM training or disaggregation).
        
    Attributes
    ----------
    requirements : Contains the requirements, the upstream.dry_run_metadata has to contain. 
                   As an example the dropoutrate node needs the device's samplerate.
    postconditions : 
    results_class : The correspoinding result class for this node.
    upstream:       The node that streams its data into this node (the source of this node's incoming data)
    See Also
    --------
    results
    folder stats
    """


    #The requirements, which have to be fullfilled by the incoming stream.
    #A dictionary with entries required in the incoming datastream.
    #If only availability of attribute necessary use 'ANY VALUE'
    #Used for within the dry_run check of the pipeline, which checks the 
    #pipeline before data is loaded.
    requirements = {}

    #Defines what will be the output of this node.
    postconditions = {}

    #The result class, which corresponds to this node
    results_class = None
    

    def __init__(self, upstream=None, generator=None):
        """
        Parameters
        ----------
        upstream : an ElecMeter or MeterGroup or a Node subclass from which 
        the data is coming from.
            Required methods:
            - dry_run_metadata
            - get_metadata
            - process (not required if `generator` supplied)
        generator : Python generator (iterator). Optional
            Used when `upstream` object is an ElecMeter or MeterGroup.
            Provides source of data.
        """
        self.upstream = upstream
        self.generator = generator
        self.results = upstream.results if not upstream == None and isinstance(upstream, Node) else {}
        self.reset()

    #region EXECUTION
    def reset(self):        
        if self.results_class is not None:
            self.results = self.results_class() # [str(results_class)] = self.results_class()

    def process(self):
        '''
        This function is overriden by the subclass. If not the
        original generator (iterator) is forwarded.
        Root node is always implemented by using the default implementation below. It
        just forwards the DataFrame iterator handed in from a DataStore.load().
        '''
        return self.generator 


    def run(self):
        """Pulls data through the pipeline.  Useful if we just want to calculate 
        some stats."""
        for _ in self.process():
            pass
    #endregion

    #region PREVALIDATION
    def check_requirements(self):
        """Checks that `self.upstream.dry_run_metadata` satisfies `self.requirements`.
        Each node calls this function when it is initialized.

        Raises
        ------
        UnsatistfiedRequirementsError
        """
        # If a subclass has complex rules for preconditions then
        # override this method.
        unsatisfied = find_unsatisfied_requirements(self.upstream.dry_run_metadata(),
                                                    self.requirements)
        if unsatisfied:
            msg = str(self) + " not satisfied by:\n" + str(unsatisfied)
            raise UnsatisfiedRequirementsError(msg)
            
    def dry_run_metadata(self):
        """Does a 'dry run' so we can validate the full pipeline before
        loading any data. Not to be called from outside the node system.
        Only used by the check_requirements function.
        For normal nodes it incorporates the postconditions. For the 
        source node, which is a elecmeter, it really inserts its metadata.

        Returns
        -------
        dict : dry run metadata
        """
        state = deepcopy(self.__class__.postconditions)
        recursively_update_dict(state, self.upstream.dry_run_metadata())
        return state
    #endregion

    #region METADATA
    def get_metadata(self):
        '''
        Recursively gets the metadata. Is the full function
        while dry_run only checks the postconditions.
        '''
        if self.results:
            metadata = deepcopy(self.upstream.get_metadata())
            results_dict = self.results.to_dict()
            recursively_update_dict(metadata, results_dict)
        else:
            # Don't bother to deepcopy upstream's metadata if 
            # we aren't going to modify it.
            metadata = self.upstream.get_metadata()
        return metadata
    #endregion


    def required_measurements(self, state):
        """
        Returns
        -------
        Set of measurements that need to be loaded from disk for this node.
        """
        return set()


class UnsatisfiedRequirementsError(Exception):
    pass


def find_unsatisfied_requirements(state, requirements):
    """
    Recursively find requirements inside a dictionary and its 
    subdictionaries. Is applied to the metadata to look for 
    certain options.

    Parameters
    ----------
    state, requirements : dict
        If a property is required but the specific value does not
        matter then use 'ANY VALUE' as the value in `requirements`.

    Returns
    -------
    list of strings describing (for human consumption) which
    conditions are not satisfied.  If all conditions are satisfied
    then returns an empty list.
    """
    unsatisfied = []

    def unsatisfied_requirements(st, req):
        for key, value in iteritems(req):
            try:
                cond_value = st[key]
            except KeyError:
                msg = ("Requires '{}={}' but '{}' not in state dict."
                       .format(key, value, key))
                unsatisfied.append(msg)
            else:
                if isinstance(value, dict):
                    unsatisfied_requirements(cond_value, value)
                elif value != 'ANY VALUE' and cond_value != value:
                    msg = ("Requires '{}={}' not '{}={}'."
                           .format(key, value, key, cond_value))
                    unsatisfied.append(msg)
    unsatisfied_requirements(state, requirements)

    return unsatisfied
