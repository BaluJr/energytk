import os

class UseCaseExecuter:
    '''
    This is a testclass which standardizes the execution of 
    mutiple runs of a usecase with different configuration.
    Likt this it is possible to check for the best configuration.
    It also calculates the metric for each outcome.
    A targetfolder has to be defined for each of the exectutions 
    sothat the different results are kept.

    In a first step only working for ANN Forecaster. Later also for 
    other models included.


    Attributes
    ----------
    usecase: python type
        The type of the class which shall be executed with the different paramters
    usecase_model: python type
        The type of the model which shall be executed with the different paramters
    target_folder: str
        The folder where the results are stored
    '''
    
    def __init__(self, usecase, usecase_model, target_folder):
        '''
        Paramter
        --------
        usecase: python type
            The type of the class which shall be executed with the different paramters
        usecase_model: python type
            The type of the model which shall be executed with the different paramters
        target_folder: str
            The folder where the results are stored
        '''
        self.usecase = usecase
        self.target_folder = target_folder
        self.usecase_model = usecase_model
        if not target_folder.endswith("/"):
            target_folder += "/"
        if not os.path.isdir(target_folder):
            os.mkdir(target_folder)


    def run(self, dataset, ext_dataset, parameters):
        '''
        Runs the assigned forecaster once for each paramter configuration. 
        Later one could also introduce different approaches to set up the 
        paramters (Eg. combinatorical).

        Paramters
        ---------
        parameters: [{}]
            A list of different dictionaries with parametrizations.
            The paratmers are used to adapt the default parameters.
        dataset: nilmtk.DataSet
            The DataSet with the meters
        ext_dataset: nilmtk.ExtDataSet
            The external Data
        '''

        for i, cur in enumerate(parameters):
            cur_folder = self.target_folder + "/"+ str(i)
            cur_model = self.usecase_model(None, cur)
            cur_usecase = self.usecase(model=cur_model)
            os.mkdir(cur_folder)
            cur_usecase.train(dataset, ext_dataset, target_folder = cur_folder, verbose = True)

