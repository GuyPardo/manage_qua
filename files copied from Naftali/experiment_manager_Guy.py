
import typing
from typing import Iterable
from copy import deepcopy
import numpy as np
from beautifultable import BeautifulTable
from dataclasses import dataclass
import Labber
from tictoc import tic, toc
from general_utils import enumerated_product
import labber_util as lu


@dataclass
class Parameter:  # TODO - I realized this class can be used for output data as well. consider change the name
    """
    a physical parameter with name, value, units.
    """
    name: str
    value: typing.Any
    units: str = None
    is_iterated = None

    def __init__(self, name: str, value, units=None, is_iterated=None):
        """
        creates a Parameter object

        :param name: str -  name of the parameter

        :param value: any type -  value of the parameter

        :param units: str or None  - physical units of the parameter. for display in Labber, plots etc.

        :param is_iterated: bool. determines whether the value is constant or iterated. if iterated, then value should
        be a python iterable object. by default self.is_iterated is defined according to whether value is an iterable,
        but this can be changed if you want, for example, a constant value that is a list.

        """

        self.name = name
        self.value = value
        self.units = units

        if is_iterated is None:
            if isinstance(self.value, Iterable):
                self.is_iterated = True
            else:
                self.is_iterated = False
        else:
            self.is_iterated = is_iterated


class Config:  # TODO - I realized this class can be used for output data as well. consider changing the name
    """
    This is an envelope-class for a list of Parameter objects with some useful methods.
    The class is used in two main ways:
    1. to store and define a configuration of an experiment to pass to Experiment class methods.
    2. to store an output of an experiment. in this case iterated Parameters will be treated as vector logs in Labber.

    attributes: param_list - the list of Parameters
    additionally each parameter in the list is added as an attribute according to the Parameter's name attribute

    #example:
    config = Config(Parameter('frequency', 4e9, 'Hz'),
                    Parameter('amplitude',1.0,'Volts'))

    #creates a Config object with attributes:
    config.param_list = [Parameter('frequency', 4e9, 'Hz'),
                        Parameter('amplitude',1.0,'Volts')]
    config.frequency = Parameter('frequency', 4e9, 'Hz')
    config.amplitude = Parameter('amplitude',1.0,'Volts')
    """


    def __init__(self, *param_list):
        """
        creates a Config object from a list of Parameter objects, and initializes attribute self.param_list

        This is an envelope-class for a list of Parameter objects with some useful methods.
        The class is used in two main ways:
        1. to store and define a configuration of an experiment to pass to Experiment class methods.
        2. to store an output of an experiment. in this case iterated Parameters will be treated as vector logs in Labber.

        attributes: param_list - the list of Parameters
        additionally each parameter in the list is added as an attribute according to the Parameter's name attribute

        #example:
        config = Config(Parameter('frequency', 4e9, 'Hz'),
                        Parameter('amplitude',1.0,'Volts'))

        #creates a Config object with attributes:
        config.param_list = [Parameter('frequency', 4e9, 'Hz'),
                            Parameter('amplitude',1.0,'Volts')]
        config.frequency = Parameter('frequency', 4e9, 'Hz')
        config.amplitude = Parameter('amplitude',1.0,'Volts')

        :param param_list: a list of Parameter objects that is stored in self.param_list
        """
        self.param_list = list(param_list)
        for param in self.param_list:
            setattr(self, param.name, param)

    def add_parameter(self, param: Parameter):
        """
        adds a Parameter to the Config.
        :param param:Parameter to be added.
        """
        self.param_list.append(param)
        setattr(self, param.name, param)

    def set_parameter(self, **kwargs):
        """
        #TODO this function is ugly. make a new one.
        set Parameter to new value, and optionally change whether it is iterated.
        self.set_parameter(name = 'a_parameter_name', value  = new_value)
        changes the value of parameter self.a_parameter_name to new_value.
        the Parameter's is_iterated might be changed according to whether the new value is an iterable object.

        self.set_parameter(name = 'a_parameter_name', value  = new_value, is_iterated=is_iterated)
        where is_iterated is a bool, allows the user to choose manually the new is_iterated value for the Parameter.

        the choice of Parameter can also be done by index instead of by name:
        self.set_parameter(index = index, value  = new_value)
        where index is an integer
        :param kwargs: a dict with specific fields, see above.
        """
        if "name" in kwargs.keys():
            getattr(self, kwargs["name"]).value = kwargs["value"]
            if "is_iterated" in kwargs.keys():
                getattr(self, kwargs["name"]).is_iterated = kwargs["is_iterated"]
            else:
                getattr(self, kwargs["name"]).is_iterated = isinstance(kwargs["value"], Iterable)

        if "index" in kwargs.keys():
            self.param_list[kwargs["index"]] = kwargs["value"]
            if "is_iterated" in kwargs.keys():
                self.param_list[kwargs["index"]].is_iterated = kwargs["is_iterated"]
            else:
                self.param_list[kwargs["index"]].is_iterated = isinstance(kwargs["value"], Iterable)

    def get_parameters_dict(self):
        """
        get a dictionary of the Config's Parameters
        :return: dict with keys that are the Parameters names and values that are the Parameters objects themselves
        """
        d = {}
        for param in self.param_list:
            d[param.name] = param
        return d

    def get_values_dict(self):
        """
        get a dictionary of the Config's Parameters values
        :return: dict with keys that are the Parameters names and values that are the Parameters values
        """
        d = {}
        for param in self.param_list:
            d[param.name] = param.value
        return d

    def get_names_list(self):
        """
        get a list of the Config's Parameters names
        :return: list
        """
        name_list = []
        for param in self.param_list:
            name_list.append(param.name)
        return name_list

    def get_values(self):
        """
        TODO rename to get_values_list()?
        get a list of the Config's Parameters values
        :return: list
        """
        values = []
        for param in self.param_list:
            values.append(param.value)
        return values

    def get_iterables(self):
        """
        # TODO: it might make more sense to return a new Config instead. this will require changing a lot of written
           code but it could be worth it
         get a list of the Parameters in self that are iterated Parameters.
         :return: list
        """
        iter_list = []
        for param in self.param_list:
            if param.is_iterated:
                iter_list.append(param)
        return iter_list

    def get_constants(self):
        """
        # TODO: it might make more sense to return a new Config instead. this will require changing a lot of written
           code but it could be worth it
         get a list of the Parameters in self that are NOT iterated Parameters.
         :return: list
        """
        const_list = []
        for param in self.param_list:
            if not param.is_iterated:
                const_list.append(param)
        return const_list

    def get_metadata_table(self):
        """
        get a nice table with all the Parameters in self, thier values and thier units.
        :return: a BeautifulTable object
        """
        table = BeautifulTable()
        table.columns.header = ["name", "value", "units"]
        for param in self.param_list:
            if param.is_iterated:
                val = "iterated"
            else:
                val = param.value
            table.rows.append([param.name, val, param.units])
            table.set_style(BeautifulTable.STYLE_NONE)
            table.precision = 20
        return table

    def is_constant(self):
        """
        determines whether the Config is constant (i.e. whether all its Parameters are not iterated)
        :return: bool
        """
        return not bool(len(self.get_iterables()))

    def get_labber_step_list(self):
        """
        get a Labber step list from the iterated Parameters of config.

        comment: the step list order is reversed relative to the Parameters order within the Config because Labber
        always treats the first variable in the trace list as the innermost loop, and in the loop
        implementation in python it is easier  to do the opposite.

        :return: a list of dicts according to Labber's step list requirements.
        """
        iterated_config = Config(*self.get_iterables())  # build a new config with just the iterables of self
        steplist = []
        for param in iterated_config.param_list:
            if param.units:
                steplist.append(dict(name=param.name, unit=param.units, values=param.value))
            else:
                steplist.append(dict(name=param.name, values=param.value))

        steplist.reverse()
        return steplist

    def get_labber_log_list(self):
        """
        Thinking about the Config as storing output of a single Experiment run, gets a Labber log list
        corresponding to the Parameters in the Config where iterated Parameters are treated as Labber vector logs.

         :return: a list of dicts according to Labber log list requirements
        """
        loglist = []
        for param in self.param_list:
            if param.units:
                loglist.append(dict(name=param.name, unit=param.units, vector=param.is_iterated))
            else:
                loglist.append(dict(name=param.name, vector=param.is_iterated))

        return loglist

    #TODO
    def get_loop_dimension(self):
        pass

    def get_total_iteration_count(self):
        """
        returns the total loop dimensions of Config (the product of the lengths of all the iterated Parameters)
        :return: int
        """
        iter_count = 1
        for var in self.get_iterables():
            iter_count = iter_count*len(var.value)
        return iter_count

#--------------- end of config class definition ----------------------#

def get_labber_trace(output_config_list):
    """
    Get a labber trace dict from a 1D loop data
    :param output_config_list:a list of Config objects, each one storing output data from a single Experiment run
    s.t. the entire list stores the result of  a 1D  loop
    :return: a dict according to Labber trace requirements
    """
    # initialize dict
    labber_dict = {}
    # initialize keys according to the first item in the loop
    for param in output_config_list[0].param_list:
        labber_dict[param.name] = []
    # set values
    for c in output_config_list:
        for param in c.param_list:
            labber_dict[param.name].append(param.value)

    # treat vector values that are not iterated
    for param in output_config_list[0].param_list:
        if not param.is_iterated:
            labber_dict[param.name] = np.array(labber_dict[param.name])
    return labber_dict


class Experiment:
    """
    an abstract class (to be used as a parent class for user-defined classes).
    a procedure (a computation or physical experiment with some controlled hardware) that you might like to run many
    times with different configurations. and save the results to Labber
    """

    def __init__(self):
        pass

    def run(self, config: Config):
        # to be implemented in child classes
        raise NotImplemented('run method not implemented')

    def one_dimensional_sweep(self, config: Config, save_to_labber=False):
        """
        execute self.run in a loop on a certain variable parameter
        :param config: a Config object with exactly one iterated Parameters  (and the others are constants)TODO: input verification
        :param save_to_labber:bool: whether to save the data in a new labber log
        :return: a dict with two entries: 'output_config' --> a list of Config objects with the data,
                    'labber_trace'--> a dict that can be inputted to labber's addEntry method
        """

        variable_param = config.get_iterables()[0]
        result = []
        for val in variable_param.value:
            current_param = Parameter(variable_param.name, val, units=variable_param.units)
            current_config = deepcopy(config)
            current_config.set_parameter(name=current_param.name, value=current_param.value)
            result.append(self.run(current_config))

        labber_trace = get_labber_trace(result)
        if save_to_labber:
            log_name = lu.get_log_name('test_exp_new')  # TODO: automatic naming
            logfile = Labber.createLogFile_ForData(log_name, result[0].get_labber_log_list(),
                                                   Config(variable_param).get_labber_step_list())
            logfile.addEntry(get_labber_trace(result))
            logfile.setComment(str(config.get_metadata_table()))
        return dict(output_config=result, labber_trace=labber_trace)

    def sweep(self, config, save_to_labber=True, labber_log_name=None):
        """
        executes self.run(...) in an N-dimneional loop with N equals the number of iterated Parameters ("variables") in
         config.
        :param config: a Config object with some iterated Parameters ("varialbes") and some non-iterated ones ("constants)
        :param save_to_labber: bool.
        :param labber_log_name: str, optional. if not supplied use automatic naming scheme that prevents overwrite
        :return: none currently. for now I want to use the save_to_labber feature anyway.
        """

        variable_config = Config(*config.get_iterables())  # a Config with only the variables

        # the last variable is the trace parameter (inner-most loop):
        tracing_parameter = variable_config.param_list[-1]

        # get labber step list:
        step_list = variable_config.get_labber_step_list()

        # create a constant configuration for test run
        curr_config = deepcopy(config)
        for variable in variable_config.param_list:
            curr_config.set_parameter(name=variable.name, value=0)

        # test run
        test_result = self.run(curr_config)

        # get labber log list
        log_list = test_result.get_labber_log_list()

        print("setp list")
        print(step_list)
        print("log list")
        print(log_list)

        # add back the tracing parameter as an iterated Parameter
        curr_config.set_parameter(name=tracing_parameter.name, value=tracing_parameter.value)

        if save_to_labber:

            # automatic naming:
            if labber_log_name:
                log_name = labber_log_name
            else:
                class_name = type(self).__name__
                log_name = f'{class_name}_sweep'

            log_name = lu.get_log_name(log_name)  # adds automatic numbering to avoid overwrite

            # create log file
            logfile = Labber.createLogFile_ForData(log_name, log_list, step_list)

            # add comment w. metadata
            logfile.setComment(str(config.get_metadata_table()))

        outer_variables = Config(*variable_config.param_list[:-1])  # "outer" means all but the inner-most loop

        # N-dimensional loop with itertools.product: # (actually N-1 )
        for indices, vals in enumerated_product(*outer_variables.get_values()):
            # update parameters to current values:
            for i, param in enumerate(outer_variables.param_list):
                curr_config.set_parameter(name=param.name, value=vals[i])

            # do 1D sweep on the tracing parameter:
            result = self.one_dimensional_sweep(curr_config, save_to_labber=False)

            # save to labber
            print("trace")
            print(result["labber_trace"])
            if save_to_labber:
                logfile.addEntry(result["labber_trace"])

            # save in python: #TODO


class AsyncExperiment(Experiment):
    """
    an abstract class to be used as a parent for user-defined classes
    a child class of Experiment for dealing with asynchronous executions
    """

    def __init__(self):
        """
        creates an AsyncExperiment objects and initializes results (=finished results) and _async_results (unfinished
        jobs) attributes.
        """
        self._async_results = []
        self.results = []

    @classmethod
    def wait_result(async_result):
        # this should be implemented in the child class.
        pass

    def _run(self, args, **kwargs):
        """
        adds job to self._async_results list
        :param args:
        :param kwargs:
        """
        self._async_results.append(self.run(*args, **kwargs))

    def wait_results(self):
        """
        performs self.wait_result ( implemented in child clas by user) for all jobs in self._async_results
        :return:
        """
        for result in self._async_results:
            self.results.append(self.wait_result(result))


class QiskitExperimentDensityMat(AsyncExperiment):
    """
    an abstract class to be used as a parent for user-defined classes.
    an experiment done on qiskit simulator where each run is the execution of a single circuit, saving the resulting
    density matrix, and then calculating some observable(s) from it.
    """

    def __init__(self):
        super().__init__()
        self.sweep_configs = None
        self.sweep_jobs = None

    def get_circ(self, config: Config):
        # to be implemented in child class. sohuld return a qiskit.QuantmCircuit object
        raise NotImplemented('get_circ method not implemented')

    def run(self, config: Config):
        """
        runs the ciscuit :param config:Config with constant (not iterated) Parameters that are used in self.get_circ(
        ), and a parameter called backend with a qiskit backend

        :return: a qiskit job.
        """
        if config.skip_simulation.value:
            return
        job = config.backend.value.run(self.get_circ(config))
        return job

    def wait_result(self, job):
        """

        :param job: a qiskit simulation job that came from a list of circuits and returns a list of density matrices
        s.t. job.result().data(i)[ "density_matrix"] is a qiskit DensityMatrix object (i is an integer)

        :return: a list of qiskit DensityMatrix objects
        """
        #TODO support the case that job is only a single circuit
        result = []
        for i in range(len(job.result().results)):
            result.append(job.result().data(i)["density_matrix"])
        return result

    def one_dimensional_job(self, config: Config):
        """
        returns a qiskit job for calculating self.get_circ in a 1d loop. and stores it in self._async_results
        :param config:Config with exactly one iterated parameter
        :return:  a qiskit job
        """
        # verify input:
        if len(config.get_iterables()) != 1:
            raise ValueError("config must have exactly one iterable Parameter")
        variable_param = config.get_iterables()[0] # config.get_iterables() is a list of length 1
        tic()
        circs = []
        for val in variable_param.value:
            current_param = Parameter(variable_param.name, val, units=variable_param.units)
            current_config = deepcopy(config)
            current_config.set_parameter(name=current_param.name, value=current_param.value)
            # print(current_config.param_list)
            circs.append(self.get_circ(current_config))

        job = config.backend.value.run(circs)
        self._async_results.append(job)
        print('1D job sent in:')
        toc()
        print('')
        return job

    def sweep(self, config):
        """
        evaluate self.get_circ() in an N dimensional loop.
        and store each inner loop 1d job in self._async_results, and additionally in the list self.sweep_jobs.
        each 1d loop config in the N-d loop is copied and stored in the list self.sweep_config
        :param config:Config with some iterated Parameters (and possibly some constant)
        """
        self.sweep_jobs = []
        self.sweep_configs = []
        variable_config = Config(*config.get_iterables())  # a Config with only the variables

        # the last variable is the trace parameter (inner-most loop):
        tracing_parameter = variable_config.param_list[-1]

        curr_config = deepcopy(config)
        for variable in variable_config.param_list[:-1]:
            curr_config.set_parameter(name=variable.name, value=0)

        outer_variables = Config(*variable_config.param_list[:-1])  # "outer" means all but the inner-most loop
        # N-dimensional loop with itertools.product: # (actually N-1 )

        counter = 1

        num_jobs = outer_variables.get_total_iteration_count()

        for indices, vals in enumerated_product(*outer_variables.get_values()): #TODO this should be a function or a method of Config
            # update parameters to current values:
            for i, param in enumerate(outer_variables.param_list):
                curr_config.set_parameter(name=param.name, value=vals[i])

            print(f"running job {counter} out of {num_jobs}...")
            # do 1D sweep on the tracing parameter:
            print('current configuration:')
            for param in outer_variables.param_list:
                print(getattr(curr_config, param.name))

            job = self.one_dimensional_job(curr_config)
            self.sweep_jobs.append(job)
            self.sweep_configs.append(deepcopy(curr_config))
            counter = counter + 1

    def get_observables(self, config: Config, density_matrix):
        raise NotImplementedError()
        # should return an output Config object
        pass

    def get_observables_1D(self, config, job):
        """
        performs self.get_observables in a 1d loop.

        :param config: Config with exacly one iterated Parameter)

        :param job: a qiskit (simulation) job that came from a list of circuits, each one returns a single density matrix

        :return: a dict with two entries: "output_config"-> a list of Config objects, "labbe_trace"--> a dict
        according to Labber' trace requirements
        """
        # returns a dict with output config list, and labber trace

        # input verification
        if not len(config.get_iterables()) == 1:
            raise ValueError("config must have exactly one iterable Parameter")

        variable_param = config.get_iterables()[0]

        density_matrices = self.wait_result(job)
        output_config = []
        config_scalar = deepcopy(config)
        for index, density_mat in enumerate(density_matrices):
            config_scalar.set_parameter(name=variable_param.name, value=variable_param.value[index])
            output_config.append(self.get_observables(config_scalar, density_mat))

        labber_trace = get_labber_trace(output_config)

        return dict(output_config=output_config, labber_trace=labber_trace)

    def labber_read(self, config, labber_log_name=None):
        """
        creates a labber log file for nd loop according to config
        adds entries from self.sweep_jobs using self.get_observables_1d for each entry (1d trace)
        :param config: Config
        :param labber_log_name:str ,optional, otherwise use automatic naming scheme
        """

        variable_config = Config(*config.get_iterables())  # a Config with only the variables

        # the last variable is the trace parameter (inner-most loop):
        tracing_parameter = variable_config.param_list[-1]

        # get labber step list:
        step_list = variable_config.get_labber_step_list()

        # get observables from one iteration of first job: #TODO: check that a job exists
        single_run_rho = self.wait_result(self.sweep_jobs[0])[0]
        single_run_config = deepcopy(config)
        for var in variable_config.param_list:
            single_run_config.set_parameter(name=var.name, value=var.value[0])

        single_run_observables = self.get_observables(single_run_config, single_run_rho)

        # get labber log list
        log_list = single_run_observables.get_labber_log_list()

        # build labber logfile
        # automatic naming:
        if labber_log_name:
            log_name = labber_log_name
        else:
            class_name = type(self).__name__
            log_name = f'{class_name}_sweep'

        log_name = lu.get_log_name(log_name)  # adds automatic numbering to avoid overwrite
        # create log file
        logfile = Labber.createLogFile_ForData(log_name, log_list, step_list)
        # add comment w. metadata
        logfile.setComment(str(config.get_metadata_table()))

        for index, job in enumerate(self.sweep_jobs):
            print(f'reading result from job {index + 1} out of {len(self.sweep_jobs)}...')
            tic()
            result = self.get_observables_1D(self.sweep_configs[index], job)
            print('1D observables read in:')
            toc()
            print('')
            logfile.addEntry(result["labber_trace"])

####################################################################
