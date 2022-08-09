# manange_qua
import itertools
import sys
import typing
from copy import deepcopy

sys.path.append("..")
sys.path.append(r"\\132.64.80.214\overlordcommon\Users\Guy\PHD\repos\experiment-manager")
import experiment_manager
#import Labber_utils as lu
sys.path.append(r"\\132.64.80.214\overlordcommon\Users\Guy\PHD\repos\manage_qua\files copied from Naftali")
# Nafteli's imports:
import two_qubit_config_gates
import importlib

importlib.reload(two_qubit_config_gates)
importlib.reload(experiment_manager)

import experiment_manager as em
from two_qubit_config_gates import *

import Labber
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from qm.qua import *
import OPX.config_generator as config_generator
import numpy as np
from matplotlib import pyplot as plt
from os.path import join
import os
import qdac as qdac_lib

plt.ion()

from dataclasses import dataclass




# ------------- utility functions------------------

def bool_list_2_int(lst):
    """
    get an int from a list of bools
    :param lst:List[bool] a list of booleans
    :return: an int that is concatenating the bools of the list into a binary number, and then converting to int.
    """
    return int(''.join(['1' if x else '0' for x in lst]), 2)






# get config dict from device params file
def get_qua_config_dict():
    cg = init_config_generator()
    create_readout_elements(cg)
    create_drive_elements(cg)
    create_pulses(cg)
    add_OPX_dc_elements(cg)
    config = cg.get_config()
    return config



def play_pulse(pulse, element, scale_amplitude=None, frequency=None, duration=None):
    """
    perform QUA's play(pulse, element) with optional dynamical change of amplitude, frequency or
    duration. If one (or more) of the three optional arguments is not supplied, default value from the QUA config
    dict will be used when the program is run

    :param pulse:str  - the name of the pulse to play as appears in the QUA config dict
    :param element:str  -  the name of the element for which to play the pulse as appears in the QUA config dict
    :param scale_amplitude:QUA-float or None  - dimensionless factor by which to scale the amplitude for pulse
    :param frequency: QUA-int or None -  new (IF) frequency for pulse in Hz
    :param duration: QUA-int or None - new duration for pulse in clock cycles = 4ns
    :return:
    """

    # the play statement supports also a None value for duratio so we only have to worry about amplitude and frequency:

    # convert the bool information of whther we change amplitude, frequency, both or none into a number from 0 to 3:
    indicator = bool_list_2_int([bool(scale_amplitude), bool(frequency)])

    # if statement:
    if indicator == 0:  # = [0,0] change nothing (only possibly duration )
        play(pulse, element, duration=duration)
    elif indicator == 1:  # = [0,1]:  change only frequency (and possibly duration)
        update_frequency(element, frequency)
        play(pulse, element, duration)
    elif indicator == 2:  # = [1,0]: change only scale_amplitude (and possibly duration)
        play(pulse * amp(scale_amplitude), element, duration=duration)
    elif indicator == 3:  # = [1,1]: change amplitude and frequency (and possibly duration)
        update_frequency(element, frequency)
        play(pulse * amp(scale_amplitude), element, duration=duration)


def iter_type(input):
    """
    get the type of the elements of iterable object input. o the type of input itself if it is not iterable.
    returns error if input has elements with different types.
    :param input: an iterable that that all its elements have the same type, or a non iterable
    :return:type :
    """
    if not isinstance(input, typing.Iterable):
        return type(input)
    else:
        first_type = type(input[0]) # get type of first element
        # verify that other elements have the same type
        for element in input:
            if type(element) != first_type:
                raise TypeError("elements of input must be of the same type")
        return first_type


def qua_declare(type_):
    """
    performs QUA declare() statement with the correct type
    :param type_:type: a type object. int, float or bool, or the string 'stream'  for q qua stream veriable
    :return: a QUA variable with the appropriate type
    """

    if type_ == int:
        return declare(int)
    elif type_ == float:
        return declare(fixed)
    elif type_ == bool:
        return declare(bool)
    elif type_ =='stream':
        return declare_stream()
    else:
        raise Exception("qua supports only int, float,  bool, or 'stream'")

@dataclass()
class QUAParameter(em.Parameter):
    """
    a class for dealing with experimental parameters that need to be represented as a QUA variable
    attributes:
    name, value, units, is_iterated, qua_type : see ctor doc-string (run QUAParameter?)

    qua_var: the QUA-variable representing the parameter. initialized to None, but can be declared and assigned
    within a QUA program. for example using the methods QUAParameter.declare() or QUAConfig.get_qua_vars()
    """
    name:str
    value:typing.Any
    units:str
    is_iterated:bool
    qua_type: type = None

    def __init__(self, name:str, value, units=None, is_iterated=None, qua_type=None):
        """
      creates a QUAParameter object
      :param name: str -  name of the parameter

      :param value: bool, int, float, or an iterable of one of them, or None (typically for a stream variable)

      :param units: str  - physical units of the parameter. !IMPORTANT WARNING!: this is a cosmetics thing for display
      in Labber only. the quantities themselves are always passed to OPX as given and treated according to
      the QUA default conventions. TODO - something smarter than that, maybe with the qualang_tools.units module

      :param is_iterated: bool. determines whether the value is constant or iterated. if iterated, then value
      should be an Iterable. by default self.is_iterated is defined according to whether value is an
      Iterable, but this can be changed if you want, for example, a constant value that is a list.

      :param qua_type:type or str:  int, float, bools or the string "stream". determines the type of the
      qua_variable. if not supplied by user, initialized automatically according to the type of value. so practically
      you'd only want to supply it manually in the case of a stream variable.

        """
        # run parent ctor:
        super().__init__(name, value, units, is_iterated)

        # initialize qua_var attribute to None
        self.qua_var = None

        #initiallize qua_type attribute
        if qua_type is None:
            self.qua_type = iter_type(self.value)
        else:
            self.qua_type = qua_type

    def declare(self):
        """
        to be used within a QUA program. performs QUA's declare() with the correct type (given by self.qua_type) and
        stores the resulting QUA variable in self.qua_var
        """
        self.qua_var = qua_declare(self.qua_type)



class QUAConfig(em.Config):
    """
    child class of experiment_manager.Config. used to store and manipulate a list of QUAParameter objects. (or a
    mixed list with some regulare Parameters and some QUAPArameters)
    """
    def get_qua_params(self):
        """
        returns a QUAConfig object with only the QUAParameters in the original QUAConfig. IMPORTANT: note that
        Parameters and QUAParameters are passed by reference! so the new QUAConfig still points to the original
        Parameters. use python's deepcopy() if you want a new copy.

        :return: QUAConfig
        """
        new_list = []
        for param in self.param_list:
            if isinstance(param, QUAParameter):
                new_list.append(param)
        return QUAConfig(*new_list)

    def get_normal_params(self):
        """
        returns an experiment_manager.Config object with only the Parameters in the original QUAConfig that are not
        QUAParameters. IMPORTANT: note that Parameters and QUAParameters are passed by reference! so the new
        Config still points to the original Parameters. use python's deepcopy() if you want a new copy.

        :return: experiment_manager.Config
        """
        new_list = []
        for param in self.param_list:
            if not isinstance(param, QUAParameter):
                new_list.append(param)
        return em.Config(*new_list)

    def get_qua_vars(self, assign_flag=True, output_format = None):
        """
        to be used within a QUA program.
        declares (and possibly assigns) all qua variables stored in the QUAConfig object.
        type for declaration and values for assignment are taken from the attributes of each QUAParameter object
        the QUA variables are stored in the qua_var attribute of each QUAParameter object.

        :param assign_flag:bool: decide whether also to assign or only to declare, but in any case iterated
        Parameters are never assigned.

        :param output_format: str: 'list' or 'dict' or None. decide which format to output, if at all.

        :return: if output_format==None: no return. if output_format=list: returns a list of QUA variables. if
        output_format=='dict' returns a dict with keys that are the QUAParameters names and values that are the
        corresponding QUA variables
        """
        get_list = output_format =='list'
        get_dict = output_format == 'dict'
        if get_list:
            qua_vars = []
        if get_dict:
            qua_vars = dict()

        for param in self.param_list:
            if isinstance(param,QUAParameter):
                if param.qua_var is None:
                    param.declare()
                if (not param.is_iterated) and assign_flag:
                    assign(param.qua_var, param.value)
                if get_list:
                    qua_vars.append(param.qua_var)
                if get_dict:
                    qua_vars[param.name] = param.qua_var
        if get_list or get_dict:
            return qua_vars


def qua_run(program, simulate=True):
    config = get_qua_config_dict()
    if simulate:
        qmManager = QuantumMachinesManager()
        job = qmManager.simulate(config, program, SimulationConfig(2500))
    return job

def qua_get_result(job, plot_samples=False):
    if plot_samples:
        job.get_simulated_samples().con1.plot()
    return job.result_handles


class QUAExperiment:
    """
    an abstract class (to be used only as a parent for user defined classes) for dealing with an experiment that uses
    QUA.  for each new experiment the user has to implement the single_run method which defines a single execution
    pulse sequence. the QUAExperiment class methods then deal with performing this sequence in loops within a QUA program.

    """

    def __init__(self):
        self.stream = None
        self.output_temp = None


    def single_run(self, config:QUAConfig):
        raise NotImplemented()

    def initialize_streams(self, stream_config:QUAConfig):
        """
        initialize the attributes self.stream and self.output_temp
        :param stream_config:QUAConfig that contains QUAParameters that define what quantities we want to stream
        for example: stream_config = QUAConfig(QUAParameter('I', None, 'a.u', qua_type = float),
                                               QUAParameter('Q', None, 'a.u', qua_type = float))
        """
        self.output_temp = deepcopy(stream_config)
        self.stream = deepcopy(stream_config)
        for param in self.stream.param_list:
            param.qua_type = 'stream'

    def nd_loop(self, loop_config:QUAConfig):
        """
        perform the experiment in an N-dimensional loop

        :param loop_config:QUAConfig containing the experiment parameters. iterated QUAParameters will be looped on.
        it also has to include a regular Parameter with name 'repetitions' amd integer value to indicate averaging repetitions
        #TODO input verification

        :return: a QUA program
        """

        # get Config object with only iterated QUAParameters (=variables): (filter out normal Parameters and
        # un-iterated QUAParameters)
        variables_config =QUAConfig(*loop_config.get_iterables()).get_qua_params()

        # unwrap looped values to get the format that QUA's for_each_() can handle (tuple of tuples, where the nth
        # inner tuple contains all the values that the nth parameter takes throughout the entire N-dimensional loop)
        iteration_tuple = tuple(zip(*tuple(itertools.product(*variables_config.get_values()))))

        # QUA program:
        with program() as prog:
            self.output_temp.get_qua_vars(assign_flag=False)
            self.stream.get_qua_vars(assign_flag=False)
            loop_config.get_qua_vars()
            # QUA loops:
            # external averaging loop:
            rep = declare(int)
            with for_(rep,0,rep<loop_config.repetitions.value, rep + 1):
                # N-dimensional QUA loop:
                with for_each_(tuple([param.qua_var for param in variables_config.param_list]), iteration_tuple):
                    self.single_run(loop_config)

                # process stream
                loop_lengths = []
                for param in variables_config.param_list:
                    loop_lengths.append(len(param.value))

            with stream_processing():
                for param in self.stream.param_list:
                    param.qua_var.buffer(*loop_lengths).average().save(param.name) # TODO make sure that names are the same. maybe say this in doc

        return prog

    def for_each(self, config:QUAConfig):
        """
        performs a looped experiment with QUA's for_each_ conventions: a 1D loop where you give a 1D set of values to
        each parameter and the loops pairs them element-wise. probably this is less useful than nd_loop (see above)
        and in particular I am not sure how to integrate this with Labber

        :param config:QUAConfig with one or more iterated QUAParameters that all have the same length. (un-iterated
        Parameters and QUAParameters are also allowed)

        :return: a QUA program

        """

        # get Config object with only iterated QUA parameters (=variables): (filter out normal Parameters and
        # un-iterated QUA parameters)
        variables_config =QUAConfig(*config.get_iterables()).get_qua_params()

        # QUA program:
        with program() as prog:
            self.output_temp.get_qua_vars(assign_flag=False)
            self.stream.get_qua_vars(assign_flag=False)
            config.get_qua_vars()
            # QUA loop:
            with for_each_(tuple([param.qua_var for param in variables_config.param_list]), tuple([param.value for param in variables_config.param_list])):
                self.single_run(config)

            # process stream
            loop_length = len(variables_config.param_list[0].value)  # from first parameter since they are all the same
            with stream_processing():
                for param in self.stream.param_list:
                    param.qua_var.buffer(loop_length).average().save(param.name) # TODO make sure that names are the same. maybe say this in doc

        return prog

    # def labber_read(self,result,config,labber_log_name=None):
    #     # get labber logfile parameters:
    #     step_list = QUAConfig(*config.get_qua_vars.get_iterables()).get_labber_step_list()
    #     log_list = self.stream.get_labber_log_list()
    #     # choose name
    #     if labber_log_name:
    #         log_name = labber_log_name
    #     else:
    #         log_name = self.experiment_name
    #     log_name = lu.get_log_name(log_name) # this generates automatic numbering and prevents overwrite
    #
    #     #create labber logfile
    #     logfile = Labber.createLogFile_ForData(log_name, log_list, step_list)
    #
    #     #add data:
    #
    #     for param in self.stream.param_list:
    #         getattr(results,param.name).fetch_all()

# -------------------------------------------------------------------------------------------------------------------- #

# an example of usage:

# define an experiment: this example experiment is a rabi sequence but it also streams the parameters amplitude and
# duration such that we will be able to see that the loops work:
class RabiExperiment(QUAExperiment):
    def __init__(self):
        # define stream variable I with units a.u. and type float,
        #        stream variable amplitude with no units and type float
        #        stream varialbe duration with units 'clock-cycle' and type int:
        stream_config = QUAConfig(QUAParameter('I', None, 'a.u.', qua_type = float),
                                  QUAParameter('amplitude',None, qua_type = float),
                                  QUAParameter('duration',None,'clock-cycles' ,qua_type=int))

        #initialize stream variables:
        self.initialize_streams(stream_config)

    def single_run(self, config:QUAConfig):
        # implement the pulse sequence:
        play_pulse(f'X_{config.qubit_idx.value}', f'drive{config.qubit_idx.value}', scale_amplitude=config.scale_amplitude.qua_var,
                   duration=config.duration.qua_var)
        align(f'drive{config.qubit_idx.value}', f'readout{config.qubit_idx.value}')
        measure("readout", f'readout{config.qubit_idx.value}', None,
                ("simple_cos", "out_I", self.output_temp.I.qua_var))

        # save data: in a real experiment we will only save I (maybe Q) but here we also want to stream the amplitude
        # and duration to see in the simulator that the loops work the way we want:
        save(self.output_temp.I.qua_var, self.stream.I.qua_var)
        save(config.scale_amplitude.qua_var, self.stream.amplitude.qua_var)
        save(config.duration.qua_var, self.stream.duration.qua_var)

        if wait_time>0:
            wait(wait_time)


# test the Rabi experiment:
#define an instance of the new class:
rabi = RabiExperiment()

# define the experiment configurations for 2D loop: we have 2 QUA parameters that we are going to loop on,
# and two normal parameters that is constant (#TODO looping on regular parameters (not in real-time) is not implemented yet)
rabi_config = QUAConfig(em.Parameter("repetitions",2),
                        QUAParameter("duration",[200//4, 400//4, 800//4], units='clock_cycles (4ns)'),
                        QUAParameter("scale_amplitude", [0.1,0.5,1.0, 2.5]),
                        em.Parameter('qubit_idx',2))

# use method nd_loops to get the QUA program
prog = rabi.nd_loop(rabi_config)

# get qua configurations dict from device parameters (external file):
config = get_qua_config_dict()

# simulate the program
qmManager = QuantumMachinesManager()
job = qmManager.simulate(config,prog, SimulationConfig(14000))

# get samples
samples = job.get_simulated_samples()

#get "measuement" results
results = job.result_handles

print('I:')
print(results.I.fetch_all()) # here we get zeros because this is just a simulation

print('amplitude:')
print(results.amplitude.fetch_all())

print('duration:')
print(results.duration.fetch_all())

#plot samples
samples.con1.plot()
