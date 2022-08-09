# manange_qua
import sys
import typing

sys.path.append("..")
sys.path.append(r"\\132.64.80.214\overlordcommon\Users\Guy\PHD\repos\experiment-manager")
import experiment_manager

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


# @dataclass
# class MicrowaveGenerator:
#     """
#     a class for storing configuration data for microwave generators
#     """
#     mg_adress:str
#     mg_class


def bool_list_2_int(lst):
    """
    get an int from a list of bools
    :param lst:List[bool] a list of booleans
    :return: an int that is concatinating the bools of the list into a binary number, and then converting to int.
    """
    return int(''.join(['1' if x else '0' for x in lst]), 2)


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
    qua_type: type = None
    units:str
    is_iterated:bool

    def __init__(self, name:str, value, units=None, is_iterated=None, qua_type=None):
        """
      creates a QUAParameter object
      :param name: str -  name of the parameter

      :param value: bool, int, float, or an iterable of one of them, or None (typically for a stream variable)

      :param units: str  - physical units of the parameter. IMPORTANT!: this is a cosmetics thing for display
      in Labber only! the quantities themselves are always passed to OPX as given and treated according to
      the QUA defaults. TODO - something smarter than that, maybe with the qualang_tools.units module

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



class QUAExperiment:
    """
    an abstract class (to be used only as a parent for user defined classes) for dealing with an experiment that uses
    QUA.  for each new experiment the user has to implement the single_run method which defines a single execution
    pulse sequence. the QUAExperiment class methods then deal with performing this sequence in loops within a QUA program.

    """
    def single_run(self, config:QUAConfig):
        raise NotImplemented()

    def nd_loop(self):
        #TODO: implement using self.for_each()
        pass

    def for_each(self, config:QUAConfig, save_to_labber=True):
        """

        :param config:QUAConfig
        :param save_to_labber:
        :return:
        """
        #TODO add repetitions
        #TODO deal with Labber

        # get Config object with only iterated QUA parameters (=variables): (filter out normal Parameters and
        # un-iterated QUA parameters)
        variables_config =QUAConfig(*config.get_iterables()).get_qua_params()
        #variables_dict = variables_config.get_values_dict()
        print(variables_config.param_list)
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



# an example:
class RabiExperiment(QUAExperiment):
    def __init__(self):
        #TODO  initialize attributes such as experiment name and other labber tags or maybe this should be in Config?
        #TODO consider writing a default init. fucntion in the parent class for the output Configs
        self.output_temp = QUAConfig(QUAParameter('I', None, 'a.u.', qua_type = float)) #TODO better name
        self.stream = QUAConfig(QUAParameter('I', None, 'a.u.', qua_type = 'stream'))

    def single_run(self, config:QUAConfig):
        play_pulse(f'X_{config.qubit_idx.value}', f'drive{config.qubit_idx.value}', scale_amplitude=config.scale_amplitude.qua_var,
                   duration=config.duration.qua_var)
        align(f'drive{config.qubit_idx.value}', f'readout{config.qubit_idx.value}')
        measure("readout", f'readout{config.qubit_idx.value}', None,
                ("simple_cos", "out_I", self.output_temp.I.qua_var))

        save(self.output_temp.I.qua_var, self.stream.I.qua_var)

        if wait_time>0:
            wait(wait_time)



# get config dict from param file
cg = init_config_generator()
create_readout_elements(cg)
create_drive_elements(cg)
create_pulses(cg)
add_OPX_dc_elements(cg)
config = cg.get_config()

## test Rabi

rabi = RabiExperiment()
rabi_config = QUAConfig(QUAParameter("scale_amplitude", [0.1,0.5,1.0, 2.0]),
                        QUAParameter("duration", [200//4, 400//4, 800//4, 300//4], units='clock_cycles (4ns)'),
                        em.Parameter('qubit_idx',2))


prog = rabi.for_each(rabi_config)


#### main #####

# simulate
qmManager = QuantumMachinesManager()
job = qmManager.simulate(config,prog, SimulationConfig(2500))
samples = job.get_simulated_samples()
results = job.result_handles


#plot
samples.con1.plot()
