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

    # the play statement accepts a None value for duration so we don't have to worry about ampliutude and frequency:

    indicator = bool_list_2_int([bool(scale_amplitude), bool(frequency)])

    if indicator == 0:  # = [0,0] change nothing (only possibly duration )
        #print("i am in case 0")
        play(pulse, element, duration=duration)
    elif indicator == 1:  # = [0,1]:  change only frequency (and possibly duration)
        #print("i am in case 1")
        update_frequency(element, frequency)
        play(pulse, element, duration)
    elif indicator == 2:  # = [1,0]: change only scale_amplitude (and possibly duration)
        #print("i am in case 2")
        play(pulse * amp(scale_amplitude), element, duration=duration)
    elif indicator == 3:  # = [1,1]: change amplitude and frequency (and possibly duration)
        #print("i am in case 3")
        update_frequency(element, frequency)
        play(pulse * amp(scale_amplitude), element, duration=duration)



def iter_type(input):
    """
    get the type of the elements of iterable object input. ot yjr type of input itself if it is not iterable
    returns error if input has elements with different types
    :param input: an iterable with elements of a single type ot a non iterable
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
    performs QUA declare() statement with the correct type ()
    :param type_:type: a type object. int, float or bool, or the string 'stream'
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

# maybe this is not neede
# class QUAParameter(em.Parameter):
#     def __init__(self, name:str, value, units = None, is_iterated = None):
#         super().__init__(name, value, units, is_iterated)

@dataclass()
class QUAParameter(em.Parameter):
    name:str
    value:typing.Any
    qua_type: type = None
    units:str
    is_iterated:bool

    def __init__(self, name:str, value, units=None, is_iterated=None, qua_type=None):
        super().__init__(name, value, units, is_iterated)
        self.qua_var = None

        if qua_type is None:
            self.qua_type = iter_type(self.value)
        else:
            self.qua_type = qua_type

    def declare(self):
        self.qua_var = qua_declare(self.qua_type)



class QUAConfig(em.Config):
    def get_qua_vars(self, assign_flag=True, get_list=False, get_dict = False):
        """
        to be used within a qua program
        TODO more doc.
        only one of get_dict or get_list can be tr
        :return:
        """
        if get_list:
            qua_vars = []
        if get_dict:
            qua_vars = dict()

        for param in self.param_list:
            # print(param.qua_type)

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

    # def get_qua_vars_dict(self,assign_flag=True):
    #     """
    #     to be used within a qua program
    #     TODO more doc.
    #     :return:
    #     """
    #     qua_vars = dict()
    #     for param in self.param_list:
    #         qua_var = qua_declare(param.type)
    #         if (not param.is_iterated) and assign_flag:
    #             assign(qua_var, param.value)
    #         qua_vars[param.name] = qua_var
    #     return qua_vars








class QUAExperiment:
    def single_run(self, **params):
        raise NotImplemented()

    def nd_loop(self):
        #TODO: implement using self.for_each()
        pass

    def for_each(self, config:QUAConfig, save_to_labber=True):
        #TODO add repetitions
        #TODO add streaming
        #TODO deal with Labber

        # get dictionary with iterated parameters (=variables):
        variables_config =QUAConfig(*config.get_iterables())
        variables_dict = variables_config.get_values_dict()
        # QUA program:
        with program() as prog:
            self.output_temp.get_qua_vars(assign_flag=False)
            self.stream.get_qua_vars(assign_flag=False)
            run_params_dict = config.get_qua_vars(get_dict=True)
            # QUA loop:
            # print(variables_dict)
            # print(run_params_dict)
            with for_each_(tuple([run_params_dict[param.name] for param in variables_config.param_list]), tuple([value for value in variables_dict.values()])):
                self.single_run(**run_params_dict)
        return prog


# example:
class RabiExperiment(QUAExperiment):
    # TODO add __init__ and initialize attributes such as experiment name and other labber tags
    def __init__(self):
        self.output_temp = QUAConfig(QUAParameter('I1', None, 'a.u.', qua_type = float), QUAParameter('I2',None,'a.u.',qua_type = float)) #TODO better name
        self.stream = QUAConfig(QUAParameter('I1', None, 'a.u.', qua_type = 'stream'), QUAParameter('I2',None,'a.u.',qua_type = 'stream'))
    def single_run(self, **rabi_params):
        play_pulse('X_1', 'drive1', scale_amplitude=rabi_params["scale_amplitude"], duration=rabi_params["duration"])
        align('drive1', 'readout1')
        measure("readout", "readout1" , None,
                ("simple_cos", "out_I", self.output_temp.I1.qua_var))

        save(self.output_temp.I1.qua_var, self.stream.I1.qua_var)

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
params = QUAConfig(QUAParameter("scale_amplitude", 1.0),QUAParameter("duration", [100, 200, 300]))


prog = rabi.for_each(params)

# ## test play_pulse
# t_vec = [10, 30, 100]
# a_vec = [0.25, 1.0, 0.5]
# f_vec = [int(30e6),int(40e6), int(50e6)] #
# # t_mesh, a_mesh = np.meshgrid(t_vec, a_vec)
#
# with program() as prog:
#     I=declare(fixed)
#
#     f = declare(int)
#     t = declare(int)
#     a = declare(fixed)
#
#     with for_each_((t, a, f), (t_vec, a_vec,f_vec)):
#         play_pulse('X_1' , 'drive1',scale_amplitude=a)
#         # align("drive1", "readout1")
#         # measure("readout", "readout1",None,("simple_cos", "out_I", I))


#### main #####

# simulate
qmManager = QuantumMachinesManager()
job = qmManager.simulate(config,prog, SimulationConfig(2500))
samples = job.get_simulated_samples()

#plot
samples.con1.plot()
