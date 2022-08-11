#TODO imports
#TODO: support for np.arrays in experiment config
# TODO get params file explicitely
#TODO frequency and amplitude: more natural user interface
#TODO time estimation
# TODO support numpy


# manange_qua
import itertools
import sys
import typing
from copy import deepcopy

import numpy as np

sys.path.append("..")
sys.path.append(r"\\132.64.80.214\overlordcommon\Users\Guy\PHD\repos\experiment-manager")
import experiment_manager
from general_utils import enumerated_product
import labber_util as lu

#sys.path.append(r"\\132.64.80.214\overlordcommon\Users\Guy\PHD\repos\manage_qua\files copied from Naftali")
#import device_parameters_2Q
# Nafteli's imports:
#import two_qubit_config_gates
import importlib

#importlib.reload(two_qubit_config_gates)
importlib.reload(experiment_manager)

import experiment_manager as em
from two_qubit_config_gates import *
import Labber
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from qm.qua import *
#import OPX.config_generator as config_generator
from matplotlib import pyplot as plt
#from os.path import join
#import os
#import qdac as qdac_lib

plt.ion()

from dataclasses import dataclass




# -------------general helper functions------------------

def bool_list_2_int(lst):
    """
    get an int from a list of bools
    :param lst:List[bool] a list of booleans
    :return: an int that is concatenating the bools of the list into a binary number, and then converting to int.
    """
    return int(''.join(['1' if x else '0' for x in lst]), 2)

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


# -------------QUA helper functions------------------
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


def qua_declare(type_):
    """
    performs QUA declare() statement with the correct type
    :param type_:type: a type object. int or , float or bool, or the string 'stream'  for q qua stream variable
    :return: a QUA variable with the appropriate type (int, fixed, bool. or stream)
    """

    if issubclass(type_,int) or isinstance(type_,np.integer):
        return declare(int)
    elif issubclass(type_,float) or isinstance(type_,np.floating):
        return declare(fixed)
    elif issubclass(type, bool) or issubclass(type, np.bool_):
        return declare(bool)
    elif type_ == 'stream':
        return declare_stream()
    else:
        raise Exception("qua supports only int, float,  bool, or 'stream'. ")


#---------------------classes------------------------#
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

      :param units: str  - physical units of the parameter. !IMPORTANT WARNING!: this is a cosmetics thing for
      displaying in Labber only. the quantities themselves are always passed to OPX as given and treated according to
      the QUA default conventions. TODO - something smarter than that, maybe with the qualang_tools.units module

      :param is_iterated: bool. determines whether the value is constant or iterated. if iterated, then value
      should be an Iterable. by default self.is_iterated is defined according to whether value is an
      Iterable, but this can be changed if you want, for example, a constant value that is a list.

      :param qua_type:type or str:  int, float, bools or the string "stream". determines the type of the
      qua_variable. if not supplied by user, initialized automatically according to the type of value. so practically
      you will probably only want to supply it manually in the case of a stream variable.

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
    mixed list with some regular Parameters and some QUAParameters)
    """
    def get_qua_params(self):
        """
        returns a QUAConfig object with only the QUAParameters in the original QUAConfig, filtering out the regular
        Parameters. IMPORTANT: note that Parameters and QUAParameters are passed by reference, so the new QUAConfig
        still points to the original Parameters. use python's deepcopy() if you want a new copy.

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
        QUAParameters. IMPORTANT: note that Parameters and QUAParameters are passed by reference, so the new
        Config still points to the original Parameters. use python's deepcopy() if you want a new copy.

        :return:experiment_manager.Config
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
        type for declaration and values for assignment are taken from the attributes of each QUAParameter.
        the QUA variables are stored in the qua_var attribute of each QUAParameter.

        :param assign_flag:bool: decide whether also to assign or only to declare, but in any case iterated
        Parameters are never assigned.

        :param output_format: str: 'list' or 'dict' or None. decide which format to output, if at all.

        :return: if output_format==None: no return if output_format=='list': returns a list of QUA variables. if
        output_format=='dict' returns a dict with keys that are the QUAParameters names and values that are the
        corresponding QUA variables
        """
        get_list = output_format == 'list'
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
    QUA.  for each new experiment the user has to implement the single_run() method which defines a single execution
    pulse sequence. and an __init__() method that initializes certain attributes like in the example below. the
    QUAExperiment class methods then deal with performing this sequence in loops within a QUA program.

    example of a child class definition:

    class RabiExperiment(QUAExperiment):
        def __init__(self):
            # define stream variable I with units a.u. and type float (you can have more than one)
            stream_config = QUAConfig(QUAParameter('I', None, 'a.u.', qua_type = float))

            #initialize stream variables:
            self.initialize_streams(stream_config)

            # initialize Labber stuff:
            self.experiment_name ="rabi_test_qua_manage"
            self.labber_tags = ["rabi", "test", "simulation"]


        def single_run(self, config:QUAConfig):
            # implement the pulse sequence with parameters from config:
            play_pulse('X_1', 'drive1', scale_amplitude=config.scale_amplitude.qua_var,
                       duration=config.duration.qua_var)
            align('drive1', 'readout1')

            # measure and store in self.output_temp.I.qua_var:
            measure("readout", 'readout1', None,
                    ("simple_cos", "out_I", self.output_temp.I.qua_var))

            # save data to stream variable:
            save(self.output_temp.I.qua_var, self.stream.I.qua_var)

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

        :param loop_config:QUAConfig containing the experiment parameters that are used by self.single_run().
        iterated QUAParameters will be looped on. loop_config also has to include a regular Parameter with name
        'repetitions' and an integer value to indicate averaging repetitions #TODO input verification

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
                    # param.qua_var.buffer(400,100).save_all(param.name)  # TODO make sure that names are the same. maybe say this in doc

        return prog

    def for_each(self, config:QUAConfig):
        """
        performs a looped experiment with QUA's for_each_ conventions: a 1D loop where you give a 1D set of values to
        each parameter and the loop pairs them element-wise. probably this is less useful than nd_loop (see above)
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

    def read_loop_job_into_labber(self, job, loop_config:QUAConfig, labber_log_name=None): #TODO better method name?
        """
        read results of an N-dimensional loop and saves to Labber.

        :param job: a QUA job object that is associated with the program generated by self.nd_loops(loop_config)

        :param loop_config:QUAConfig that was used while running the loop

        :param labber_log_name:str: optional. if None - use automatic naming scheme. WARNING:if you don't using the
        automatic naming be careful not to overwrite an existing log file.

        :return: Labber.logfile object
        """
        # get labber logfile parameters:
        step_list = QUAConfig(*loop_config.get_iterables()).get_labber_step_list()
        log_list = self.stream.get_labber_log_list()
        # choose name
        if labber_log_name:
            log_name = labber_log_name
        else:
            log_name = self.experiment_name
            log_name = log_name + "_loop_on"
            for d in step_list:
                log_name=log_name + "_" + d["name"]

        log_name = lu.get_log_name(log_name) # this generates automatic numbering and prevents overwrite

        #create labber logfile
        logfile = Labber.createLogFile_ForData(log_name, log_list, step_list)

        # get data:
        result = job.result_handles
        result.wait_for_all_values()

        variables = loop_config.get_iterables()
        outer_variables_config = em.Config(*variables[:-1])

        for indices, vals in enumerated_product(*outer_variables_config.get_values()): # loop on all but the innermost loop
            labber_trace = dict()
            for param in self.stream.param_list:
                labber_trace[param.name] = getattr(result, param.name).fetch_all()[indices]

            #add data to logfile
            logfile.addEntry(labber_trace)

        # set comment, tags etc:
        # experiment parameters:
        comment_str = "experiment parameters:\n"
        comment_str = comment_str + str(loop_config.get_metadata_table())
        comment_str = comment_str +'\n\n'
        #device parameters:
        comment_str = comment_str +"device parameters:\n"

        for key in device_parameters_2Q.device_parameters.keys(): #TODO later device_parameter will be an argument that all the functions need to recieve
            comment_str = comment_str + key + "  :  " + str(device_parameters_2Q.device_parameters[key])  + "\n"

        logfile.setComment(comment_str)
        logfile.setTags(self.labber_tags)

        return logfile



# -------------------------------------------------------------------------------------------------------------------- #
