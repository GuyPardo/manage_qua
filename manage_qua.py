#manange_qua
import sys
sys.path.append("..")
sys.path.append(r"\\132.64.80.214\overlordcommon\Users\Guy\PHD\repos\experiment-manager")
import experiment_manager as em
sys.path.append(r"\\132.64.80.214\overlordcommon\Users\Guy\PHD\repos\manage_qua\files copied from Naftali")
#Nafteli's imports:
import two_qubit_config_gates
import importlib
importlib.reload(two_qubit_config_gates)

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
    return int(''.join(['1' if x else '0' for x in lst]),2)



def play_pulse(pulse, element, scale_amplitude=None, frequency=None, duration=None):
    """
    perform QUA's play(pulse, element) of pulse to element with optional dynamical change of amplitude, frequency or
    duration. If one (or more) of the three optional arguments is not supplied, default value from the QUA config
    dict will be used when the program is run

    :param pulse:str  - the name of the pulse to play as appears in config
    :param element:str  -  the name of the element for which to play the pulse as appears in config
    :param scale_amplitude:QUA-float or None  - dimensionless factor by which to scale the amplitude for pulse
    :param frequency: QUA-int or None -  new (IF) frequency for pulse in Hz
    :param duration: QUA-int or None - new duration for pulse in clock cycles = 4ns
    :return:
    """

    # the play statement accepts a None value for duration so we don't have to worry about ampliutude and frequency:

    indicator = bool_list_2_int([bool(scale_amplitude), bool(frequency)])

    if indicator == 0: # = [0,0] change nothing (only possibly duration )
        print("i am in case 0")
        play(pulse, element, duration=duration)
    elif indicator == 1:  # = [0,1]:  change only frequency (and possibly duration)
        print("i am in case 1")
        update_frequency(element, frequency)
        play(pulse, element, duration)
    elif indicator == 2:  # = [1,0]: change only scale_amplitude (and possibly duration)
        print("i am in case 2")
        play(pulse * amp(scale_amplitude), element, duration=duration)
    elif indicator == 3:  # = [1,1]: change amplitude and frequency (and possibly duration)
        print("i am in case 3")
        update_frequency(element, frequency)
        play(pulse * amp(scale_amplitude), element, duration=duration)





# get config dict from param file
cg = init_config_generator()
create_readout_elements(cg)
create_drive_elements(cg)
create_pulses(cg)
add_OPX_dc_elements(cg)
config = cg.get_config()



t_vec = [10, 30, 100]
a_vec = [0.25, 1.0, 0.5]
f_vec = [int(30e6),int(40e6), int(50e6)] #
# t_mesh, a_mesh = np.meshgrid(t_vec, a_vec)

with program() as prog:
    I=declare(fixed)

    f = declare(int)
    t = declare(int)
    a = declare(fixed)

    with for_each_((t, a, f), (t_vec, a_vec,f_vec)):
        play_pulse('X_1' , 'drive1',scale_amplitude=a)
        # align("drive1", "readout1")
        # measure("readout", "readout1",None,("simple_cos", "out_I", I))


#### main #####

# simulate
qmManager = QuantumMachinesManager()
job = qmManager.simulate(config,prog, SimulationConfig(2500))
samples = job.get_simulated_samples()

#plot
samples.con1.plot()


