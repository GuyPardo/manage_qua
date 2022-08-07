#Time Rabi with 2 qubit setup
#Written by Naftali 6/22, uses gates pulses

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


save_to_Labber = True

labber_file_name = "2Q_qubit1_2Q_setup_26_7_4"

labber_comment = "Time_Rabi_Labber_2Q_setup.py. IQM 2 qubit chip. See system diagram in Wiki (http://quantum.wiki.huji.ac.il/index.php?title=Experimental_setup)." \
                 " -20 dB attenuators at qubit 2 and coupler flux lines."

tags = ["IQM"]
project_name = "IQM 2 qubit"
user_name = "Guy Pardo"#"Naftali Kirsh"#"Guy Pardo"#



#-----------Parameters-------------

qubit_to_drive = 1 #1-2

repetitions =250

#sweep parameters
drive_len_min = 16 #at least 16
drive_len_delta = 4 #multiples of 4
drive_len_steps = 200
drive_len_max = drive_len_min+(drive_len_steps-1)*drive_len_delta
drive_lens = np.linspace(drive_len_min,drive_len_max,num=drive_len_steps,endpoint=True)

#-------------

if save_to_Labber:
    lStep = [dict(name='Drive length', unit='ns', values=drive_lens)]
    lLog = [dict(name="I", unit="AU", vector=False)]
    f = Labber.createLogFile_ForData(labber_file_name, lLog, lStep)
    labber_comment = labber_comment + str(params) + ("\n repetitions = %d" % repetitions)
    labber_comment = labber_comment + "\n"
    f.setProject(project_name)
    f.setComment(labber_comment)
    f.setTags(tags)
    f.setUser(user_name)

running_time = repetitions*1e-9*len(drive_lens)*(4*wait_time+pulse_length_ro[qubit_to_drive-1]+pulse_length_drive[qubit_to_drive-1])
print("Estimated running time is %g minutes. Press Ctrl-C to stop." % ((running_time) / 60.0))


cg = init_config_generator()
create_readout_elements(cg)
create_drive_elements(cg)
create_pulses(cg)
add_OPX_dc_elements(cg)
config = cg.get_config()


# OPX measurement program
with program() as prog:
    stream_I = declare_stream()
    curr_drive_len = declare(int)
    rep = declare(int)
    I = declare(fixed)

    readout_rotation()

    with for_(rep, 0, rep < repetitions, rep + 1):
        with for_(curr_drive_len, drive_len_min//4, curr_drive_len<=(drive_len_max//4), curr_drive_len+(drive_len_delta//4)):
            play(("X_%d" % qubit_to_drive), "drive%d" % qubit_to_drive, duration=curr_drive_len)

            align("drive%d" % qubit_to_drive, "readout%d" % qubit_to_drive)
            measure("readout", "readout%d" % qubit_to_drive, None,
                    ("simple_cos", "out_I", I))

            save(I, stream_I)

            if wait_time > 0:
                wait(wait_time)

    with stream_processing():
        stream_I.buffer(drive_len_steps).average().save("I")


# ----------------main program---------------
# run
qmManager = QuantumMachinesManager()
qm = qmManager.open_qm(config)

setup_MGs()
setup_DC(qm)

pending_job = qm.queue.add_to_start(prog, duration_limit=0, data_limit=0)
#get data
job = pending_job.wait_for_execution()
result_handles = job.result_handles
result_handles.wait_for_all_values()
print("Got results")

MGs_off()
DC_off(qm)

I = result_handles.I.fetch_all()

if save_to_Labber:
    data_add = {"I": I}
    f.addEntry(data_add)


plt.figure(9999)
plt.plot(drive_lens, I, '*-')
plt.xlabel("Drive length [ns]")
plt.ylabel("I")

