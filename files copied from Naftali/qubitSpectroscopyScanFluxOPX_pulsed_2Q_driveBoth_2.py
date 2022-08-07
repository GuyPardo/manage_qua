#Send constant pulse and read demodulated data
#Constant readout frequency. Scan XY frequency and DC bias using OPX
#use square pulsed DC bias for the OPX DC channels
#Measure both qubits. save to Labber
#Written by Naftali 5/22
#New in v2: amp_OPX_dc0 is 0.4 instead of 0.5 (in two_qubits_config), 0.5 caused bugs.

import two_qubit_config # added by Guy
import importlib # added by Guy
importlib.reload(two_qubit_config)# added by Guy

from two_qubit_config import *

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


labber_file_name = "flux_2Q_pulsed_Q2flux_afterWarmpup_test3"
labber_comment = "Script: qubitSpectroscopyScanFluxOPX_pulsed_2Q_driveBoth_2.py. Scan coupler bias.\n" \
                 " IQM 2 qubit chip. See system diagram in Wiki (http://quantum.wiki.huji.ac.il/index.php?title=Experimental_setup). \n"\
"-20 dB  attenuators at ports C1,C3"
tags = ["IQM"]
project_name = "IQM 2 qubit"
user_name = "Naftali Kirsh"#"Guy Pardo"#


save_to_Labber = True

cg = init_config_generator()
create_readout_elements(cg)
create_drive_elements(cg)
create_pulses(cg)
add_OPX_dc_elements(cg)
add_OPX_dc_pulses(cg)
config = cg.get_config()

#-----------Parameters-------------

DC_to_scan = 2 #1,2 : qubits, 3: coupler. Must be an OPX controlled DC channel
dc_channel_scan = dc_channels[DC_to_scan-1]

dc_no_coupling = dc_bias_coupler

if not not_use_QDAC[DC_to_scan-1]:
    print("The scanned DC channel must be OPX controlled!")
    exit()

repetitions =250
stabilization_time = 0 #for DC bias, ns, multiples of 4

#sweep parameters
bias_levels_start = -317.5e-3#15.4e-3#-35e-3*10+dc_offsets[DC_to_scan-1]#-4e-3*10 #*10 because of attenuation on the opx bias outputs
bias_levels_end =  -317.5e-3#+dc_offsets[DC_to_scan-1]#-2e-3*10
bias_levels_steps = 1#201
if bias_levels_steps==1:
    bias_levels_delta = 1e-3
else:
    bias_levels_delta = (bias_levels_end-bias_levels_start)/(bias_levels_steps-1)
bias_levels =np.linspace(bias_levels_start,bias_levels_end,bias_levels_steps)
span =50e6#0.5e6
num_points = 121#1201#750
deltas = np.linspace(-span/2, span/2, num_points)
if_freqs1 = if_freq_drive[0]-deltas
if_freqs2 = if_freq_drive[1]-deltas
if len(if_freqs1)>1:
    if_delta = if_freqs1[1]-if_freqs1[0]
else:
    if_delta = -100


dc_pulse_len = max(pulse_length_drive[0],pulse_length_drive[1])



if save_to_Labber:
    lStep = [dict(name='Drive delta', unit='Hz', values=deltas),dict(name="Flux bias",unit="V",values=bias_levels)]
    lLog = [dict(name="I1", unit="AU", vector=False), dict(name="I2", unit="AU", vector=False)]
    f = Labber.createLogFile_ForData(labber_file_name, lLog, lStep)
    labber_comment = labber_comment + str(params) + ("\n repetitions = %d, DC_to_scan = %d \n" % (repetitions,DC_to_scan))
    labber_comment += ("stabilization_time = %d ns" % stabilization_time)
    f.setProject(project_name)
    f.setComment(labber_comment)
    f.setTags(tags)
    f.setUser(user_name)




#OPX measurement program
with program() as prog:
    # stream_samples = declare_stream(adc_trace=True) #for ADC data
    stream_I1 = declare_stream()
    stream_I2 = declare_stream()
    if_step = declare(int)
    curr_if_freq1 = declare(int)
    curr_if_freq2 = declare(int)
    bias_idx = declare(int)
    bias_amp_factor = declare(fixed)
    rep = declare(int)
    I1 = declare(fixed)
    I2 = declare(fixed)

    if params["phase_rot_1"] > 0:
        frame_rotation(params["phase_rot_1"], "readout1")
    if params["phase_rot_2"] > 0:
        frame_rotation(params["phase_rot_2"], "readout2")

    with for_(rep, 0, rep < repetitions, rep + 1):
        with for_(bias_idx, 0, bias_idx<bias_levels_steps, bias_idx+1):
            assign(bias_amp_factor, (bias_levels_start+Cast.mul_fixed_by_int(bias_levels_delta,bias_idx))/amp_OPX_dc0)
            with for_(if_step, 0, if_step < num_points, if_step + 1):
                assign(curr_if_freq1, if_freqs1[0] + int(if_delta) * if_step)
                assign(curr_if_freq2, if_freqs2[0] + int(if_delta) * if_step)
                update_frequency("drive1", curr_if_freq1)
                update_frequency("drive2", curr_if_freq2)
                if stabilization_time>0:
                    wait(stabilization_time//4,"drive1")
                    wait(stabilization_time // 4, "drive2")
                play("const_dc" * amp(bias_amp_factor), "dc%d" % dc_channel_scan,
                     duration=int((dc_pulse_len + stabilization_time) // 4))
                play("drive1","drive1")
                play("drive2", "drive2")
                align("drive1","readout1")
                align("drive2", "readout2")
                measure("readout", "readout1", None,
                        ("simple_cos", "out_I", I1))
                measure("readout", "readout2", None,
                        ("simple_cos", "out_I", I2))


                save(I1, stream_I1)
                save(I2, stream_I2)

                if wait_time>0:
                    wait(wait_time)

    with stream_processing():
        stream_I1.buffer(bias_levels_steps,len(if_freqs1)).average().save("I1")
        stream_I2.buffer(bias_levels_steps,len(if_freqs2)).average().save("I2")




#----------------main program---------------
#run

#MG
mg_ro = [MG_ro(mg_address_readout[0]),MG_ro(mg_address_readout[1])]
mg_ro[0].setup_MG(lo_freq_readout[0]/1e6,lo_power_readout[0])
mg_ro[1].setup_MG(lo_freq_readout[1]/1e6,lo_power_readout[1])
mg_drive = MG_drive(mg_address_drive)
mg_drive.setup_MG(lo_freq_drive/1e6,lo_power_drive)



qmManager = QuantumMachinesManager()
qm = qmManager.open_qm(config)

run_time  = \
    repetitions*(max(pulse_length_ro)*1e-9+dc_pulse_len*1e-9 + stabilization_time*1e-9
                 + 4*wait_time*1e-9)*len(if_freqs1)*len(bias_levels)/60.0
print("Estimated running time is %g minutes. Press Ctrl-C to stop." % run_time)

with qdac_lib.qdac(QDAC_port) as qdac:

    if params[use_QDAC_params[DC_to_scan-1]]:
        dc_device = qdac
    else:
        dc_device = qm

    for c in range(1,3):
        if params[("use_QDAC_q%d" % c)]:
            qdac.setVoltageRange(DC_channel[c-1], 10)
            qdac.setCurrentRange(DC_channel[c-1], 1e-4)
            set_dc_voltage(qdac, DC_channel[c-1], dc_bias[c-1] , True)
        else:
            if not DC_to_scan==c:
                set_dc_voltage(qm, DC_channel[c-1], dc_bias[c-1], False)

    if params["use_QDAC_coupler"]:
        qdac.setVoltageRange(DC_channel_coupler, 10)
        qdac.setCurrentRange(DC_channel_coupler, 1e-4)
        set_dc_voltage(qdac, DC_channel_coupler, dc_bias_coupler, True)
    else:
        if not DC_to_scan == 3:
            set_dc_voltage(qm, DC_channel_coupler, dc_bias_coupler, False)



    pending_job = qm.queue.add_to_start(prog, duration_limit=0, data_limit=0)
    #get data
    job = pending_job.wait_for_execution()
    result_handles = job.result_handles
    result_handles.wait_for_all_values()
    print("Got results")

    #analyze
    I1 = result_handles.I1.fetch_all()
    I2 = result_handles.I2.fetch_all()


    if save_to_Labber:
        for bias_idx in range(bias_levels_steps):
            data_add = {"I1":I1[bias_idx,:],"I2":I2[bias_idx,:]}
            f.addEntry(data_add)

    mg_ro[0].set_on(False)
    mg_ro[1].set_on(False)
    mg_drive.set_on(False)
    for c in range(3):
        if params[use_QDAC_params[c]]:
            dc_device = qdac
        else:
            dc_device = qm
        set_dc_voltage(dc_device, dc_channels[c], dc_offsets_end[c], params[use_QDAC_params[c]])


