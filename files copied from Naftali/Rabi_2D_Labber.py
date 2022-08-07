#Send constant pulse and read demodulated data
#Constant readout and XY frequency, scan drive strength and length
#Written by Naftali 3/22

import device_parameters
import Labber
import importlib
importlib.reload(device_parameters)
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from qm.qua import *
import OPX.config_generator as config_generator
import numpy as np
from matplotlib import pyplot as plt
from os.path import join
import os
plt.ion()

params = device_parameters.device_parameters

def set_zero(qdac_lib, qdac, QDAC_channel):
    # set DC to 0 to prevent hysteresis etc.
    qdac_lib.set_and_show_QDAC(qdac, QDAC_channel, 0.0)

save_to_Labber = True
labber_file_name = "Rabi_2D_2Q_qubit1"

labber_comment = "Rabi_2D_Labber.py. IQM 2 qubit chip."
tags = ["IQM"]
project_name = "IQM"
user_name = "Naftali Kirsh"

#-----------Parameters-------------

debug = False #set True to measure a constant frequency pulse
simulate = False

repetitions =1000

#sweep parameters
drive_amps_min = 0
drive_amps_max = 0.2
drive_amps_steps = 200
drive_amps = np.linspace(drive_amps_min,drive_amps_max,num=drive_amps_steps,endpoint=True)
drive_amp0 = 0.1

drive_len_min = 16
drive_len_delta = 32
drive_len_steps = 200
drive_len_max = drive_len_min+(drive_len_steps-1)*drive_len_delta
drive_lens = np.linspace(drive_len_min,drive_len_max,num=drive_len_steps,endpoint=True)

if save_to_Labber:
    lStep = [dict(name="Drive amplitude",unit="V",values=drive_amps),dict(name="Drive length",unit="ns",values=drive_lens)]
    lLog = [dict(name="Amplitude", unit="AU", vector=False), dict(name="Unwrapped Phase", unit="radian", vector=False)]
    f = Labber.createLogFile_ForData(labber_file_name, lLog, lStep)

#Local oscillator
mg_address_readout = params["mg_address_readout"]
mg_address_drive = params["mg_address_drive"]
MG_ro = params["MG_ro"]
MG_drive = params["MG_drive"]
f_r = params["f_r"] #Hz
lo_freq_readout = params["lo_freq_readout"] #Hz
if_freq_readout = lo_freq_readout-f_r #Hz, SBM frequency is lo_freq-if_freq
lo_power_readout = params["lo_power_readout"] #dBm
f_d = params["f_d"]#Hz
lo_freq_drive = params["lo_freq_drive"] #Hz
if_freq_drive = lo_freq_drive-f_d #Hz, SBM frequency is lo_freq-if_freq
lo_power_drive = params["lo_power_drive"] #dBm

#DC bias
use_QDAC = True
QDAC_port = params["QDAC_port"]
QDAC_channel = params["QDAC_channel"]
dc_bias = params["dc_bias"]



#OPX

wait_time = params["wait_time"]

#input=readout channels
I_input_channel = params["I_input_channel"]
Q_input_channel = params["Q_input_channel"]
I_input_offset = params["I_input_offset"]
Q_input_offset = params["Q_input_offset"]
#output channels
I_channel_ro = params["I_channel_ro"]
Q_channel_ro = params["Q_channel_ro"]
I_output_offset_ro = params["I_output_offset_ro"]
Q_output_offset_ro = params["Q_output_offset_ro"]
I_channel_drive = params["I_channel_drive"]
Q_channel_drive = params["Q_channel_drive"]
I_output_offset_drive = params["I_output_offset_drive"]
Q_output_offset_drive = params["Q_output_offset_drive"]
I_channel_ro_monitor = params["I_channel_ro_monitor"]
Q_channel_ro_monitor = params["Q_channel_ro_monitor"]
I_channel_drive_monitor = params["I_channel_drive_monitor"]
Q_channel_drive_monitor = params["Q_channel_drive_monitor"]

#Pulse - drive
drive_amps_delta = (drive_amps_max-drive_amps_min)/(drive_amps_steps-1)

#Pulse - readout
pulse_length_ro = params["pulse_length_ro"]  # ns
ampl_ro = params["ampl_ro"]


running_time = repetitions*1e-9*len(drive_amps)*len(drive_lens)*(4*wait_time+pulse_length_ro+np.mean(drive_lens))#TODO
print("Estimated running time is %g minutes. Press Ctrl-C to stop." % ((running_time) / 60.0))


# Readout
trigger_delay = 0
trigger_length = 10
time_of_flight = params["time_of_flight"]  # ns. must be at least 28

# OPX config
cg = config_generator.ConfigGenerator(
    output_offsets={I_channel_ro: I_output_offset_ro, Q_channel_ro: Q_output_offset_ro,
                    I_channel_drive: I_output_offset_drive, Q_channel_drive: Q_output_offset_drive,
                    I_channel_ro_monitor: I_output_offset_ro,
                    Q_channel_ro_monitor: Q_output_offset_ro,
                    I_channel_drive_monitor: I_output_offset_drive,
                    Q_channel_drive_monitor: Q_output_offset_drive
                    },
    input_offsets={I_input_channel: I_input_offset, Q_input_channel: Q_input_offset})
cg.add_mixer("mixer_ro", {(lo_freq_readout, if_freq_readout): [1.0, 0.0, 0.0, 1.0]})
cg.add_mixer("mixer_drive", {(lo_freq_drive, if_freq_drive): [1.0, 0.0, 0.0, 1.0]})
cg.add_mixed_readout_element("readout", lo_freq_readout + if_freq_readout, lo_freq_readout, I_channel_ro, Q_channel_ro,
                             {"out_I": I_input_channel, "out_Q": Q_input_channel}, "mixer_ro", time_of_flight)
cg.add_mixed_input_element("readout_mon", lo_freq_readout + if_freq_readout, lo_freq_readout, I_channel_ro_monitor,
                           Q_channel_ro_monitor, "mixer_ro")
cg.add_mixed_input_element("drive", lo_freq_drive + if_freq_drive, lo_freq_drive, I_channel_drive, Q_channel_drive,
                           "mixer_drive")
cg.add_mixed_input_element("drive_mon", lo_freq_drive + if_freq_drive, lo_freq_drive, I_channel_drive_monitor,
                           Q_channel_drive_monitor, "mixer_drive")

# Output / readout
cg.add_constant_waveform("const_ro", ampl_ro)
# cg.add_arbitrary_waveform("arb_ro", arb_ro_samples)
cg.add_constant_waveform("const_drive", drive_amp0)
cg.add_constant_waveform("const_zero", 0.0)
cg.add_integration_weight("simple_cos", [1.0] * (pulse_length_ro // 4), [0.0] * (pulse_length_ro // 4))
cg.add_integration_weight("simple_sin", [0.0] * (pulse_length_ro // 4), [1.0] * (pulse_length_ro // 4))
cg.add_mixed_measurement_pulse("const_readout", pulse_length_ro, ["const_ro", "const_ro"],
                               {"simple_cos": "simple_cos", "simple_sin": "simple_sin"},
                               cg.TriggerType.RISING_TRIGGER, trigger_delay, trigger_length)
cg.add_operation("readout", "readout", "const_readout")
# cg.add_mixed_measurement_pulse("arb_readout", total_length_ro, ["arb_ro", "arb_ro"],
#                                {"simple_cos": "simple_cos", "simple_sin": "simple_sin"},
#                                cg.TriggerType.RISING_TRIGGER, trigger_delay, trigger_length)
# cg.add_operation("readout", "readout", "arb_readout")
cg.add_mixed_control_pulse("const_drive", 16, ["const_drive", "const_drive"])
cg.add_mixed_control_pulse("const_zero", 16, ["const_zero", "const_zero"])
cg.add_operation("drive", "drive", "const_drive")
cg.add_operation("drive", "drive_zero", "const_zero")
cg.add_operation("drive_mon", "drive", "const_drive")
cg.add_operation("drive_mon", "drive_zero", "const_zero")
cg.add_mixed_control_pulse("const_readout_mon", pulse_length_ro, ["const_ro", "const_ro"])
cg.add_operation("readout", "zero_readout", "const_zero")
cg.add_operation("readout_mon", "readout_mon", "const_readout_mon")
cg.add_operation("readout_mon", "zero_readout", "const_zero")



# OPX measurement program
with program() as prog:
    stream_II = declare_stream()
    stream_IQ = declare_stream()
    stream_QI = declare_stream()
    stream_QQ = declare_stream()
    amp_idx = declare(int)
    curr_amp = declare(fixed)
    curr_drive_len = declare(int)
    rep = declare(int)
    II = declare(fixed)
    IQ = declare(fixed)
    QI = declare(fixed)
    QQ = declare(fixed)

    with for_(rep, 0, rep < repetitions, rep + 1):
        with for_(curr_drive_len, drive_len_min//4, curr_drive_len<=(drive_len_max//4), curr_drive_len+(drive_len_delta//4)):
            with for_(amp_idx, 0, amp_idx<drive_amps_steps, amp_idx+1):
                curr_amp = (drive_amps_min+Cast.mul_fixed_by_int(drive_amps_delta,amp_idx))/drive_amp0
                play("drive"*amp(curr_amp), "drive",duration=curr_drive_len)
                align("readout","drive")
                measure("readout", "readout", None,
                        ("simple_cos", "out_I", II), ("simple_sin", "out_I", IQ),
                        ("simple_cos", "out_Q", QI), ("simple_sin", "out_Q", QQ))
                play("drive"*amp(curr_amp), "drive_mon",duration=curr_drive_len)
                align("readout_mon", "drive_mon")
                play("readout_mon", "readout_mon")
                save(II, stream_II)
                save(IQ, stream_IQ)
                save(QI, stream_QI)
                save(QQ, stream_QQ)

                if wait_time > 0:
                    wait(wait_time)

    with stream_processing():
        stream_II.buffer(drive_len_steps,drive_amps_steps).average().save("II")
        stream_IQ.buffer(drive_len_steps,drive_amps_steps).average().save("IQ")
        stream_QI.buffer(drive_len_steps,drive_amps_steps).average().save("QI")
        stream_QQ.buffer(drive_len_steps,drive_amps_steps).average().save("QQ")

# ----------------main program---------------
# run
# DC bias
if not simulate:
    if use_QDAC:
        import qdac as qdac_lib

        with qdac_lib.qdac(QDAC_port) as qdac:
            # Setup QDAC
            qdac.setVoltageRange(QDAC_channel, 10)
            qdac.setCurrentRange(QDAC_channel, 1e-4)
            qdac_lib.set_and_show_QDAC(qdac, QDAC_channel, dc_bias)
    # MG
    mg_ro = MG_ro(mg_address_readout)
    mg_ro.setup_MG(lo_freq_readout / 1e6, lo_power_readout)
    mg_drive = MG_drive(mg_address_drive)
    mg_drive.setup_MG(lo_freq_drive / 1e6, lo_power_drive)

qmManager = QuantumMachinesManager()
config = cg.get_config()
# add trigger - TODO: using config_generator
config["elements"]["readout_mon"]["digitalInputs"] = {
    "trigger1":
        {
            "port": ("con1", 1),
            "delay": 144,
            "buffer": 10
        }
}
config["pulses"]["const_readout_mon"]["digital_marker"] = "trigger"
config["digital_waveforms"]["trigger"] = {"samples": [(1, 0)]}

config["controllers"]["con1"]["digital_outputs"] = {}
config["controllers"]["con1"]["digital_outputs"][1] = {}

qm = qmManager.open_qm(config)
if simulate:
    job = qmManager.simulate(config, prog,
                      SimulationConfig(int(running_time*1e9*1.5) // 4, LoopbackInterface([("con1", 1, "con1", 1), ("con1", 2, "con1", 2)])))#, include_analog_waveforms=True))
    samples = job.get_simulated_samples()
    plt.figure()
    samples.con1.plot()
    # plt.subplot(211)
    # samples.con1.plot(analog_ports=["1","2"],digital_ports=["1"])
    # plt.subplot(212)
    # samples.con1.plot(analog_ports=["3", "4"])
    plt.show()
    result_handles = job.result_handles
    result_handles.wait_for_all_values()
else:
    pending_job = qm.queue.add_to_start(prog, duration_limit=0, data_limit=0)
    # get data
    job = pending_job.wait_for_execution()
    result_handles = job.result_handles
    result_handles.wait_for_all_values()
    print("Got results")
    mg_ro.set_on(False)
    mg_drive.set_on(False)
    if use_QDAC:
        with qdac_lib.qdac(QDAC_port) as qdac:
            qdac_lib.set_and_show_QDAC(qdac, QDAC_channel, 0.0)

# analyze
II = result_handles.II.fetch_all()
QI = result_handles.QI.fetch_all()
IQ = result_handles.IQ.fetch_all()
QQ = result_handles.QQ.fetch_all()
# I_m,Q_m = negative detuning from lo_freq by if_freq
I_m = II + QQ
Q_m = IQ - QI
s = I_m + 1j * Q_m
# I_p = II - QQ
# Q_p = -QI - IQ
# s = I_p+1j*Q_p
amp_all = np.abs(s)
phase_all = np.angle(s)


if save_to_Labber:
    phase = np.unwrap(phase_all)

    for len_idx in range(len(drive_lens)):
        data_add = {"Amplitude": np.abs(s[len_idx,:]), "Unwrapped Phase": phase[len_idx,:]}
        f.addEntry(data_add)

    labber_comment = labber_comment + str(params) + ("\n repetitions = %d" % repetitions)
    labber_comment = labber_comment + "\n" + (use_QDAC * "with ") + ((not use_QDAC) * "without ") + "DC bias."
    f.setProject(project_name)
    f.setComment(labber_comment)
    f.setTags(tags)
    f.setUser(user_name)


# plt.figure(9999)
# plt.plot(drive_amps, amp_all/pulse_length_ro, '*-', label=("ro pulse length=%g ns, readout ampl.=%g, drive len.=%g, repetitions=%d" % (pulse_length_ro,ampl_ro,pulse_length_drive,repetitions)))
# plt.legend()
# plt.xlabel("Drive amplitude [V]")
# plt.ylabel("Mean amplitude/(readout pulse length)")
# plt.figure(9998)
# plt.plot(drive_amps, np.unwrap(phase_all)/2/np.pi, '*-',label=("ro pulse length=%g ns, readout ampl.=%g, drive len.=%g, repetitions=%d" % (pulse_length_ro,ampl_ro,pulse_length_drive,repetitions)))
# plt.xlabel("Drive amplitue [V]")
# plt.ylabel("Mean phase/2\pi (unwrapped)")
# plt.legend()
# plt.figure(10000)
# plt.plot(I_m/pulse_length_ro, Q_m/pulse_length_ro, '*-')
# plt.xlabel("<I>/(readout pulse length)")
# plt.ylabel("<Q>/(readout pulse length)")

