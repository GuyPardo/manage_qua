#Create a configuration for a two qubits, working in gate level, TODO: make it using OOP without global variables etc...
#Written by Naftali 6/22
#New in this version: use only I (or Q) for X (Y) rotations

import device_parameters_2Q
import importlib
importlib.reload(device_parameters_2Q)
import OPX.config_generator as config_generator
import qdac as qdac_lib
import numpy as np
from qm.qua import frame_rotation

Sx2_pulse_file = r"X:\Measurements\2022.dir\cooldown_05_2022\GinossarMax1\firstPulses\pulse2.csv"


params = device_parameters_2Q.device_parameters

def pad_zeros(arr):
    """Pad zeros to a 1d numpy array in order to make in a multiple of 4"""
    return np.append(arr,[0.0]*np.mod(arr.shape[0],4))

def set_dc_voltage(device, channel, volt, is_QDAC):
    """
    device is qdac object in case of QDAC, qm in case of OPX
    """
    if is_QDAC:
        qdac_lib.set_and_show_QDAC(device, channel, volt)
    else:
        device.set_output_dc_offset_by_element("dc%d" % channel, "single", volt)

#Local oscillator
mg_address_readout = [params["mg_address_readout1"],params["mg_address_readout2"]]
mg_address_drive = params["mg_address_drive"]
MG_ro = params["MG_ro"]
MG_drive = params["MG_drive"]
f_r = np.array([params["f_r1"],params["f_r2"]]) #Hz
lo_freq_readout = np.array([params["lo_freq_readout1"],params["lo_freq_readout2"]]) #Hz
if_freq_readout = lo_freq_readout - f_r #Hz, SBM frequency is lo_freq-if_freq
lo_power_readout = np.array([params["lo_power_readout1"],params["lo_power_readout2"]]) #dBm
f_d = np.array([params["f_d1"],params["f_d2"]])#Hz
f_d_12 = np.array([params["f_d1_12"],params["f_d2_12"]])#Hz
lo_freq_drive = params["lo_freq_drive"] #Hz
if_freq_drive = lo_freq_drive-f_d #Hz, SBM frequency is lo_freq-if_freq
if_freq_drive_12 = lo_freq_drive-f_d_12 #Hz, SBM frequency is lo_freq-if_freq
lo_power_drive = params["lo_power_drive"] #dBm

#DC bias
QDAC_port = params["QDAC_port"]
QDAC_channel = np.array([params["QDAC_channel_q1"],params["QDAC_channel_q2"]])
QDAC_channel_coupler = params["QDAC_channel_coupler"]
dc_bias = np.zeros((2,))
DC_channel = np.zeros((2,))
for c in range(2):
    if params[("use_QDAC_q%d" % (c+1))]:
        dc_bias[c] = params[("dc_bias%d" % (c+1))]
        DC_channel[c] = params[("QDAC_channel_q%d" % (c+1))]
    else:
        dc_bias[c] = params[("dc_bias%d" % (c+1))]+params["OPX_offsets"][c][0]
        DC_channel[c] = params[("OPX_channel_q%d" % (c+1))]
if params["use_QDAC_coupler"]:
    DC_channel_coupler = params["QDAC_channel_coupler"]
    dc_bias_coupler = params["dc_bias_coupler"]
else:
    DC_channel_coupler = params["OPX_channel_coupler"]
    dc_bias_coupler = params["dc_bias_coupler"]+params["OPX_offsets"][2][0]

dc_channels = np.append(DC_channel,DC_channel_coupler)

use_QDAC_params = ["use_QDAC_q1","use_QDAC_q2","use_QDAC_coupler"]
not_use_QDAC = [not params.get(key) for key in use_QDAC_params]

dc_offsets = np.array(list(zip(*params["OPX_offsets"])))[0]*not_use_QDAC
dc_offsets_end = np.array(list(zip(*params["OPX_offsets"])))[1]*not_use_QDAC

dc_channels_OPX = np.unique(np.array(
    [params["OPX_channel_q1"], params["OPX_channel_q2"], params["OPX_channel_coupler"]]))

amp_OPX_dc0 = 0.4 #amplitude for DC pulses, should be changed using *amp() in the qua program

wait_time = params["wait_time"]

#input=readout channels
########################################################
input_channel = np.array([params["Qubit1_input_channel"],params["Qubit2_input_channel"] ])
input_channel_offset = np.array([params["Qubit1_input_input_offset"],params["Qubit2_input_input_offset"] ])
###########################################################

#output channels
I_channel_ro = np.array([params["I_channel_ro_1"],params["I_channel_ro_2"]])
Q_channel_ro = np.array([params["Q_channel_ro_1"],params["Q_channel_ro_2"]])
I_output_offset_ro = np.array([params["I_output_offset_ro_1"],params["I_output_offset_ro_2"]])
Q_output_offset_ro = np.array([params["Q_output_offset_ro_1"],params["Q_output_offset_ro_2"]])
I_channel_drive = np.array([params["I_channel_drive_1"],params["I_channel_drive_2"]])
Q_channel_drive = np.array([params["Q_channel_drive_1"],params["Q_channel_drive_2"]])
I_output_offset_drive = np.array([params["I_output_offset_drive_1"],params["I_output_offset_drive_2"]])
Q_output_offset_drive = np.array([params["Q_output_offset_drive_1"],params["Q_output_offset_drive_2"]])
#I_channel_ro_monitor = params["I_channel_ro_monitor"]
#Q_channel_ro_monitor = params["Q_channel_ro_monitor"]
#I_channel_drive_monitor = params["I_channel_drive_monitor"]
#Q_channel_drive_monitor = params["Q_channel_drive_monitor"]

#Pulse - drive
pulse_length_drive = np.array([params["pulse_length_drive_qubit1"],params["pulse_length_drive_qubit2"]]) #ns
ampl_X = np.array([params["ampl_drive_qubit1"],params["ampl_drive_qubit2"]]) #amplitude for a pi rotation around X
ampl_X_12 = np.array([params["ampl_drive_qubit1_12"],params["ampl_drive_qubit2_12"]]) #amplitude for a pi rotation of 12 transition around X

#Pulse - readout
pulse_length_ro =  np.array([params["pulse_length_ro_qubit1"],params["pulse_length_ro_qubit2"]]) # ns
ampl_ro = np.array([params["ampl_ro_qubit1"],params["ampl_ro_qubit2"]])


# Readout
trigger_delay = 0
trigger_length = 10
time_of_flight = params["time_of_flight"]  # ns. must be at least 28

#MGs
mg_ro = [MG_ro(mg_address_readout[0]),MG_ro(mg_address_readout[1])]
mg_drive = MG_drive(mg_address_drive)


def init_config_generator():
    cg = config_generator.ConfigGenerator(
        output_offsets={I_channel_ro[0]: I_output_offset_ro[0], Q_channel_ro[0]: Q_output_offset_ro[0],
                        I_channel_ro[1]: I_output_offset_ro[1], Q_channel_ro[1]: Q_output_offset_ro[1],
                        I_channel_drive[0]: I_output_offset_drive[0],
                        Q_channel_drive[0]: Q_output_offset_drive[0],
                        I_channel_drive[1]: I_output_offset_drive[1],
                        Q_channel_drive[1]: Q_output_offset_drive[1],
                        dc_channels_OPX[0]:0.0, dc_channels_OPX[1]:0.0},
        input_offsets={input_channel[0]: input_channel_offset[0], input_channel[1]: input_channel_offset[1]})

    return cg

def create_readout_elements(cg):
    for c in range(1,3):
        cg.add_mixer("mixer_ro%d" % c, {(lo_freq_readout[c-1], if_freq_readout[c-1]): [1.0, 0.0, 0.0, 1.0]})
        cg.add_mixed_readout_element("readout%d" % c, lo_freq_readout[c-1] + if_freq_readout[c-1],
                                     lo_freq_readout[c-1], I_channel_ro[c-1],
                                     Q_channel_ro[c-1],
                                     {"out_I": input_channel[c-1], "out_Q": input_channel[c-1]},
                                     "mixer_ro%d" % c, time_of_flight)
        cg.add_integration_weight("simple_cos%d" % c, [1.0] * (pulse_length_ro[c-1] // 4),
                                  [0.0] * (pulse_length_ro[c-1] // 4))
        cg.add_integration_weight("simple_sin%d" % c, [0.0] * (pulse_length_ro[c-1] // 4),
                                  [1.0] * (pulse_length_ro[c-1] // 4))

def create_drive_elements(cg):
    for c in range(1,3):
        cg.add_mixer("mixer_drive%d" % c, {(lo_freq_drive, if_freq_drive[c-1]): [1.0, 0.0, 0.0, 1.0]})
        cg.add_mixer("mixer_drive%d_12" % c, {(lo_freq_drive, if_freq_drive_12[c - 1]): [1.0, 0.0, 0.0, 1.0]})

        cg.add_mixed_input_element("drive%d" % c, lo_freq_drive + if_freq_drive[c - 1], lo_freq_drive,
                                   I_channel_drive[c - 1], Q_channel_drive[c - 1],
                                   "mixer_drive%d" % c)

        cg.add_mixed_input_element("drive%d_12" % c, lo_freq_drive + if_freq_drive_12[c - 1], lo_freq_drive,
                                   I_channel_drive[c - 1], Q_channel_drive[c - 1],
                                   "mixer_drive%d_12" % c)


def create_pulses(cg, amp0 = None, amp0_12 = None):

    if amp0 is not None:
        ampl_X_local = np.array([amp0, amp0])
    else:
        ampl_X_local = ampl_X

    if amp0_12 is not None:
        ampl_X_12_local = np.array([amp0_12, amp0_12])
    else:
        ampl_X_12_local = ampl_X_12

    cg.add_constant_waveform("zero_waveform", 0.0)

    for c in range(1, 3):
        cg.add_constant_waveform("const_ro%d" % c, ampl_ro[c-1])

        cg.add_mixed_measurement_pulse("const_readout%d" % c, pulse_length_ro[c-1], ["const_ro%d" % c, "const_ro%d" % c],
                                       {"simple_cos": ("simple_cos%d" % c), "simple_sin": ("simple_sin%d" % c)},
                                       cg.TriggerType.RISING_TRIGGER, trigger_delay, trigger_length)
        cg.add_operation("readout%d" % c, "readout", "const_readout%d" % c)



        #pi rotation around X
        cg.add_constant_waveform("const_X_%d" % c, ampl_X_local[c-1])
        cg.add_mixed_control_pulse("const_X_%d" % c, pulse_length_drive[c-1], ["const_X_%d" % c, "zero_waveform"])

        cg.add_operation("drive%d" % c, "X_%d" % c, "const_X_%d" % c)

        # pi/2 rotation around X
        cg.add_constant_waveform("const_X/2_%d" % c, ampl_X_local[c - 1]/2)
        cg.add_mixed_control_pulse("const_X/2_%d" % c, pulse_length_drive[c - 1], ["const_X/2_%d" % c, "zero_waveform"])

        cg.add_operation("drive%d" % c, "X/2_%d" % c, "const_X/2_%d" % c)

        # 3*pi/2 rotation around X
        cg.add_constant_waveform("const_-X/2_%d" % c, -ampl_X_local[c - 1]/2)
        cg.add_mixed_control_pulse("const_-X/2_%d" % c, pulse_length_drive[c - 1], ["const_-X/2_%d" % c, "zero_waveform"])

        cg.add_operation("drive%d" % c, "-X/2_%d" % c, "const_-X/2_%d" % c)

        # pi rotation around Y
        cg.add_constant_waveform("const_Y_%d" % c, ampl_X_local[c - 1])
        cg.add_mixed_control_pulse("const_Y_%d" % c, pulse_length_drive[c - 1], ["zero_waveform", "const_Y_%d" % c])

        cg.add_operation("drive%d" % c, "Y_%d" % c, "const_Y_%d" % c)

        # pi/2 rotation around Y
        cg.add_constant_waveform("const_Y/2_%d" % c, ampl_X_local[c - 1] / 2)
        cg.add_mixed_control_pulse("const_Y/2_%d" % c, pulse_length_drive[c - 1], ["zero_waveform", "const_Y/2_%d" % c])

        cg.add_operation("drive%d" % c, "Y/2_%d" % c, "const_Y/2_%d" % c)

        # 3*pi/2 rotation around Y
        cg.add_constant_waveform("const_-Y/2_%d" % c, -ampl_X_local[c - 1] / 2)
        cg.add_mixed_control_pulse("const_-Y/2_%d" % c, pulse_length_drive[c - 1], ["zero_waveform", "const_-Y/2_%d" % c])

        cg.add_operation("drive%d" % c, "-Y/2_%d" % c, "const_-Y/2_%d" % c)

        # pulse which includes both X and Y components
        cg.add_mixed_control_pulse("XY_%d" % c, pulse_length_drive[c - 1], ["const_X_%d" % c, "const_X_%d" % c])

        cg.add_operation("drive%d" % c, "XY_%d" % c, "XY_%d" % c)

        # pi rotation around X for 12 transition
        cg.add_constant_waveform("const_X_%d_12" % c, ampl_X_12_local[c - 1])
        cg.add_mixed_control_pulse("const_X_%d_12" % c, pulse_length_drive[c - 1], ["const_X_%d_12" % c, "zero_waveform"])

        cg.add_operation("drive%d_12" % c, "X_%d_12" % c, "const_X_%d_12" % c)


def create_arbitrary_pulses(cg):
    Sx2_pulse_samples = np.loadtxt(Sx2_pulse_file, delimiter=',')
    sx2_I = pad_zeros(Sx2_pulse_samples[:,0])
    sx2_Q = pad_zeros(Sx2_pulse_samples[:,1])
    cg.add_arbitrary_waveform("Sx2_opt_I", sx2_I)
    cg.add_arbitrary_waveform("Sx2_opt_Q", sx2_Q)

    cg.add_mixed_control_pulse("Sx2_opt_pulse", len(sx2_I), ["Sx2_opt_I","Sx2_opt_Q"])

    cg.add_operation("drive2", "Sx2_opt", "Sx2_opt_pulse")




def add_OPX_dc_elements(cg):
    for dc_channel in dc_channels_OPX:
        cg.add_single_input_element("dc%d" % dc_channel , 0.0, dc_channel)

def add_OPX_dc_pulses(cg):
    #the default DC pulses has amplitude of amp_OPX_dc0 and duration of 16 ns.
    cg.add_constant_waveform("const_dc", amp_OPX_dc0)
    cg.add_single_control_pulse("const_dc", 16, "const_dc")

    for dc_channel in dc_channels_OPX:
        cg.add_operation("dc%d" % dc_channel, "const_dc", "const_dc")


#initialization and tear-off

def setup_MGs():
    mg_ro[0].setup_MG(lo_freq_readout[0]/1e6,lo_power_readout[0])
    mg_ro[1].setup_MG(lo_freq_readout[1]/1e6,lo_power_readout[1])
    mg_drive.setup_MG(lo_freq_drive/1e6,lo_power_drive)

def setup_DC(qm):
    with qdac_lib.qdac(QDAC_port) as qdac:

        for c in range(1, 3):
            if params[("use_QDAC_q%d" % c)]:
                qdac.setVoltageRange(DC_channel[c - 1], 10)
                qdac.setCurrentRange(DC_channel[c - 1], 1e-4)
                set_dc_voltage(qdac, DC_channel[c - 1], dc_bias[c - 1], True)
            else:
                set_dc_voltage(qm, DC_channel[c - 1], dc_bias[c - 1], False)

        if params["use_QDAC_coupler"]:
            qdac.setVoltageRange(DC_channel_coupler, 10)
            qdac.setCurrentRange(DC_channel_coupler, 1e-4)
            set_dc_voltage(qdac, DC_channel_coupler, dc_bias_coupler, True)
        else:
            set_dc_voltage(qm, DC_channel_coupler, dc_bias_coupler, False)

def MGs_off():
    mg_ro[0].set_on(False)
    mg_ro[1].set_on(False)
    mg_drive.set_on(False)

def DC_off(qm):
    with qdac_lib.qdac(QDAC_port) as qdac:
        for c in range(3):
            if params[use_QDAC_params[c]]:
                dc_device = qdac
            else:
                dc_device = qm
            set_dc_voltage(dc_device, dc_channels[c], dc_offsets_end[c], params[use_QDAC_params[c]])


def readout_rotation():
    #Macro for readout phase rotation
    if params["phase_rot_1"] > 0:
        frame_rotation(params["phase_rot_1"], "readout1")
    if params["phase_rot_2"] > 0:
        frame_rotation(params["phase_rot_2"], "readout2")