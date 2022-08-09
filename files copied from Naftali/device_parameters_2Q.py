import instruments_py27.anritsu as Antitsu_MG
import instruments_py27.E8241A as E8241A_MG
import instruments_py27.M9347A as M9347A_MG
import numpy as np

device_parameters = \
    {
        "mg_address_readout1": ("TCPIP0::DESKTOP-VT04ESJ::hislip1::INSTR",1),
        "mg_address_readout2": ("TCPIP0::DESKTOP-VT04ESJ::hislip1::INSTR",2),
        "mg_address_drive" : "GPIB0::5::INSTR", #VISA address of XY drive microwave generator
        "MG_ro": M9347A_MG.M9347A_MG,#, #class of readout microwave generator
        "MG_drive": Antitsu_MG.Anritsu_MG,#E8241A_MG.E8241A_MG, #class of XY drive microwave generator
        "f_r1": 4960.4e6, #center readout 1 frequency ,Hz
        "f_r2": 6177.5e6,#6193.5e6, # 4960.4e6, #center readout 2 frequency ,Hz
        "lo_freq_readout1":4960.4e6+150e6,# Readout 1 local oscillator  frequency, Hz
        "lo_freq_readout2" :6177.5e6+150e6,# 5107.3e6,# Readout 2 local oscillator frequency, Hz
        "lo_power_readout1": 10.0,  # Readout 1 local oscillator power, dBm
        "lo_power_readout2": 10.0,  # Readout 1 local oscillator power, dBm
        "f_d1": 4007e6,#4017e6,# 4000.5e6,# #4009.81e6,#4000e6,#4064e6, #4018e6,#4024e6,# Center XY drive 1 frequency, Hz
        "f_d1_12": 3794e6,# 3741    e6,#4007e6-220e6,#1->2 frequency qubit 1, Hz
        "f_d2":4009.5e6,#4209.6e6,#64263.5e6,#4009.5e6,# 4012.1e6,#4263.5e6,##4101.7e6,# 4007.5e6,#4009.81e6,#4261.085e6,#4151.6e6,#4272e6,#4009.81e6,#4000e6,#4064e6, #4018e6,#4024e6,# Center XY drive 2 frequency, Hz
        "f_d2_12":3793e6,# 3974e6,#3793e6+200e6,#3793e6,  # 1->2 frequency qubit 2, Hz
        "lo_freq_drive":  4063e6,#4201e6,#4321e6,#4420e6,#4400e6,#4250e6,#4221.0e6,#4139e6,#4161.0e6, # # XY drive local oscillator frequency, Hz
        "lo_power_drive": 10.0,  # XY drive local oscillator power, dBm
        #Notice that the OPX offsets will be added to the next 3 values when applicable
        "dc_bias1": -1.05,#DC bias qubit 1 voltage, Volt
        "dc_bias2":-317.5e-3+8.03e-3,#13.7e-3,#DC bias qubit 2 voltage, Volt. *10 because of attenuation on the opx bias outputs. be careful if switching to QDAC!
        "dc_bias_coupler": -122e-3+5.37e-3,#DC bias for coupler voltage, Volt
        "dc_coupler_off" : -122e-3, #the coupler dc bias in which the coupling is off
        "QDAC_port": "COM7",
        "use_QDAC_q1" : True, #use QDAC for DC bias or OPX
        "use_QDAC_q2": False,
        "use_QDAC_coupler": False,
        "QDAC_channel_q1": 2, #QDAC channel qubit #1
        "QDAC_channel_q2": 10,#2, #QDAC channel qubit #2
        "QDAC_channel_coupler": 18, #QDAC channel qubit #2
        "OPX_channel_q1": 10,
        "OPX_channel_q2": 8,
        "OPX_channel_coupler": 10,
        "OPX_offsets": [(0.0,0.0),(-8.03e-3,-18.8e-3),(2e-3,-5.37e-3)], #q1,q2,coupler (with pulse, without pulse), V
        "Qubit1_input_channel": 1, #OPX analog input channel for readout qubit 1 readout
        "Qubit2_input_channel": 2, #OPX analog input channel for readout qubit 2 readout
        "Qubit1_input_input_offset": 0.0, #mixer offset for OPX analog input channel for qubit 1 readout, Volt
        "Qubit2_input_input_offset": 0.0, #mixer offset for OPX analog input channel for qubit 2 readout, Volt
        "I_channel_ro_1": 1, #OPX analog output channel for readout I for qubit 1
        "Q_channel_ro_1": 2, #OPX analog output channel for readout Q for qubit 1
        "I_channel_ro_2": 5,  # OPX analog output channel for readout I for qubit 2
        "Q_channel_ro_2": 6,  # OPX analog output channel for readout Q for qubit 2
        "I_output_offset_ro_1": 0, #mixer offset for OPX analog output channel for readout I for qubit 1, Volt
        "Q_output_offset_ro_1": 0, #mixer offset for OPX analog output channel for readout Q for qubit 1, Volt
        "I_output_offset_ro_2": 0,  # mixer offset for OPX analog output channel for readout I for qubit 2, Volt
        "Q_output_offset_ro_2": 0,  # mixer offset for OPX analog output channel for readout Q for qubit 2, Volt
        "I_channel_drive_1": 3, #OPX analog output channel for XY drive I for qubit 1
        "Q_channel_drive_1": 4, #OPX analog output channel for XY drive Q for qubit 1
        "I_output_offset_drive_1": 0.0,#-0.06011239, #mixer offset for OPX analog output channel for XY drive I for qubit 1, Volt
        "Q_output_offset_drive_1": 0.0,#-0.05336217, #mixer offset for OPX analog output channel for XY drive Q for qubit 1, Volt
        "I_channel_drive_2":  7,#3, #OPX analog output channel for XY drive I for qubit 2
        "Q_channel_drive_2": 9,#4, #OPX analog output channel for XY drive Q for qubit 2
        "I_output_offset_drive_2": 0.0,#-0.07735229, #mixer offset for OPX analog output channel for XY drive I for qubit 2, Volt
        "Q_output_offset_drive_2": 0.0,#-0.05746273,#-0.098644, #mixer offset for OPX analog output channel for XY drive Q for qubit 2, Volt
        # "I_channel_ro_monitor": 9,#7, #OPX analog output channel for readout I monitor
        # "Q_channel_ro_monitor": 10,#,6, #OPX analog output channel for readout Q monitor
        # "I_channel_drive_monitor": 9,#5, #OPX analog output channel for XY drive I monitor
        # "Q_channel_drive_monitor": 10,#8, #OPX analog output channel for XY drive Q monitor
        "pulse_length_drive_qubit1":  64,#120000,#,###512,##XY drive pulse length for qubit 1, ns
        "pulse_length_drive_qubit2":  64,#120000,##512,##XY drive pulse length for qubit 2, ns
        "ampl_drive_qubit1":0.0015,#28.5e-3,# ,#0.0005,##0.002,#0.00294,#0.01, # #OPX XY drive amplitude for qubit 1, Volt
        "ampl_drive_qubit2":0.0015,# 30e-3,#33.4e-3,#18.1e-3,##0.0002,#3.9e-3,#0.00294,#0.01, # #OPX XY drive amplitude for qubit 2, Volt
        "ampl_drive_qubit1_12": 28.5e-3,#33.4e-3/np.sqrt(2),  #OPX XY drive amplitude for qubit 1 1->2 transition, Volt
        "ampl_drive_qubit2_12":30e-3,# 33.4e-3/np.sqrt(2), #OPX XY drive amplitude for qubit 2 1->2 transition, Volt
        "ampl_ro_qubit1": 0.0015, #changed for simulation #0.1,  # 0.1, #OPX readout amplitude for qubit 1, Volt
        "ampl_ro_qubit2": 0.0015, #changed for simulation   #0.1,#0.1, #OPX readout amplitude for qubit 2, Volt
        "pulse_length_ro_qubit1": 1000,#2000, #Readout pulse length for qubit 1, ns
        "pulse_length_ro_qubit2": 1000,  # 2000, #Readout pulse length for qubit 2, ns
        "time_of_flight": 252,#256, #ns
        "wait_time": 150, #changed for simulation #30000#500, # *4 ns. Time to wait after each experiment for qubit decay
        "phase_rot_1": 0.445104,# rotation of readout, qubit 1 (radians)
        "phase_rot_2": -0.4492,# # rotation of readout, qubit 2 (radians)
        "threshold_1": 0, #threshold for qubit 1 measurement
        "threshold_sign_1" : 1,#1 if mean g qubit 1 < threshold_1 , -1 otherwise
        "mean_g_1": 0, #mean measurement value for ground state qubit #1
        "mean_e_1": 0, #mean measurement value for excited state qubit #1
        "g_c_1": 1,  # qubit 1 ground state fidelity using threshold
        "e_c_1": 1,  # qubit 1 excited state fidelity using threshold
        "threshold_2": -0.002461, #threshold for qubit 2 measurement
        "threshold_sign_2": 1,  # 1 if mean g qubit 2 < threshold_2, -1 otherwise
        "mean_g_2":  -0.003218, #mean measurement value for ground state qubit #2
        "mean_e_2": -0.001705, #mean measurement value for excited state qubit #2
        "g_c_2": 0.8116, #qubit 2 ground state fidelity using threshold
        "e_c_2": 0.7862, #qubit 2 excited state fidelity using threshold
    }

