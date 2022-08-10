
import sys
sys.path.append("..")
sys.path.append(r"\\132.64.80.214\overlordcommon\Users\Guy\PHD\repos\experiment-manager")
sys.path.append(r"\\132.64.80.214\overlordcommon\Users\Guy\PHD\repos\manage_qua")
sys.path.append(r"\\132.64.80.214\overlordcommon\Users\Guy\PHD\repos\manage_qua\files copied from Naftali")
import importlib
import device_parameters_2Q
import two_qubit_config_gates
import experiment_manager
import manage_qua
importlib.reload(experiment_manager)
importlib.reload(manage_qua)
importlib.reload(device_parameters_2Q)
importlib.reload(two_qubit_config_gates)

import experiment_manager as em
from manage_qua import *


# an example of usage:

# define an experiment: this example experiment is a rabi sequence but it also streams the parameters amplitude and
# duration such that we will be able to see that the loops work:
class RabiExperiment(QUAExperiment):
    def __init__(self):
        # define stream variable I with units a.u. and type float,
        #        stream variable amplitude with no units and type float
        #        stream varialbe duration with units 'clock-cycle' and type int:
        stream_config = QUAConfig(QUAParameter('I', None, 'a.u.', qua_type = float),
                                  QUAParameter('amplitude_stream',None, qua_type = float),
                                  QUAParameter('duration_stream',None,'clock-cycles' ,qua_type=int))

        #initialize stream variables:
        self.initialize_streams(stream_config)

        #labber stuff: #TODO maybe this needs to be elsewhere
        self.experiment_name ="rabi_test_qua_manage"
        self.labber_tags = ["rabi", "test", "simulation"]


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
        save(config.scale_amplitude.qua_var, self.stream.amplitude_stream.qua_var)
        save(config.duration.qua_var, self.stream.duration_stream.qua_var)

        if wait_time>0:
            wait(wait_time)


# test the Rabi experiment:
#define an instance of the new class:
rabi = RabiExperiment()

# define the experiment configurations for 2D loop: we have 2 QUA parameters that we are going to loop on,
# and two normal parameters that are constant.  looping on regular parameters (not in real-time) is not implemented yet
rabi_loop_config = QUAConfig(em.Parameter("repetitions", 1),
                             QUAParameter("duration",[100, 200, 500], units='clock cycles (4ns)'),
                             QUAParameter("scale_amplitude", [0.1,0.5,1.0, 2.5]),
                             em.Parameter('qubit_idx',2))

# use method nd_loops to get the QUA program
prog = rabi.nd_loop(rabi_loop_config)

# get qua configurations dict from device parameters (external file):
cg = init_config_generator()
create_readout_elements(cg)
create_drive_elements(cg)
create_pulses(cg)
add_OPX_dc_elements(cg)
config = cg.get_config()

# simulate the program
qmManager = QuantumMachinesManager()
job = qmManager.simulate(config,prog, SimulationConfig(10000))

# read into labber:
rabi.read_loop_job_into_labber(job, rabi_loop_config)

# get samples
samples = job.get_simulated_samples()

#get "measurement" results
results = job.result_handles

print('I:')
print(results.I.fetch_all()) # here we get zeros because this is just a simulation

print('amplitude_stream:')
print(results.amplitude_stream.fetch_all())

print('duration_stream:')
print(results.duration_stream.fetch_all())

#plot samples
samples.con1.plot()
