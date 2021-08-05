from qiskit_dynamics.builders.operators_library import *
from qiskit_dynamics.builders.lindbladmpo_simulation_builder import *
from qiskit_dynamics.signals import Signal
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import os

s_output_path = os.path.abspath('./output') + '/'
if not os.path.exists(s_output_path):
	os.mkdir(s_output_path)
s_file_prefix = "chain"

N = 4
Jz = 1.
g1 = 0.1
t_f = 2.2
step = 0.02
t_span = [0, t_f]
t_eval = np.arange(0., t_f, step)

r_qubits = range(N)
subsystems = OrderedDict()
H = .5 * Sz(0)
rho_0 = .5 *(Id(0) + Sz(0))
noise_signals = [Signal(g1)] * N
noise_operators = []; obs_1q = []; obs_2q = []

for i_qubit in r_qubits:
	subsystems[i_qubit] = 2
	obs_1q.append(Sz(i_qubit))
	if i_qubit > 0:
		H += .5 * (Sz(i_qubit) + Sx(i_qubit))
		H += .5 * Jz * Sz(i_qubit - 1) * Sz(i_qubit)
		rho_0 *= .5 *(Id(i_qubit) - Sx(i_qubit))
		obs_2q.append(Sz(i_qubit - 1) * Sz(i_qubit))
	noise_operators.append(Sp(i_qubit))

sim_def = SimulationDef(t_f = t_f, t_eval = t_eval, initial_state = rho_0,
						hamiltonian_operators = [H], hamiltonian_signals = [Signal(1.)],
						noise_operators = noise_operators, noise_signals = noise_signals,
						observable_operators = obs_1q + obs_2q)
sim_full = DenseSimulationBuilder(sim_def)
sim_full.build(subsystems)
sim_full.solve()

tmp = 2

# H = .5 * Sz(0) * (Sx(1) + 2 * Sz(0)) + + 2 * Sz(0) * Sz(0)
# result = {('z', 0, 'x', 1): .5, (DynamicalOperatorKey('z', 0), DynamicalOperatorKey('z', 0)): 3.}

# Exceptions below, WIP

# sim_prune = DenseSimulationBuilder(sim_def)
# subsystems_pruned = subsystem_dims.copy()
# subsystems_pruned[2] = 0
# subsystems_pruned[3] = 0
# sim_prune.build(subsystems_pruned)
# sim_prune.solve()
#
sim_def.dt = 0.01
sim_mpo = LindbladMPOSimulationBuilder(sim_def)
mpo_params = {'output_files_prefix': s_output_path + s_file_prefix,
			  'cut_off_rho': 1e-16, 'max_dim_rho': 100}
sim_mpo.build(subsystems, build_options = mpo_params)
sim_mpo.solve()

tmp = 3
# plot comparative graphs


# def plot_space_time(data: np.ndarray, t_steps: np.ndarray,  # t_ticks: np.ndarray, qubits: np.ndarray,
# 					ax = None, fontsize = 16, b_save_figures = True, s_file_prefix = ''):
# 	if ax is None:
# 		_, ax = plt.subplots(figsize = (14, 9))
# 	plt.rcParams.update({'font.size': fontsize})
# 	im = ax.imshow(data, interpolation = 'none', aspect = 'auto')
# 	divider = make_axes_locatable(ax)
# 	cax = divider.append_axes('right', size = '5%', pad = 0.05)
# 	plt.colorbar(im, cax = cax)
# 	# plt.colorbar(im)
# 	ax.set_xlabel('$t$', fontsize = fontsize)
# 	ax.set_xticks(t_steps)
# 	# ax.set_xticklabels(t_ticks, fontsize = fontsize)
# 	# ax.set_yticks(qubits)
# 	# ax.set_yticklabels(qubits, fontsize = fontsize)
# 	ax.set_ylabel('qubits', fontsize = fontsize)
# 	ax.set_title('$\\langle\sigma_z(t)\\rangle$')
# 	if b_save_figures:
# 		plt.savefig(s_file_prefix + '.sigma_z.png')
#
#
# plot_space_time(sim_dense.obs_data[0:4, :], np.asarray(t_eval), b_save_figures = False)
# plt.show()
