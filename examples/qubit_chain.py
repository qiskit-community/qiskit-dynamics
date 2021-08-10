# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name,no-member,attribute-defined-outside-init
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

sim_def = SimulationDef(initial_state = rho_0,
						hamiltonian_operators = [H], hamiltonian_signals = [Signal(1.)],
						noise_operators = noise_operators, noise_signals = noise_signals,
						observable_operators = obs_1q + obs_2q)
sim_times = SimulationTimes(t_f = t_f, t_eval = t_eval)

full_builder = DenseSimulationBuilder(sim_def)
full_builder.build(subsystems)
full_sim = DenseSimulation(full_builder)
full_sim.solve(sim_times)

tmp = 2

sim_times.dt = 0.01
mpo_builder = LindbladMPOSimulationBuilder(sim_def)
mpo_params = {'output_files_prefix': s_output_path + s_file_prefix,
			  'cut_off_rho': 1e-16, 'max_dim_rho': 100}
mpo_builder.build(subsystems)
mpo_sim = LindbladMPOSimulation(mpo_builder)
mpo_sim.solve(sim_times, mpo_params)

tmp = 3

# H = .5 * Sz(0) * (Sx(1) + 2 * Sz(0)) + + 2 * Sz(0) * Sz(0)
# result = {('z', 0, 'x', 1): .5, (DynamicalOperatorKey('z', 0), DynamicalOperatorKey('z', 0)): 3.}

# Exceptions below, WIP

# prune_builder = DenseSimulationBuilder(sim_def)
# prune_subsystems = subsystems.copy()
# prune_subsystems[2] = 0
# prune_subsystems[3] = 0
# prune_builder.build(prune_subsystems)
# prune_builder.solve()
#

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
