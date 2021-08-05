
from lindbladmpo.LindbladMPOSolver import *
from .dynamical_simulation import *


class LindbladMPOSimulationBuilder(SimulationBuilderStaticModel):

	EQUALITY_PRECISION = 1e-9
	"""An absolute precision required for checking equality of two coefficients that should be equal."""

	def _build_observables(self):
		pass

	def solve(self, **kwargs):
		solver = LindbladMPOSolver(self.solver_params)
		solver.solve()
		tmp = 2

	def _build_initial_state(self):
		pass

	def __init__(self, sim_def: SimulationDef):
		super().__init__(sim_def)
		self.solver_params = None
		self._sys_indexes = None

	def build(self, subsystem_dims: OrderedDict, prune_subsystems: Optional[dict] = None,
			  build_options = None):
		self._subsystems = subsystem_dims
		self._sys_indexes = {}
		keys = []
		for key in subsystem_dims.keys():
			keys.append(key)
		for i_sys, sys_id in enumerate(subsystem_dims):
			self._sys_indexes[sys_id] = i_sys
		if build_options is None:
			build_options = {}
		sim_def = self.sim_def

		H_ops, _ = self._build_ops(sim_def.hamiltonian_operators)
		n_qubits = len(subsystem_dims)
		r_qubits = range(n_qubits)
		for dim in subsystem_dims.values():
			if dim != 2:
				raise Exception('All subsystems of the simulation (as given by the parameter '
								'subsystem_dims) must be two-level systems (qubits).')

		h_a_op = ['x', 'y', 'z']
		h_a_param = ['h_x', 'h_y', 'h_z']
		g_j_op = ['sp', 'sm', 'z']
		g_j_param = ['g_0', 'g_1', 'g_2']
		h_a = {}; g_j = {}; J_aa = {}
		for a in h_a_op:
			h_a[a] = np.zeros((n_qubits,))
			J_aa[a] = np.zeros((n_qubits, n_qubits))
		for j in g_j_op:
			g_j[j] = np.zeros((n_qubits,))
		for op_dict in H_ops:
			for key_tuple, val in op_dict.items():
				n_prod_ops = len(key_tuple)
				missing_key = None
				# b_unsupported = False
				b_identity_bond = False
				if n_prod_ops == 1:
					key: DynamicalOperatorKey = key_tuple[0]
					b_unsupported, missing_key, i_sys = self._validate_key(key, h_a_op)
					if i_sys is not None:
						h_a[key.s_type][i_sys] += val
				elif n_prod_ops == 2:
					key1: DynamicalOperatorKey = key_tuple[0]
					b_unsupported, missing_key, i_sys1 = self._validate_key(key1, h_a_op)
					if i_sys1 is not None:
						key2: DynamicalOperatorKey = key_tuple[1]
						b_unsupported, missing_key, i_sys2 = self._validate_key(key2, h_a_op)
						if i_sys2 is not None:
							if key1.s_type != key2.s_type:
								b_unsupported = True
							elif i_sys1 == i_sys2:
								b_identity_bond = True
							else:
								J_aa[key1.s_type][i_sys1, i_sys2] += val
				else:
					b_unsupported = True
				self._raise_validations(b_unsupported, b_identity_bond, missing_key,
										key_tuple, b_dissipator = False)

		if sim_def.noise_operators is not None:
			L_ops, _ = self._build_ops(sim_def.noise_operators)
			for op in L_ops:
				for key_tuple, val in op.items():
					n_prod_ops = len(key_tuple)
					missing_key = None
					# b_unsupported = False
					b_identity_bond = False
					if n_prod_ops == 1:
						key: DynamicalOperatorKey = key_tuple[0]
						b_unsupported, missing_key, i_sys = self._validate_key(key, g_j_op)
						if i_sys is not None:
							g_j[key.s_type][i_sys] += val
					else:
						b_unsupported = True
					self._raise_validations(b_unsupported, b_identity_bond, missing_key,
											key_tuple, b_dissipator = True)
		J_xx = J_aa['x']
		J_yy = J_aa['y']
		for i in r_qubits:
			for j in r_qubits:
				if abs(J_xx[i, j] - J_yy[i, j]) > self.EQUALITY_PRECISION:
					raise Exception(f'The interaction between the qubit with key {keys[i]} and the qubit'
									f'with key {keys[j]} has different coefficient for the XX and YY'
									' terms, which is unsupported by the Lindblad MPO solver.')
		params = {'N': n_qubits}
		if self.sim_def.dt > 0.:
			params['tau'] = self.sim_def.dt
		else:
			raise Exception("The discretization step field dt of the SimulationDef instance "
							"must be define for fixed-time-step solvers.")
		if self.sim_def.t_0 == 0. and self.sim_def.t_f > 0.:
			params['t_final'] = self.sim_def.t_f
		else:
			raise Exception("The simulation time span field t_span of the SimulationDef instance "
							"must have 0 as the initial time, and a positive final time.")
		for a in range(len(h_a_op)):
			params[h_a_param[a]] = h_a[h_a_op[a]]
		for j in range(len(g_j_op)):
			params[g_j_param[j]] = g_j[g_j_op[j]]
		params['J'] = J_xx
		params['J_z'] = J_aa['z']
		params['l_x'] = 0
		self.solver_params = params
		self.solver_params.update(build_options)
		self._build_initial_state()
		self._build_observables()

	def _validate_key(self, key: DynamicalOperatorKey, op_names: list):
		i_sys = None
		missing_key = None
		b_unsupported = False
		if key.s_type not in op_names:
			b_unsupported = True
		else:
			i_sys = self._sys_indexes.get(key.system_id, None)
			if i_sys is None:
				missing_key = key.system_id
		return b_unsupported, missing_key, i_sys

	@staticmethod
	def _raise_validations(b_unsupported, b_identity_bond, missing_key, key_tuple, b_dissipator):
		if b_unsupported:
			if b_dissipator:
				s_ops = 'dissipator'
			else:
				s_ops = 'or two-qubit Hamiltonian'
			raise Exception(f"The operator with key {key_tuple} is not one of the single-qubit"
							f" {s_ops} operators supported by the Lindblad MPO solver.")
		if b_identity_bond:
			raise Exception(f"The operator with key {key_tuple} does not correspond "
							"to a valid two-qubit interaction (it involves a single qubit).")
		if missing_key is not None:
			raise Exception(
				f"An operator was defined with id = {missing_key}, "
				"but this id does not appear in the subsystem_dims parameter.")

	@staticmethod
	def _build_ops(ops) -> (List, List):
		if type(ops) is list:
			if len(ops) > 0:
				op_type = type(ops[0])
			else:
				op_type = None
		else:
			raise Exception("The Hamiltonian/noise/observable operators of a simulation definition "
							"must be passed as a list.")
		op_labels = []
		if issubclass(op_type, DynamicalOperator):
			op_dicts, op_labels = build_dictionaries(ops)
		elif op_type is None:
			op_dicts = []
		else:
			raise Exception(f"An unsupported type {op_type} passed as a Hamiltonian/noise operator.")
		return op_dicts, op_labels
