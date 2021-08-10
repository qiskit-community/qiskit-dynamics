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
from typing import Tuple, Sequence

from .dynamical_operators import *
from ..models.lindblad_models import *
from ..dispatch import Array
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info.states import Statevector, DensityMatrix
from qiskit_dynamics import solve_lmde
from scipy.integrate import OdeSolver


class SimulationDef:
	"""A container for the defining operators and signals of a dynamical simulation."""

	def __init__(self,
				 initial_state: Union[Array, QuantumState, BaseOperator, DynamicalOperator] = None,
				 hamiltonian_operators: Union[List[Array], List[Operator], List[DynamicalOperator]] = None,
				 hamiltonian_signals: Union[List[Signal], SignalList] = None,
				 noise_operators: Optional[Union[List[Array], List[Operator], List[DynamicalOperator]]] = None,
				 noise_signals: Optional[Union[List[Signal], SignalList]] = None,
				 observable_operators: Optional[Union[List[Array], List[BaseOperator],
													  List[DynamicalOperator]]] = None):
		"""Initialize the operators and signals of a dynamical simulation.

		Args:
			initial_state: The system state at the simulation start time.
            hamiltonian_operators: list of matices (as arrays), Operator objects, or DynamicalOperator,
            	where summation over the list elements is implied.
            hamiltonian_signals: Corresponding to the Hamiltonian operators, specifiable as either
             	a SignalList, a list of Signal objects, or as the inputs to signal_mapping.
            noise_operators: list of noise (jump) operators defining the Lindbladian dissipators.
            noise_signals: list of noise signals corresponding to the noise operators.
            observable_operators: The operators specifying the observables to be calculated from the
            	solution.

		"""
		self.initial_state = initial_state
		self.hamiltonian_operators = hamiltonian_operators
		self.hamiltonian_signals = hamiltonian_signals
		self.noise_operators = noise_operators
		self.noise_signals = noise_signals
		self.observable_operators = observable_operators
		# observable_labels: Optional[List] = None):


class SimulationTimes(ABC):
	"""A class storing simulation time parameters (that don't require rebuilding simulation operators)."""

	def __init__(self,
				 t_0: float = 0.,
				 t_f: Optional[float] = None,
				 dt: Optional[float] = None,
				 n_steps: Optional[int] = None,
				 t_eval: Optional[Sequence[float]] = None,
				 t_obs: Optional[Sequence[float]] = None):
		"""Set the simulation time and output times, implicitly calculating some parameters.

		Args:
			# t_span: ``Tuple`` or `list` of initial and final time.
			t_0: Initial simulation time, defaults to 0.
			t_f : Final simulation time. Must either be specified explicitly, or is calculated
				implicitly as ``t_f`` = ``t_0`` + ``n_steps`` * ``dt``, in which case ``n_steps``
				and ``dt`` must be defined.
			n_steps: Number of discrete time steps (``dt``) for fixed-time-step solvers. If it is
				None and ``t_f`` and ``dt`` are specified, it is automatically calculated from the
				formula above.
			dt: Discretization step for fixed-time-step solvers. If it is None and``t_f`` and ``dt``
				are specified, it is automatically calculated from the formula above.
			t_eval: Times at which to store the solution. Must lie within ``t_0`` and ``t_f``. If
				``dt`` is not None (or calculated implicitly), times must be multiples of ``dt``.
			t_obs: Times at which to calculate and store observables. Must lie within ``t_0`` and
				``t_f``. If ``dt`` is not None (or calculated implicitly), times must be multiples
				of ``dt``.
		"""

		if dt is not None and not dt > 0.:
			raise Exception("dt must be a positive floating point number, or left as None.")
		if n_steps is not None and not n_steps > 0:
			raise Exception("n_steps must be a positive integer number, or left as None.")
		if t_f is None:
			if t_0 is not None and n_steps is not None and dt is not None:
				t_f = t_0 + n_steps * dt
			else:
				raise Exception("t_f is not set explicitly, and cannot be deduced from dt and n_steps"
								"since these were not given valid values either.")
		else:
			if n_steps is not None and dt is not None:
				t_f_ = t_0 + n_steps * dt
				if abs(t_f_ - t_f) > dt / 2.:
					raise Exception("Parameter t_f was given a value inconsistent with n_steps and dt.")
			if n_steps is not None and dt is None:
				dt = (t_f - t_0) / n_steps
			if n_steps is None and dt is not None:
				n_steps = int((t_f - t_0) / dt)  # Rounding down
		self._verify_fixed_time_step(t_eval, dt, 't_eval')
		self._verify_fixed_time_step(t_obs, dt, 't_obs')

		self.t_0 = t_0
		self.t_f = t_f
		self.dt = dt
		self.n_steps = n_steps
		self.t_eval = t_eval
		self.t_obs = t_obs

	@staticmethod
	def _verify_fixed_time_step(t_array, dt, s_param_name):
		if t_array is not None and dt is not None:
			for t in t_array:
				t_mod = t % dt
				if t_mod / dt > 1e-6:  # TODO some relative precision here
					raise Exception(f'Times specified in {s_param_name} parameter must be integer'
									' multiples of the fixed time step parameter dt.')


class DenseSimulationBuilder(ABC):
	"""A class for building dense matrices for all model operators."""

	def __init__(self, sim_def: SimulationDef):
		self.sim_def = sim_def
		self._subsystem_dims = None
		self._prune_subsystems = None
		self.y0 = None
		self.model = None
		self.obs_matrices = []
		self.obs_labels = None

	def build(self,
			  subsystem_dims: Optional[OrderedDict] = None,
			  prune_subsystems: Optional[dict] = None,
			  frame: Optional[Union[Operator, Array]] = None,
			  cutoff_freq: Optional[float] = None,
			  validate: bool = True):
		"""Build the matrix representations of the model, initial state and observable operators.

		Args:
			subsystem_dims: If operators are defined as instances of ``DynamicalOperator``, this argument
			 	must be set to an ordered dictionary for each subsystem (identified using the system_id
				field of the DynamicalOperator), indicating the matrix dimension to assign for
				it, or 0 to discard it from the built results. Otherwise, it should be None.
			prune_subsystems: An optional dict specifying subsystem_dims to remove, as the keys of
				entries in the dict. The values are not used. This parameter can only be used if
				the operators are defined using DynamicalOperators.
            frame: Rotating frame operator. If specified with a 1d
                            array, it is interpreted as the diagonal of a
                            diagonal matrix.
            cutoff_freq: Frequency cutoff when evaluating the model.
            validate: If True check input operators are Hermitian.

        Raises:
            Exception: if operators are not Hermitian and ``validate`` is True.
		"""

		# TODO: default invoke of build() in solve() or __init__() - requires more parameters there.
		# TODO: Verify that simulation of matrices/superops is supported. Vectorized/nonvectorized.

		self._subsystem_dims = subsystem_dims
		self._prune_subsystems = prune_subsystems
		sim_def = self.sim_def

		H_ops, _ = self._build_ops(sim_def.hamiltonian_operators)
		hamiltonian = HamiltonianModel(operators = H_ops,
									   signals = sim_def.hamiltonian_signals,
									   frame = frame,
									   cutoff_freq = cutoff_freq,
									   validate = validate)
		if sim_def.noise_operators is not None:
			L_ops, _ = self._build_ops(sim_def.noise_operators)
			lindbladian = LindbladModel.from_hamiltonian(hamiltonian = hamiltonian,
														 noise_operators = L_ops,
														 noise_signals = sim_def.noise_signals)
			self.model = lindbladian
		else:
			self.model = hamiltonian
		self._build_initial_state()
		self._build_observables()

	def _build_initial_state(self):
		initial_state = self.sim_def.initial_state
		op_type = type(initial_state)
		if issubclass(op_type, DynamicalOperator):
			self.y0 = DensityMatrix(build_matrices(initial_state, self._subsystem_dims, self._prune_subsystems))
		elif issubclass(op_type, QuantumState):
			self.y0 = initial_state  # ? TODO
		elif issubclass(op_type, Operator) or issubclass(op_type, Array):
			self.y0 = initial_state  # ? TODO
		else:
			raise Exception(f"An unsupported type {op_type} passed as an initial state.")

	def _build_observables(self):
		self.obs_matrices, self.obs_labels = self._build_ops(self.sim_def.observable_operators)
		# if self.obs_labels is None:
		# 	self.obs_labels = obs_labels

	def _build_ops(self, ops) -> (List, List):
		op_type: Union[type, None] = None
		if type(ops) is list:
			if len(ops) > 0:
				op_type = type(ops[0])
		else:
			raise Exception("The Hamiltonian/noise/observable operators of a simulation definition "
							"must be passed as a list.")
		op_labels = []
		if op_type is None:
			op_matrices = []
		elif issubclass(op_type, DynamicalOperator):
			op_matrices = build_matrices(ops, self._subsystem_dims, self._prune_subsystems)
		elif issubclass(op_type, BaseOperator) or issubclass(op_type, Array):
			for op in ops:
				if type(op) != op_type:
					raise Exception("All operators in a list passed used in a simulation definition must "
									"be of identical type.")
			op_matrices = ops
		else:
			raise Exception(f"An unsupported type {op_type} passed as a Hamiltonian/noise operator.")
		return op_matrices, op_labels


class DenseSimulation(ABC):
	"""A class for solving and storing the results of a dynamical simulation using dense matrices."""

	def __init__(self, sim_builder: DenseSimulationBuilder):
		super().__init__()
		self.solution = None
		self.obs_data = None
		self.sim_builder = sim_builder

	def solve(self,
			  sim_times: SimulationTimes,
			  method: Optional[Union[str, OdeSolver]] = "DOP853",
			  input_frame: Optional[Union[str, Array]] = "auto",
			  solver_frame: Optional[Union[str, Array]] = "auto",
			  output_frame: Optional[Union[str, Array]] = "auto",
			  solver_cutoff_freq: Optional[float] = None,
			  **kwargs):
		"""Solve the built model and store the results.

		Args:
			sim_times: A class describing the different simulation (solution and observables) time
				parameters.
			method: Solving method to use.
			input_frame: Frame that the initial state is specified in. If ``input_frame == 'auto'``,
				defaults to using the frame the generator is specified in.
			solver_frame: Frame to solve the system in. If ``solver_frame == 'auto'``, defaults to
				using the drift of the generator when specified as a
				:class:`BaseGeneratorModel`.
			output_frame: Frame to return the results in. If ``output_frame == 'auto'``,
				defaults to using the frame the generator is specified in.
			solver_cutoff_freq: Cutoff frequency to use (if any) for doing the rotating
				wave approximation.
			kwargs: Additional arguments to pass to the solver.

		Returns:
			OdeResult: Results object.

		Raises:
			QiskitError: If specified method does not exist, or if dimension of y0 is incompatible
						 with generator dimension.
		"""
		sol = solve_lmde(self.sim_builder.model, t_span = Array((sim_times.t_0, sim_times.t_f)),
						 y0 = self.sim_builder.y0, method = method, t_eval = sim_times.t_eval,
						 input_frame = input_frame, solver_frame = solver_frame,
						 output_frame = output_frame, solver_cutoff_freq = solver_cutoff_freq,
						 **kwargs)
		self.solution = sol
		n_obs = len(self.sim_builder.obs_matrices)
		if n_obs == 0:
			self.obs_data = None
			return
		n_evals = len(sol.y)
		data = np.ndarray((n_obs, n_evals))
		for t_i, s_t in enumerate(sol.y):
			for i_obs, obs in enumerate(self.sim_builder.obs_matrices):
				data[i_obs, t_i] = s_t.expectation_value(obs)
		self.obs_data = data


# class SimulationBuilder(ABC):
# 	"""A base class for storing, building and solving a simulation, and calculating observables."""
#
# 	def __init__(self, sim_def: SimulationDef):
# 		self.sim_def = sim_def
# 		self._subsystem_dims = None
# 		self._build_options = None
#
# 	@abstractmethod
# 	def set_simulation_times(self,
# 							 t_0: float = 0.,
# 							 t_f: Optional[float] = None,
# 							 dt: Optional[float] = None,
# 							 n_steps: Optional[int] = None,
# 							 t_eval: Optional[Union[List[float], Tuple[float]]] = None,
# 							 t_obs: Optional[Union[List[float], Tuple[float]]] = None):
# 		"""Set the simulation time and output times."""
# 		pass
#
# 	@abstractmethod
# 	def set_initial_state(self, initial_state):
# 		pass
#
# 	@abstractmethod
# 	def set_observables(self, observable_operators):
# 		pass
#
# 	@abstractmethod
# 	def build(self, subsystem_dims, build_options = None):
# 		pass
#
# 	@abstractmethod
# 	def solve(self, **kwargs):
# 		pass

