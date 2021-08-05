from typing import Tuple

from .dynamical_operators import *
from ..models.lindblad_models import *
from ..dispatch import Array
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info.states import Statevector, DensityMatrix
from qiskit_dynamics import solve_lmde


class SimulationDef:
	"""A container for the defining elements of a dynamical simulation."""

	def __init__(self,
				 t_0: float = 0.,
				 t_f: Optional[float] = None,
				 dt: Optional[float] = None,
				 n_steps: Optional[int] = None,
				 t_eval: Optional[Union[List[float], Tuple[float]]] = None,
				 t_obs: Optional[Union[List[float], Tuple[float]]] = None,
				 initial_state: Union[Array, QuantumState, BaseOperator, DynamicalOperator] = None,
				 hamiltonian_operators: Union[List[Array], List[Operator], List[DynamicalOperator]] = None,
				 hamiltonian_signals: Union[List[Signal], SignalList] = None,
				 noise_operators: Optional[Union[List[Array], List[Operator], List[DynamicalOperator]]] = None,
				 noise_signals: Optional[Union[List[Signal], SignalList]] = None,
				 observable_operators: Optional[Union[List[Array], List[Operator], List[DynamicalOperator]]] = None,
				 observable_labels: Optional[List] = None):
		"""Initialize the definition of a dynamical simulation.

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
			initial_state: State at initial time.
		"""
		self.t_0 = None
		self.t_f = None
		self.dt = None
		self.n_steps = None
		self.t_eval = None
		self.t_obs = None
		self.set_simulation_times(t_0, t_f, dt, n_steps, t_eval, t_obs)
		self.initial_state = initial_state
		self.hamiltonian_operators = hamiltonian_operators
		self.hamiltonian_signals = hamiltonian_signals
		self.noise_operators = noise_operators
		self.noise_signals = noise_signals
		self.observable_operators = observable_operators
		self.observable_labels = observable_labels

	def set_simulation_times(self,
							 t_0: float = 0.,
							 t_f: Optional[float] = None,
							 dt: Optional[float] = None,
							 n_steps: Optional[int] = None,
							 t_eval: Optional[Union[List[float], Tuple[float]]] = None,
							 t_obs: Optional[Union[List[float], Tuple[float]]] = None):
		"""Set the simulation time and output times, implicitly calculating some parameters."""

		if dt is not None and not dt > 0.:
			raise Exception("dt must be a positive floating point number, or left as None.")
		if n_steps is not None and not n_steps > 0:
			raise Exception("n_steps must be a positive integer number, or left as None.")
		if t_f is None:
			t_f = t_0 + n_steps * dt
			if t_f is None:
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


class SimulationBuilder(ABC):
	"""A base class for storing, building and solving a simulation, and calculating observables."""

	def __init__(self, sim_def: SimulationDef):
		self.sim_def = sim_def
		self._subsystems = None
		self._build_options = None

	@abstractmethod
	def set_simulation_times(self,
							 t_0: float = 0.,
							 t_f: Optional[float] = None,
							 dt: Optional[float] = None,
							 n_steps: Optional[int] = None,
							 t_eval: Optional[Union[List[float], Tuple[float]]] = None,
							 t_obs: Optional[Union[List[float], Tuple[float]]] = None):
		"""Set the simulation time and output times."""
		pass

	@abstractmethod
	def set_initial_state(self, initial_state):
		pass

	@abstractmethod
	def set_observables(self, observable_operators):
		pass

	@abstractmethod
	def build(self, subsystems, build_options = None):
		pass

	@abstractmethod
	def solve(self, **kwargs):
		pass


class SimulationBuilderStaticModel(SimulationBuilder, ABC):
	"""A class implementing some common set functions that don't require rebuilding the model."""

	def __init__(self, sim_def: SimulationDef):
		super().__init__(sim_def)

	def set_simulation_times(self,
							 t_0: float = 0.,
							 t_f: Optional[float] = None,
							 dt: Optional[float] = None,
							 n_steps: Optional[int] = None,
							 t_eval: Optional[Union[List[float], Tuple[float]]] = None,
							 t_obs: Optional[Union[List[float], Tuple[float]]] = None):
		"""Set the simulation time and output times, no other side effects."""
		self.sim_def.set_simulation_times(t_0, t_f, dt, n_steps, t_eval, t_obs)

	def set_initial_state(self, initial_state):
		"""Set the initial state, and possibly rebuild its concrete representation."""
		self.sim_def.initial_state = initial_state
		if self._subsystems is not None:
			self._build_initial_state()

	def set_observables(self, observable_operators):
		"""Set the observable operators, and possibly rebuild their concrete representation."""
		self.sim_def.observable_operators = observable_operators
		if self._subsystems is not None:
			self._build_observables()

	@abstractmethod
	def _build_initial_state(self):
		pass

	@abstractmethod
	def _build_observables(self):
		pass


class DenseSimulationBuilder(SimulationBuilderStaticModel):
	"""A class for building dense matrices for all model operators."""

	def __init__(self, sim_def: SimulationDef):
		super().__init__(sim_def)
		self.y0 = None
		self.model = None
		self.solution = None
		self.obs_data = None
		self.obs_matrices = []

	def build(self, subsystems: Optional[OrderedDict] = None, build_options = None):
		"""Build the matrix representations of the model, initial state and observable operators.

		Args:
			subsystems: If operators are defined as instances of ``DynamicalOperator``, this argument
			 	must be set to an ordered dictionary for each subsystem (identified using the system_id
				field of the DynamicalOperator), indicating the matrix dimension to assign for
				it, or 0 to discard it from the built results. Otherwise, it should be None.
			build_options: An optional dictionary with options used for building the operators and
				Hamiltonian/Lindbladian model. The supported options are 'frame', 'cutoff_freq',
				and 'validate', which are passed to the constructor of ``HamiltonianModel``.
		"""
		self._subsystems = subsystems
		if build_options is None:
			build_options = {}
		self._build_options = build_options
		sim_def = self.sim_def

		H_ops, _ = self._build_ops(sim_def.hamiltonian_operators)
		hamiltonian = HamiltonianModel(operators = H_ops,
									   signals = sim_def.hamiltonian_signals,
									   frame = build_options.get('frame', None),
									   cutoff_freq = build_options.get('cutoff_freq', None),
									   validate = build_options.get('validate', True))
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

	def solve(self, **kwargs):
		"""Solve the built model (assumes build() was previously called), and store the results.

		Args:
			kwargs: keyword arguments passed directly to the solver.
		"""
		sol = solve_lmde(self.model, t_span = (self.sim_def.t_0, self.sim_def.t_f),
						 y0 = self.y0, t_eval = self.sim_def.t_eval, **kwargs)
		n_obs = len(self.obs_matrices)
		if n_obs == 0:
			self.obs_data = None
			return
		n_evals = len(sol.y)
		data = np.ndarray((n_obs, n_evals))
		for t_i, s_t in enumerate(sol.y):
			for obs_i, obs in enumerate(self.obs_matrices):
				data[obs_i, t_i] = s_t.expectation_value(obs)
		self.solution = sol
		self.obs_data = data

	def _build_initial_state(self):
		initial_state = self.sim_def.initial_state
		op_type = type(initial_state)
		if issubclass(op_type, DynamicalOperator):
			self.y0 = DensityMatrix(build_matrices(initial_state, self._subsystems))
		elif issubclass(op_type, QuantumState):
			self.y0 = initial_state  # ? TODO
		elif issubclass(op_type, BaseOperator) or issubclass(op_type, Array):
			self.y0 = initial_state  # ? TODO
		else:
			raise Exception(f"An unsupported type {op_type} passed as an initial state.")

	def _build_observables(self):
		self.obs_matrices, obs_labels = self._build_ops(self.sim_def.observable_operators)
		if self.sim_def.observable_labels is None:
			self.sim_def.observable_labels = obs_labels

	def _build_ops(self, ops) -> (List, List):
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
			op_matrices = build_matrices(ops, self._subsystems)
		elif issubclass(op_type, BaseOperator) or issubclass(op_type, Array):
			for op in ops:
				if type(op) != op_type:
					raise Exception("All operators in a list passed used in a simulation definition must "
									"be of identical type.")
			op_matrices = ops
		elif op_type is None:
			op_matrices = []
		else:
			raise Exception(f"An unsupported type {op_type} passed as a Hamiltonian/noise operator.")
		return op_matrices, op_labels
