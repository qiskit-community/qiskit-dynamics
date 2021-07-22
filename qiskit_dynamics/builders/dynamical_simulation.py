from .dynamical_operators import *
from ..models.lindblad_models import *
from ..dispatch import Array
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info.states import Statevector, DensityMatrix
from qiskit_dynamics import solve_lmde


class SimulationDef:

	def __init__(self,
				 t_span: Array,
				 t_eval: Array,
				 initial_state: Union[Array, QuantumState, BaseOperator, DynamicalOperator],
				 hamiltonian_operators: Union[List[Array], List[Operator], List[DynamicalOperator]],
				 hamiltonian_signals: Union[List[Signal], SignalList],
				 noise_operators: Optional[Union[List[Array], List[Operator], List[DynamicalOperator]]] = None,
				 noise_signals: Optional[Union[List[Signal], SignalList]] = None,
				 observable_operators: Optional[Union[List[Array], List[Operator], List[DynamicalOperator]]] = None,
				 observable_labels: Optional[List] = None):
		self.t_span = t_span
		self.t_eval = t_eval
		self.initial_state = initial_state
		self.hamiltonian_operators = hamiltonian_operators
		self.hamiltonian_signals = hamiltonian_signals
		self.noise_operators = noise_operators
		self.noise_signals = noise_signals
		self.observable_operators = observable_operators
		self.observable_labels = observable_labels


class SimulationBuilder(ABC):

	def __init__(self, sim_def: SimulationDef):
		self.sim_def = sim_def
		self._subsystems = None
		self._build_options = None

	@abstractmethod
	def set_times(self, t_span, t_eval):
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


class SimulationBuilderStatic(SimulationBuilder, ABC):

	def __init__(self, sim_def: SimulationDef):
		super().__init__(sim_def)

	def set_times(self, t_span, t_eval):
		self.sim_def.t_span = t_span
		self.sim_def.t_eval = t_eval

	def set_initial_state(self, initial_state):
		self.sim_def.initial_state = initial_state
		if self._subsystems is not None:
			self._build_initial_state()

	def set_observables(self, observable_operators):
		self.sim_def.observable_operators = observable_operators
		if self._subsystems is not None:
			self._build_observables()

	@abstractmethod
	def _build_initial_state(self):
		pass

	@abstractmethod
	def _build_observables(self):
		pass


class DenseSimulationBuilder(SimulationBuilderStatic):

	def __init__(self, sim_def: SimulationDef):
		super().__init__(sim_def)
		self.y0 = None
		self.model = None
		self.solution = None
		self.obs_data = None
		self.obs_matrices = []

	def build(self, subsystems, build_options = None):
		self._subsystems = subsystems
		self._build_options = build_options
		sim_def = self.sim_def

		H_ops, _ = self._build_ops(sim_def.hamiltonian_operators)
		hamiltonian = HamiltonianModel(operators = H_ops, signals = sim_def.hamiltonian_signals)
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
		sol = solve_lmde(self.model, t_span = self.sim_def.t_span, y0 = self.y0,
						 t_eval = self.sim_def.t_eval, **kwargs)
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
			builder = OperatorBuilder()
			self.y0 = DensityMatrix(builder.build_matrices(initial_state, self._subsystems))
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
			builder = OperatorBuilder()
			op_matrices = builder.build_matrices(ops, self._subsystems)
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

