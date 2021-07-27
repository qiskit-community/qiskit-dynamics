from .dynamical_simulation import *


class MPOSimulationBuilder(SimulationBuilderStaticModel):

	def __init__(self, sim_def: SimulationDef):
		super().__init__(sim_def)
		self.parameters = None

	def build(self, subsystems, build_options = None):
		self._subsystems = subsystems
		self._build_options = build_options
		sim_def = self.sim_def

		H_ops = self._build_ops(sim_def.hamiltonian_operators)
		if sim_def.noise_operators is not None:
			L_ops = self._build_ops(sim_def.noise_operators)
		self._build_initial_state()
		self._build_observables()

	def _build_ops(self, ops) -> List:
		if type(ops) is list:
			if len(ops) > 0:
				op_type = type(ops[0])
			else:
				op_type = None
		else:
			raise  # must be a list, can be empty though
		if op_type is DynamicalOperator:  # USE instances!
			builder = OperatorBuilder()
			op_dicts = builder.build_dictionaries(ops, self._subsystems)
		elif op_type is None:
			op_dicts = []
		else:
			raise  # unsupported type
		return op_dicts
