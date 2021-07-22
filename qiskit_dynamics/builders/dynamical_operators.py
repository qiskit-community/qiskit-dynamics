from typing import OrderedDict, Dict, List, Union, Any
from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy
from qiskit.quantum_info.operators.base_operator import BaseOperator


class DynamicalOperator(ABC):
	"""An class for operators used in defining dynamical simulations."""

	DEFAULT_ALIASES = \
		{
			# Note that both strings must be lower case
			'id': 'i',
			'sx': 'x',
			'sy': 'y',
			'sz': 'z',
		}
	"""Aliases available by default for the two-level qubit operators that are realized in this class."""

	def __init__(self, system_id: Any = '', s_type = '', aliases = None):
		""" Initialization of an operator using (optional) id of the subsystem, type, and aliases.

		Args:
			system_id: A unique identifier of the subsystem (degree of freedom) of the operator,
				or an empty string to defer such an identification.
			s_type: A string name of the type of operator.
			aliases: A dictionary of aliases for operator types, allowing users to expand the
				s_type names used for known operators. If ``None``, the field ``DEFAULT_ALIASES``
				is used.
		"""
		self.system_id = system_id
		self._s_type = ''
		self._s_type_unique = ''
		if aliases is None:
			aliases = self.DEFAULT_ALIASES
		self.aliases = aliases
		# must be assigned first, before setting the property self.s_type, which searches the aliases.

		self.s_type = s_type  # must be assigned after self.aliases, because it searches the aliases.
		self.compound_type = ''
		self.compound_ops = None

	def __deepcopy__(self, memo = None):
		cc = DynamicalOperator(self.system_id, self.s_type, self.aliases)
		cc.compound_type = self.compound_type
		cc.compound_ops = deepcopy(self.compound_ops, memo)
		return cc

	def new_operator(self):
		"""A method that must be implemented by subclasses, to return the correct instance."""
		return DynamicalOperator(aliases = self.aliases)

	@property
	def s_type(self) -> str:
		"""A string defining the operator type. Aliases (multiple names) may be used."""
		return self._s_type

	@property
	def s_type_unique(self) -> str:
		"""A unique string defining the operator type, which all aliases must map to."""
		return self._s_type_unique

	@s_type.setter
	def s_type(self, s_type):
		"""A setter of the operator's type, which searches aliases and sets the unique type as well."""
		self._s_type = s_type
		s_type = self.s_type.lower()  # should be here after assigning _s_type with user's string as is
		self._s_type_unique = self.aliases.get(s_type, s_type)

	def __add__(self, other):
		"""Addition of two DynamicalOperators. Returns a new (compound) DynamicalOperator."""
		if not isinstance(other, DynamicalOperator):
			raise Exception("Both operands in an addition must be instances of a DynamicalOperator.")
		result = self.new_operator()
		result.compound_type = '+'
		result.compound_ops = [self, other]
		return result

	# def __pow__(self, power, modulo=None):
	# 	if type(power) is not int:
	# 		raise Exception("Only integer powers are currently supported for operators.")
	# 	result = DynamicalOperator(aliases = self.aliases)
	# 	result.compound_type = '**'
	# 	result.compound_ops = [self, power]
	# 	return result

	def __sub__(self, other):
		"""Subtraction of two DynamicalOperators. Returns a new (compound) DynamicalOperator."""
		return self.__add__(-other)

	def __mul__(self, other):
		"""Multiplication by a DynamicalOperator or a scalar."""
		result = self.new_operator()
		if isinstance(other, DynamicalOperator):
			result.compound_type = '@'
			result.compound_ops = [self, other]
			return result
		else:
			other_type = type(other)
			if other_type is complex or other_type is float or other_type is int:
				result.compound_type = '*'
				result.compound_ops = [self, other]
				return result
		raise Exception("The second operand of a multiplication must be a DynamicalOperator class or a scalar.")

	def __rmul__(self, other):
		"""Multiplication of a DynamicalOperator by a scalar."""
		result = self.__mul__(other)
		return result

	def __neg__(self):
		"""Unary negation of a DynamicalOperator."""
		result = self.__rmul__(-1.)
		return result

	def __pos__(self):
		"""Unary plus operator prepending a DynamicalOperator."""
		return self

	def get_operator_matrix(self, s_type_unique, dim: int) -> np.ndarray:
		"""Returns a matrix describing a concrete realization of the DynamicalOperator.

		This function is not declared as static in order to allow subclasses to override the
		implementation. However, fields of the ``self`` object are not being used.
		Args:
			s_type_unique: A unique operator type name to generate.
			dim: The physical dimension of the matrix to generate.
		"""
		# TODO: replace qubit creation with Qiskit operator.from_label()
		# TODO: support 'i x y z sp sm n p q a a_ n n^2 0 1 p_0_1'
		if dim == 2:
			if s_type_unique == 'i':
				return np.identity(dim)
			elif s_type_unique == 'x':
				return np.asarray([[0, 1], [1, 0]], complex)
			elif s_type_unique == 'y':
				return np.asarray([[0, -1j], [1j, 0]], complex)
			elif s_type_unique == 'z':
				return np.asarray([[1, 0], [0, -1]], complex)
			elif s_type_unique == 'sp':
				return np.asarray([[0, 1], [0, 0]], complex)
			elif s_type_unique == 'sm':
				return np.asarray([[0, 0], [1, 0]], complex)
			elif s_type_unique == '0':
				return np.asarray([[1, 0], [0, 0]], complex)
			elif s_type_unique == '1':
				return np.asarray([[0, 0], [0, 1]], complex)
		# Just a partial implementation for now
		raise Exception(
			f"Operator type {s_type_unique} unknown or unsupported for matrix generation with dimension {dim}.")


class DynamicalOperatorKey:
	"""A container for a unique key identifying an operator and a subsystem."""

	def __init__(self, op: DynamicalOperator):
		self.system_id = op.system_id
		self.s_type_unique = op.s_type_unique

	def __hash__(self):
		return hash((self.system_id, self.s_type_unique))


class OperatorBuilder(ABC):

	def __init__(self):
		self._ids = None
		self._identity_matrices = None
		self._subsystem_dims = None
		self._total_dim = 0
		self._dyn_op = None

	def build_dictionaries(self, operators: Union[DynamicalOperator, List[DynamicalOperator]]) -> List[Dict]:
		results = []
		b_flatten = False
		if type(operators) != list:
			b_flatten = True
			operators = [operators]
		for op in operators:
			results.append(self._build_one_dict(op))
		if b_flatten:
			results = results[0]
		return results

	def _build_one_dict(self, operator: DynamicalOperator):
		# TODO verify value semantics of dict (object) keys
		if operator.compound_type == '+':
			result = {}
			for op in operator.compound_ops:
				op_dict: dict = self._build_one_dict(op)
				for key, val in op_dict.items():
					val_sum = val + result.get(key, complex(0.))
					result[key] = val_sum
		elif operator.compound_type == '@':
			new_key = []
			new_val = complex(1.)
			for op in operator.compound_ops:
				op_dict = self._build_one_dict(op)
				for key, val in op_dict.items():
					for key_element in key:
						new_key.append(key_element)
					new_val *= val
			result = {tuple(new_key): new_val}
		elif operator.compound_type == '*':
			op = operator.compound_ops[0]
			scalar = operator.compound_ops[1]
			op_dict = self._build_one_dict(op)
			for key, val in op_dict.items():
				op_dict[key] = val * scalar
			result = op_dict
		elif operator.compound_type == '':
			result = {tuple([DynamicalOperatorKey(operator)]): complex(1.)}
		else:
			raise Exception(f"Unknown/unsupported concatenation operator {operator.compound_type}.")
		return result

	def build_matrices(self, operators: Union[DynamicalOperator, Dict, List[DynamicalOperator], List[Dict]],
					   subsystem_dims: OrderedDict, dyn_op: DynamicalOperator = None):
		if len(subsystem_dims) == 0:
			return None
		b_flatten = False
		if type(operators) != list:
			b_flatten = True
			operators = [operators]
		dims = subsystem_dims.values()
		self._total_dim = 1
		for dim in dims:
			self._total_dim *= dim
		if len(operators) == 0:
			return np.zeros(self._total_dim)
		b_dictionaries = False
		for op in operators:
			op_type = type(op)
			if op_type is dict:
				b_dictionaries = True
			elif not isinstance(op, DynamicalOperator):
				raise Exception(f"Unsupported class type in parameter operators: {op_type}.")
			elif b_dictionaries:
				raise Exception("All operators must be of the same type (a dictionary or a DynamicalOperator).")

		if b_dictionaries:
			operators_dict: List[Dict] = operators
		else:
			operators_dict = self.build_dictionaries(operators)
		self._subsystem_dims = subsystem_dims
		self._ids = []
		for s_id in subsystem_dims.keys():
			self._ids.append(s_id)
		self._identity_matrices = []
		for dim in dims:
			self._identity_matrices.append(np.identity(dim, complex))
		results = []
		self._dyn_op = dyn_op
		if self._dyn_op is None:
			self._dyn_op = DynamicalOperator()
		for op_dict in operators_dict:
			results.append(self._build_one_matrix(op_dict))
		if b_flatten:
			results = results[0]
		return results

	def _build_one_matrix(self, operator_dict: Dict) -> np.ndarray:
		matrix = np.zeros((self._total_dim, self._total_dim), complex)
		for key, val in operator_dict.items():
			sub_matrices = {}
			for key_element in key:
				operator_key: DynamicalOperatorKey = key_element
				dim = self._subsystem_dims.get(operator_key.system_id, None)
				if dim is None:
					raise Exception(
						f"An operator was defined with id = {operator_key.system_id}, but this id does not appear in the subsystem_dims parameter.")
				new_sub_matrix = self._dyn_op.get_operator_matrix(operator_key.s_type_unique, dim)
				sub_matrix = sub_matrices.get(operator_key.system_id, None)
				if sub_matrix is not None:
					new_sub_matrix = sub_matrix @ new_sub_matrix  # note that order of matrix product matters
				sub_matrices[operator_key.system_id] = new_sub_matrix
			op_matrix = None
			n_subsystems = len(self._ids)
			for i in range(n_subsystems):
				sub_matrix = sub_matrices.get(self._ids[i], self._identity_matrices[i])
				if i == 0:
					op_matrix = sub_matrix
				else:
					op_matrix = np.kron(op_matrix, sub_matrix)
			matrix += val * op_matrix  # TODO verify does not require the Identity?
		return matrix


class Sx(DynamicalOperator):
	def __init__(self, system_id = ''):
		super().__init__(system_id, 'x')


class Sy(DynamicalOperator):
	def __init__(self, system_id = ''):
		super().__init__(system_id, 'y')


class Sz(DynamicalOperator):
	def __init__(self, system_id = ''):
		super().__init__(system_id, 'z')


class Sp(DynamicalOperator):
	def __init__(self, system_id = ''):
		super().__init__(system_id, 'sp')


class Sm(DynamicalOperator):
	def __init__(self, system_id = ''):
		super().__init__(system_id, 'sm')


class Sid(DynamicalOperator):
	def __init__(self, system_id = ''):
		super().__init__(system_id, 'id')
