# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""Tests for type_utils.py."""

import numpy as np
from scipy.sparse import csr_matrix

from qiskit_dynamics.array import Array
from qiskit_dynamics.type_utils import (
    vec_dissipator,
    vec_commutator,
)

from .common import QiskitDynamicsTestCase


class Testvec_commutator_dissipator(QiskitDynamicsTestCase):
    """Tests for vec_commutator and vec_dissipator."""

    def test_sparse_commutator_dissipator(self):
        """Tests that vec_commutator and vec_dissipator gives
        identical results, whether the array passed is a (k,n,n)
        Array or a (k,) Array of (n,n) sparse matrices."""
        np.random.seed(21301239)
        r = lambda *args: np.random.uniform(-1, 1, args)

        spm = csr_matrix(r(8, 8))
        self.assertAllClose(vec_commutator(spm).toarray(), vec_commutator(spm.toarray()))
        multi_matrix = r(3, 8, 8)
        den_commutator = vec_commutator(multi_matrix)
        sps_commutator = vec_commutator([csr_matrix(mat) for mat in multi_matrix])
        self.assertTrue(
            np.all(
                [
                    np.allclose(den_com, sps_com.toarray())
                    for den_com, sps_com in zip(den_commutator, sps_commutator)
                ]
            )
        )

        den_dissipator = vec_dissipator(multi_matrix)
        sps_dissipator = vec_dissipator([csr_matrix(mat) for mat in multi_matrix])
        self.assertTrue(
            np.all(
                [
                    np.allclose(den_dis, sps_dis.toarray())
                    for den_dis, sps_dis in zip(den_dissipator, sps_dissipator)
                ]
            )
        )
