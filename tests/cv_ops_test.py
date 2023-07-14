from cv_tfq.cv_ops import BinaryOp, PositionOp, MomentumOp
from cv_tfq.cv_subroutines import ComputationalLayerInteger
import tensorflow_quantum as tfq
import numpy as np
import pytest
import cirq
import unittest


#######################
# Comparison variables
#######################

test_indices = [1, 3, 4, 5]
test_qubits = [cirq.GridQubit(0, i) for i in test_indices]
Z_outs = [cirq.Z(i) for i in test_qubits]
test_precision = len(test_indices)
test_op = cirq.PauliSum()
for n, q in enumerate(test_qubits):
    J = cirq.PauliString(-(2 ** (test_precision - n - 1)) / 2 * cirq.Z(q))
    I = cirq.PauliString(2 ** (test_precision - n - 1) / 2 * cirq.I(q))
    test_op += J
    test_op += I
test_operator_pos = test_op + (-1) * (2 ** (test_precision - 1)) * cirq.I(
    test_qubits[-1]
)
test_operator_pos *= np.sqrt(2 * np.pi / (2**test_precision))
expectation = tfq.layers.Expectation()


########
# Tests
########


class TestOps(unittest.TestCase):
    def test_instantiate(self):
        _ = BinaryOp(test_qubits)
        _ = PositionOp(test_qubits)
        _ = MomentumOp(test_qubits)
        passed = True
        self.assertTrue(passed)

    def test_instantiate_error(self):
        with pytest.raises(TypeError):
            _ = BinaryOp([1, "junk"])
        with pytest.raises(TypeError):
            _ = BinaryOp([-1])

        with pytest.raises(TypeError):
            _ = PositionOp([1, "junk"])
        with pytest.raises(TypeError):
            _ = PositionOp([-1])

        with pytest.raises(TypeError):
            _ = MomentumOp([1, "junk"])
        with pytest.raises(TypeError):
            _ = MomentumOp([-1])
        passed = True
        self.assertTrue(passed)

    def test_properties(self):
        binary_register = BinaryOp(test_qubits)
        self.assertEqual(binary_register.op, test_op)

        position_op = PositionOp(test_qubits)
        self.assertEqual(position_op.qubits, test_qubits)
        self.assertEqual(position_op.op, test_operator_pos)
        self.assertEqual(position_op.precision, test_precision)

        momentum_op = MomentumOp(test_qubits)
        self.assertEqual(momentum_op.qubits, test_qubits)
        self.assertEqual(momentum_op.op, test_operator_pos)
        self.assertEqual(momentum_op.precision, test_precision)

    def test_eq(self):
        self.assertEqual(BinaryOp(test_qubits), BinaryOp(test_qubits))
        self.assertEqual(PositionOp(test_qubits), PositionOp(test_qubits))
        self.assertEqual(MomentumOp(test_qubits), MomentumOp(test_qubits))

    def test_binary_register_expectation_basic(self):
        # Should have eigenvalues from 0 through (2^N)-1
        binary_register = BinaryOp(test_qubits)
        readout_ops = binary_register.op

        # For n qubits, test each of the 2**n available values
        for n in range(2 ** len(test_qubits)):
            l_test = ComputationalLayerInteger(n, test_qubits)
            exp = expectation([l_test], operators=[readout_ops]).numpy()[0]
            self.assertAlmostEqual(exp, n, delta=1e-2)

    def test_position_expectation_basic(self):
        phi = PositionOp(test_qubits)
        readout_ops = phi.op
        factor = np.sqrt(2 * np.pi / (2 ** len(test_qubits)))

        # For n qubits, test each of the 2**n available values
        for n in range(2 ** len(test_qubits)):
            l_test = ComputationalLayerInteger(n, test_qubits)
            exp = expectation([l_test], operators=[readout_ops]).numpy()[0]
            this_eigval = factor * (n - (2 ** (len(test_qubits) - 1)))
            self.assertAlmostEqual(exp, this_eigval, delta=1e-2)

    def test_position_expectation(self):
        # Expectation value of Phi w.r.t. GHZ state should yield (domain[0] + domain[1]-(domain[1]-domain[0])/2**N)/2
        test_registers = [1, 2, 5]
        register_qubits = [cirq.GridQubit(0, i) for i in test_registers]
        l_x = cirq.Circuit([cirq.H(i) for i in register_qubits])
        l_0 = cirq.Circuit([cirq.I(i) for i in register_qubits])
        phi = PositionOp(register_qubits)
        readout_ops = phi.op
        measurement_x_basis = expectation([l_x], operators=[readout_ops]).numpy()[0]
        measurement_0_basis = expectation([l_0], operators=[readout_ops]).numpy()[0]
        factor = np.sqrt(np.pi / (2 ** (len(register_qubits) - 1)))
        mean_val = -factor / 2
        self.assertAlmostEqual(measurement_x_basis, mean_val, delta=1e-2)
        self.assertAlmostEqual(
            measurement_0_basis,
            (-1) * (2 ** (len(register_qubits) - 1)) * factor,
            delta=1e-2,
        )
