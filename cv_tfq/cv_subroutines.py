"""Contains code for all relevant CV subroutines."""
import cirq
import tensorflow_quantum as tfq  # type: ignore[import]
import numpy as np  # type: ignore[import]
from typing import List, Optional
from .cv_ops import BinaryOp, PositionOp, MomentumOp


def ComputationalLayerBinary(s: str, qubits: List[cirq.Qid]) -> cirq.Circuit:
    """
    Prepare a computational state corresponding to the binary string s.

    Args:
        - s (string): binary string representing the state to be prepared,
            that accounts for all qubits in the circuit
        - qubits (list): list of qubits that contain the bits

    Returns:
        - (circuit): state preparation layer to prepare the state represented
            by binary state 's'

    Raises:
        - ValueError: if the binary string is not the same length as the number
            of qubits
    """
    if len(s) != len(qubits):
        raise ValueError("Binary string must be same length as qubits")

    circuit = cirq.Circuit()
    for ind, c in enumerate(s):
        if c == "1":
            circuit += cirq.X(qubits[ind])
        else:
            circuit += cirq.I(qubits[ind])
    return circuit


def ComputationalLayerInteger(j: int, qubits: List[cirq.Qid]) -> cirq.Circuit:
    """
    Prepare a computational state |j> on the given qubits.

    Args:
        - j (int): Integer representing the decimal value of the qubit state
            to be prepared on the provided qubits
        - qubits (list): ordered list of qubits

    Returns:
        - (circuit): state preparation layer to prepare state |j>

    Raises:
        - ValueError: if j < 0 or if j > 2^(number of qubits)
    """
    if j < 0:
        raise ValueError("Cannot prepare state based on negative value")

    n = len(qubits)
    if j >= 2**n:
        raise ValueError("Cannot prepare state |%i> on %i qubits" % (j, n))

    binstr = bin(j)[2:]
    # pad the resulting binary string
    binstr = "0" * (n - len(binstr)) + binstr
    return ComputationalLayerBinary(binstr, qubits)


def QFT(
    qubits: List[cirq.Qid], swap: Optional[bool] = True, inverse: Optional[bool] = False
) -> cirq.Circuit:
    """
    Calculate the Quantum Fourier Transform (or inverse) with optional swaps.

    Adapted from https://quantumai.google/cirq/experiments/textbook_algorithms#quantum_fourier_transform

    Args:
        - qubits (list): ordered list of qubits
        - swap (optional, bool): whether or not to use swaps at the end of the QFT
        - inverse (optional, bool): whether or not to invert the QFT

    Returns:
        - (circuit): the QFT circuit as defined by the input params
    """

    def qft_base(qubits):  # type: ignore
        qreg = list(qubits)
        while len(qreg) > 0:
            q_head = qreg.pop(0)
            yield cirq.H(q_head)
            for i, qubit in enumerate(qreg):
                yield (cirq.CZ ** (1 / 2 ** (i + 1)))(qubit, q_head)

    def swaps(qubits):  # type: ignore
        start = 0
        end = len(qubits) - 1

        while end - start >= 1:
            yield (cirq.SWAP)(qubits[start], qubits[end])
            start += 1
            end -= 1

    qft_circuit = cirq.Circuit(qft_base(qubits))
    if swap:
        qft_circuit += cirq.Circuit(swaps(qubits))
    if inverse:
        qft_circuit = cirq.inverse(qft_circuit)

    return qft_circuit


def centeredQFT(
    qubits: List[cirq.Qid], swap: Optional[bool] = True, inverse: Optional[bool] = False
) -> cirq.Circuit:
    """
    QFT modified to operate on a Somma-convention position operator.

    The centered Fourier Transform applies phases to elements in a list in a manner that is
    symmetric w/r to the center element of the list instead of the initial element

    Args:
        - qubits (list): ordered list of qubits
        - swap (optional, bool): whether or not to use swaps at the end of the QFT
        - inverse (optional, bool): whether or not to invert the QFT

    Returns:
        - (circuit): the centeredQFT circuit as defined by the input params
    """
    center_qft = cirq.Circuit(cirq.X(qubits[0]))
    center_qft += QFT(qubits, swap=swap, inverse=inverse)
    center_qft += cirq.X(qubits[0])
    return center_qft


def kick_position(qubits: List[cirq.Qid], kicks: float) -> cirq.Circuit:
    """
    Construct a layer to kick a position state by a certain number of steps on a discretized CV lattice.

    If |j> is a computational basis state, this does:
    |j> -> |j + kicks>

    Args:
        - qubits (list): the qubits on which the kick will act
        - kicks (float): number of kicks to give the input state.  Must be an integer for clock kicking.

    Returns:
        - (circuit): op to apply discrete kick to position register
    """
    F = QFT(qubits)
    kick_magnitude = 2 * np.pi * kicks / (2 ** len(qubits))
    op = BinaryOp(qubits).op
    displaced = F + tfq.util.exponential([op], [kick_magnitude])
    Fdag = displaced + QFT(qubits, inverse=True)
    return Fdag


def kick_momentum(qubits: List[cirq.Qid], kicks: float) -> cirq.Circuit:
    """
    Construct a layer to kick a momentum state by a certain number of steps on a discretized CV lattice.

    Args:
        - qubits (list): the qubits on which the kick will act
        - kicks (float): number of kicks to give the input state.  Must be an integer for clock kicking.

    Returns:
        - (circuit): op to apply discrete kick to momentum register
    """
    kick_magnitude = 2 * np.pi * kicks / (2 ** len(qubits))
    return tfq.util.exponential([BinaryOp(qubits).op], [kick_magnitude])


def adder(control: List[cirq.Qid], target: List[cirq.Qid]) -> cirq.Circuit:
    """
    Construct an adder layer on control and target discretized registers.

    |x1>|x2> -> |x1>|x1 + x2>

    Args:
        - control (list): register containing control CV operators
        - target (list): register containing target CV operators

    Returns:
        - (circuit): Layer to add control and target eigenstates
            into the target register
    """
    theta = -1
    phi_control = PositionOp(control)
    pi_target = MomentumOp(target)
    adder_layer = discrete_continuous(theta, [pi_target, phi_control])
    return adder_layer


def subtractor(control: List[cirq.Qid], target: List[cirq.Qid]) -> cirq.Circuit:
    """
    Construct a subtractor for a control and target discretized registers.

    |x1>|x2> -> |x1>|x1 - x2>

    Args:
        - control (list): register containing control CV operators
        - target (list): register containing target CV operators

    Returns:
        - (circuit): Layer to subtract control from target qmode
    """
    # invert the eigenvalue of the control register by FF|x> = |-x>
    Fc1 = centeredQFT(control)
    Fc2 = Fc1 + centeredQFT(control)
    subtractor_layer = Fc2 + adder(control, target)
    return subtractor_layer


def swap(control: List[cirq.Qid], target: List[cirq.Qid]) -> cirq.Circuit:
    """
    Construct a SWAP for two discretized registers.

    |x1>|x2> -> |x2>|x1>

    This layer is constructed from a generalization of the CNOT scheme for
    swapping two qubits

    Args:
        - control (list): register containing control CV operators
        - target (list): register containing target CV operators
    Returns:
        - (circuit): Layer to swap control and target qmodes
    """
    CNOT1 = subtractor(control, target)
    CNOT2 = CNOT1 + subtractor(target, control)
    CNOT3 = CNOT2 + subtractor(control, target)
    return CNOT3


def discrete_continuous(parameter: float, operators: List) -> cirq.Circuit:
    """
    Create a hybrid discrete-continuous layer using position, momentum, and standard operators.

    Args:
        - parameter (float): value to be used as the parameter in the parametric layer
            created using the combined operators of all passed operators
        - operators (list): Operators from which to create the layer. All operators in this
            list are multiplied together to create the parameterized layer;
            therefore it is an error to pass operators which share indices

    Returns:
        - (circuit): the hybrid layer as a circuit
    """
    operator_list = []
    pi_list = []
    phi_list = []
    layer = cirq.Circuit()
    operator_ = None

    # compose a product of the operators
    for operator in operators:
        if isinstance(operator, cirq.PauliSum):
            operator_list.append(operator)
            op = operator
        elif isinstance(operator, PositionOp) and not isinstance(operator, MomentumOp):
            phi_list.append(operator)
            op = operator.op
        elif isinstance(operator, MomentumOp):
            pi_list.append(operator)
            op = operator.op
        else:
            raise TypeError("Invalid operator type passed to discrete_continuous_layer")

        if operator_ is None:
            operator_ = op.copy()
        else:
            operator_ *= op

    for pi in pi_list:
        layer += centeredQFT(pi.qubits, inverse=False)

    parameter = float(parameter)
    layer += tfq.util.exponential([operator_], [parameter])

    for pi in pi_list:
        layer += centeredQFT(pi.qubits, inverse=True)

    return layer


def signum_layer(qmode: List[cirq.Qid]) -> cirq.Circuit:
    """
    Construct a signum layer on a given CV qmode.

    Args:
        - qmode (list): register containing respective discretized operators

    Returns:
        - (circuit): Layer to enact signum function on input qmode
    """
    signum_op = cirq.PauliSum()
    J: cirq.PauliString = cirq.PauliString(-1 / 2 * cirq.Z(qmode[0]))
    I: cirq.PauliString = cirq.PauliString(1 / 2 * cirq.I(qmode[0]))
    signum_op += J
    signum_op += I
    sig_layer = tfq.util.exponential([signum_op], [5.0])
    return sig_layer


def relu_layer(strength: float, qmode: List[cirq.Qid]) -> cirq.Circuit:
    """
    Implement a ReLU on a given CV mode.

    Args:
        - strength (float): the strength of the J and I terms
        - qmode (list): register containing respective discretized operators

    Returns:
        - (circuit): Layer to enact RELU function on input qmode
    """
    signum_op = cirq.PauliSum()
    J: cirq.PauliString = cirq.PauliString(-strength / 2 * cirq.Z(qmode[0]))
    I: cirq.PauliString = cirq.PauliString(strength / 2 * cirq.I(qmode[0]))
    signum_op += J
    signum_op += I
    phi = PositionOp(qmode)
    op = signum_op * phi.op * signum_op
    relu_layer = tfq.util.exponential([signum_op, phi.op, signum_op], [1.0, 1.0, 1.0])
    return relu_layer
