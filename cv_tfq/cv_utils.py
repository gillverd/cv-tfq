"""Contains all utility functions that may be useful for CV operations."""
import numpy as np  # type: ignore[import]
from typing import List, Tuple, Optional
from tensorflow import linalg as tfl  # type: ignore[import]
import tensorflow as tf  # type: ignore[import]
import cirq
from .cv_subroutines import ComputationalLayerInteger
import matplotlib.pyplot as plt  # type: ignore[import]


def plot_wfs(sim_wfs: np.ndarray) -> None:
    """
    Plot the provided wavefunction(s).

    The real, imaginary, and nomalized wavefunctions are plotted each as columnar elements
    in a 1 x 3 subplot.

    Args:
        - sim_wfs (array): the array of wavefunctions to plot

    Returns:
        - None
    """
    d = np.size(sim_wfs[0])
    fig, (ax_re, ax_im, ax_norm) = plt.subplots(1, 3, sharey=True, figsize=(15, 30))
    for wf in sim_wfs:
        ax_re.plot(range(d), [v.real for v in np.nditer(wf.T)])  # type: ignore
        ax_im.plot(range(d), [v.imag for v in np.nditer(wf.T)])  # type: ignore
        ax_norm.plot(range(d), [np.abs(v) for v in np.nditer(wf.T)])

    ax_re.set_title("Re(psi)")
    ax_im.set_title("Im(psi)")
    ax_norm.set_title("norm(psi)")
    plt.legend(["kick %i" % k for k in range(d)])
    plt.show()


def prep_state_integer(j: int, n: int) -> np.ndarray:
    """
    Prepare a computational state |j> in array form.

    Sister state prep layer is ComputationalLayerInteger

    Args:
        - j (int): Integer value of the qubit state to be prepared
        - n (int): Number of qubits in register

    Returns:
        - array form of wavefunction representing |j> on n qubits
    """
    return np.array([0 if i != j else 1 for i in range(2**n)])


def prep_state_binary(s: str) -> np.ndarray:
    """
    Prepare a computational state from a binary string.

    Sister state prep layer is ComputationalLayerBinary

    Args:
        - s (str): binary string representing the state to be prepared

    Returns:
        - array form of wavefunction representing binary string s
    """
    j = int(s, base=2)
    return prep_state_integer(j, len(s))


def prepare_base_state(j: int, n: int) -> Tuple[cirq.Circuit, List[int]]:
    """
    Prepare a basis state |j> on n_qubits.

    Also provides the gateset to generate this state.

    Args:
        - j (int): the state to prepare
        - n (int): the number of qubits

    Returns:
        - (cirq.Circuit): the prepared state in the form of a circuit
        - (list): list containing the statevector for this prepared circuit
    """
    state_prep = ComputationalLayerInteger(j, [cirq.GridQubit(0, i) for i in range(n)])
    state = [0 for i in range(2**n)]
    state[j] = 1

    return state_prep, state


def pure_density_matrix_to_statevector(dm: np.ndarray) -> np.ndarray:
    """
    Convert density matrix to statevector.

    This assumes that the density matrix represents a pure state.
    Adapted from: https://qiskit.org/documentation/_modules/qiskit/quantum_info/states/densitymatrix.html#DensityMatrix

    Args:
        - dm (NDArray[N, N]): the NxN matrix representing the density matrix

    Returns:
        - (NDArray[N]): the array that represents the statevector
    """
    evals, evecs = np.linalg.eig(dm)
    return evecs[:, np.argmax(evals)]


def state_vector_to_density_matrix(sv: np.ndarray) -> np.ndarray:
    """
    Convert statevector to density matrix.

    Args:
        - sv (NDArray[N]): the statevector to be converted

    Returns:
        - (NDArray[N, N]): the associated density matrix
    """
    psi = np.expand_dims(sv, axis=0)
    rho = psi.conj().T @ psi
    return rho


def trace_out(
    dm: np.ndarray, size1: int, size2: int, sv: Optional[bool] = False
) -> np.ndarray:
    """
    Trace out the density matrix into the given sizes.

    size1 * size2 == dm.shape[0] must be true.

    Args:
        - dm (NDArray): the density matrix (or statevector) to be traced out
        - size1 (int): the size of the first matrix traced out (the one that will
            be returned)
        - size2 (int): the size of the second matrix to be traced out (will not be
            returned)
        - sv (optional, bool): whether or not the input was actually a density matrix
            or a statevector (which is then converted to a density matrix)

    Returns:
        - (NDArray[size1, size1]): the traced out matrix
    """
    if sv:
        dm = state_vector_to_density_matrix(dm)
    dm = dm.reshape([size1, size1, size2, size2])
    t = tfl.trace(dm)
    return t


def domain_float(
    bin: List[int],
    domain: Optional[List[float]] = None,
    lendian: Optional[bool] = False,
) -> float:
    """
    Convert discretized value to be converted back into a float.

    Args:
        - bin (list): the digital representation of the continuous variable
        - domain (optional, list): the upper and lower bounds on the domain for
            the representation
        - lendian (optional, bool): whether this representation is in big or
            little endian

    Returns:
        - (float): the converted floating point value
    """
    precision = len(bin)
    if domain == None or not isinstance(domain, list):
        domain = [
            -np.sqrt(2 * np.pi * 2**precision) / 2,
            np.sqrt(2 * np.pi * 2**precision) / 2,
        ]
    a, b = domain[0], domain[1]
    base = 1 / 2**precision

    v = a
    if not lendian:
        bin = bin[::-1]
    for idx, bit in enumerate(bin):
        v += int(bit) * (2**idx) * base * (b - a)

    return v


def domain_float_tf(
    bins: tf.Tensor, precision: int, domain: Optional[List[float]] = None
) -> float:
    """
    Convert discretized value to be converted back into a float.

    Compatible with @tf.function decorators.

    Args:
        - bin (list): the digital representation of the continuous variable
        - precision (int): len(bin)
        - domain (optional, list): the upper and lower bounds on the domain for
            the representation

    Returns:
        - (float): the converted floating point value
    """
    if domain == None or not isinstance(domain, list):
        domain = [
            -tf.math.sqrt(2 * np.pi * 2**precision) / 2,
            tf.math.sqrt(2 * np.pi * 2**precision) / 2,
        ]
    a, b = domain
    base = 1 / 2**precision

    v = tf.fill([bins.shape[0]], a)
    idxs = tf.range(precision, dtype=tf.float32)
    mult = 2**idxs * base * (b - a)
    adder = tf.math.multiply(bins, mult)
    adder = tf.math.reduce_sum(adder, axis=1)
    v += adder

    return v


def domain_bin(
    v: float,
    precision: int,
    domain: Optional[List[float]] = None,
    lendian: Optional[bool] = False,
) -> str:
    """
    Pass a float value and domain for discretization.

    Convert it to a string that represents the binary representation of the value
    on the given domain with an implicit '.' before the MSB

    example usage:
        domain_bin(3.6, [3,4], 3) = '101' since .101 => 5/8 = .625 is the nearest
        decimal representation of 3.6 accurate to 3 bits of precision

    Args:
        - v (float): float to convert to domain-specific binary
        - precision (int): number of bits of precision for output string
        - domain (list, optional): bounds of the discretized continuous variable [x_min,x_max)
        - lendian (bool, optional): dictates if the result is in the 'little-endian' format

    Returns:
        - (string): string representation of the state representing v
        discretized on the given domain/precision scheme
    """
    if domain == None or not isinstance(domain, list):
        domain = [
            -np.sqrt(2 * np.pi * 2**precision) / 2,
            np.sqrt(2 * np.pi * 2**precision) / 2,
        ]

    a, b = domain
    # no wrapping for now; just give a value in the range
    # convert this float to its decimal representation on the interval
    base = 1 / 2**precision
    # implementing boundary conditions
    while v < a:
        v += b - a
    while v >= b:
        v -= b - a
    if v < a or v >= b:
        print("float %6.4f to be converted is not in the domain provided" % v)
        raise ValueError
    decimal_val = int(round((v - a) / (base * (b - a))))

    # construct and pad the binary version of this, to the specified precision
    unpadded_bin = bin(decimal_val)[2:]
    padded_bin = (
        "".join(["0" for i in range(precision - len(unpadded_bin))]) + unpadded_bin
    )
    if lendian == True:
        padded_bin = padded_bin[::-1]
    if len(padded_bin) > precision:
        print(padded_bin)

        raise ValueError("binary conversion overflow")
    return padded_bin


def domain_bin_tf(
    z: float, precision: int, domain: Optional[List[float]] = None
) -> tf.Tensor:
    """
    Convert float to discretized bin.

    Passed a float value and domain for discretization, convert it to a string
    that represents the binary representation of the value on the given domain
    with an implicit '.' before the MSB

    Compatible with @tf.function decoration.

    Example usage:
        domain_bin(3.6, [3,4], 3) = [1, 0, 1] since .101 => 5/8 = .625 is the nearest
        decimal representation of 3.6 accurate to 3 bits of precision

    Args:
        - z (float): float to convert to domain-specific binary
        - precision (int): number of bits of precision for output string
        - domain (optional, list): bounds of the discretized continuous variable [x_min,x_max)

    Returns:
        - (string): string representation of the state representing v
        discretized on the given domain/precision scheme
    """
    if domain == None or not isinstance(domain, list):
        domain = [
            -tf.math.sqrt(2 * np.pi * 2**precision) / 2,
            tf.math.sqrt(2 * np.pi * 2**precision) / 2,
        ]

    v = tf.identity(z)

    a, b = domain
    base = 1 / 2**precision
    incr = b - a

    def add_incr(v):  # type: ignore
        inds = tf.where(v < a)
        return tf.tensor_scatter_nd_add(
            v, inds, incr * tf.ones(shape=tf.shape(inds)[0], dtype=tf.float32)
        )

    def less_a(v):  # type: ignore
        return tf.reduce_any(v < a)

    v = tf.while_loop(less_a, add_incr, [v], shape_invariants=[v.get_shape()])[0]

    def sub_incr(v):  # type: ignore
        inds = tf.where(v >= b)
        return tf.tensor_scatter_nd_add(
            v, inds, -incr * tf.ones(shape=tf.shape(inds)[0], dtype=tf.float32)
        )

    def great_b(v):  # type: ignore
        return tf.reduce_any(v >= b)

    v = tf.while_loop(great_b, sub_incr, [v], shape_invariants=[v.get_shape()])[0]

    decimal_val = tf.cast(tf.math.round((v - a) / (base * (b - a))), tf.int32)

    # Converts to binary
    padded_bin = tf.reverse(
        tf.math.floormod(
            tf.bitwise.right_shift(tf.expand_dims(decimal_val, 1), tf.range(precision)),
            2,
        ),
        axis=[-1],
    )

    return padded_bin
