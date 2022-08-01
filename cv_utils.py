import numpy as np
from tensorflow.linalg import trace

# Adapted from: https://qiskit.org/documentation/_modules/qiskit/quantum_info/states/densitymatrix.html#DensityMatrix
def pure_density_matrix_to_statevector(dm):
    evals, evecs = np.linalg.eig(dm)
    return evecs[:, np.argmax(evals)]

def state_vector_to_density_matrix(sv):
    psi = np.expand_dims(sv, axis=0)
    rho = psi.conj().T @ psi
    return rho

def trace_out(dm, size1, size2, sv=False):
    if sv:
        dm = state_vector_to_density_matrix(dm)
    dm = dm.reshape([size1, size1, size2, size2])
    t = trace(dm)
    return t

def domain_float(bin, domain=None, lendian=False):
    precision = len(bin)
    if domain == None:
        domain = [
            -np.sqrt(2 * np.pi * 2 ** precision) / 2,
            np.sqrt(2 * np.pi * 2 ** precision) / 2,
        ]
    a, b = domain[0], domain[1]
    base = 1 / 2 ** precision

    v = a
    if not lendian:
        bin = bin[::-1]
    for idx, bit in enumerate(bin):
        v += int(bit) * (2**idx) * base * (b - a)

    return v

def domain_float_tf(bins, precision, domain=None):
    if domain == None:
        domain = [
            -tf.math.sqrt(2 * np.pi * 2 ** precision) / 2,
            tf.math.sqrt(2 * np.pi * 2 ** precision) / 2,
        ]
    a, b = domain
    base = 1 / 2 ** precision

    v = tf.fill([bins.shape[0]], a)
    idxs = tf.range(precision, dtype=tf.float32)
    mult = 2**idxs * base * (b - a)
    adder = tf.math.multiply(bins, mult)
    adder = tf.math.reduce_sum(adder, axis=1)
    v += adder
    
    return v

def domain_bin_tf(z, precision, domain=None):
    """
    passed a float value and domain for discretization, convert it to a string
    that represents the binary representation of the value on the given domain
    with an implicit '.' before the MSB
    example usage:
        domain_bin(3.6, [3,4], 3) = '101' since .101 => 5/8 = .625 is the nearest
        decimal representation of 3.6 accurate to 3 bits of precision
        domain_bin(5.2, [3,4], 3) =
    args:
        z (float): float to convert to domain-specific binary
        domain (list): bounds of the discretized continuous variable [x_min,x_max)
        precision (int): number of bits of precision for output string
    return:
        padded_bin (string): string representation of the state representing v
        discretized on the given domain/precision scheme
    """
    if domain == None:
        domain = [
            -tf.math.sqrt(2 * np.pi * 2 ** precision) / 2,
            tf.math.sqrt(2 * np.pi * 2 ** precision) / 2,
        ]

    v = tf.identity(z)

    a, b = domain
    base = 1 / 2 ** precision
    incr = b - a

    def add_incr(v):
        inds = tf.where(v < a)
        return tf.tensor_scatter_nd_add(v, inds, incr * tf.ones(shape=tf.shape(inds)[0], dtype=tf.float32))
    
    def less_a(v):
        return tf.reduce_any(v < a)
    
    v = tf.while_loop(less_a, add_incr, [v], shape_invariants=[v.get_shape()])[0]

    def sub_incr(v):
        inds = tf.where(v >= b)
        return tf.tensor_scatter_nd_add(v, inds, -incr * tf.ones(shape=tf.shape(inds)[0], dtype=tf.float32))
    
    def great_b(v):
        return tf.reduce_any(v >= b)

    v = tf.while_loop(great_b, sub_incr, [v], shape_invariants=[v.get_shape()])[0]

    decimal_val = tf.cast(tf.math.round((v - a) / (base * (b - a))), tf.int32)

    padded_bin = tf.reverse(tf.math.floormod(tf.bitwise.right_shift(tf.expand_dims(decimal_val, 1), tf.range(precision)), 2), axis=[-1])
    
    return padded_bin
