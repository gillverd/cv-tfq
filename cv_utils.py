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