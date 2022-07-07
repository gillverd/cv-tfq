import sys
sys.path.append("..")

import numpy as np
from cv_subroutines import ComputationalLayerInteger


def prep_state_integer(j, n):
    """
    prepare a computational state |j> in array form; sister state prep layer is
    ComputationalLayerInteger

    args:
        j (int): Integer value of the qubit state to be prepared
        n (int): Number of qubits in register
    return:
        array form of wavefunction representing |j> on n qubits

    """
    return np.array([0 if i != j else 1 for i in range(2 ** n)])


def prep_state_binary(s):
    """
    prepare a computational state from a binary string; sister state prep layer
    is ComputationalLayerBinary

    args:
        s (:string:): binary string representing the state to be prepared; sister
    return:
        array form of wavefunction representing binary string s
    """
    j = int(s, base=2)
    return prep_state_integer(j, len(s))


def prepare_base_state(j, n):
    # prepare a basis state |j> on n_qubits, and provide the gateset to
    # generate this state
    state_prep = ComputationalLayerInteger(j, range(n))
    state = [0 for i in range(2 ** n)]
    state[j] = 1

    return state_prep, state
