import sys
sys.path.append("../cv-tfq")

import cirq

from cv_ops import BinaryOp, PositionOp, MomentumOp
from cv_subroutines import ComputationalLayerBinary, ComputationalLayerInteger, QFT, centeredQFT, kick_position, kick_momentum, discrete_continuous

def H_mix(qubits):
    return 1/2 * MomentumOp(qubits).op * MomentumOp(qubits).op

def H_cost_poly(qubits, poly):
    H_c = cirq.Circuit()
    for q in qubits:
        H_c += cirq.PauliString(poly[0]*cirq.I(q))
    for idx, p in enumerate(poly[1:]):
        H_c = p * PositionOp(qubits).op
        for _ in range(idx):
            H_c *= PositionOp(qubits).op

    return H_c
    
