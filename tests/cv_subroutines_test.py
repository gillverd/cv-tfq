import sys
sys.path.append("..")

import math

import cirq
import numpy as np

from cv_ops import PositionOp, MomentumOp
from cv_subroutines import ComputationalLayerBinary, ComputationalLayerInteger, discrete_continuous
from cv_subroutines import QFT, centeredQFT, kick_position, kick_momentum
from tests.util.cvutil import domain_bin, plot_wfs
from tests.util.util import prep_state_binary, prep_state_integer

def init_phi_circuit(x, qubits, domain=None):
    # initialize a CV state of a definite position |phi=x>
    x_str = domain_bin(x, len(qubits), lendian=False)
    init_circuit = ComputationalLayerBinary(x_str, qubits)
    return init_circuit

def init_pi_circuit(p, qubits, domain=None):
    # initialize a CV state of a definite momentum |pi=p>
    p_str = domain_bin(p, len(qubits), lendian=False)
    init_circuit = ComputationalLayerBinary(p_str, qubits)
    init_circuit += centeredQFT(qubits)
    return init_circuit

def visualize_kick_momentum():
    qubits = cirq.LineQubit.range(3)

    circuit = cirq.Circuit([cirq.H(q) for q in qubits])
    circuit += centeredQFT(qubits)
    init_state = cirq.Simulator().simulate(circuit).final_state_vector

    circuit += centeredQFT(qubits, inverse=True)
    circuit += kick_momentum(qubits, 1)
    circuit += centeredQFT(qubits)
    final_state = cirq.Simulator().simulate(circuit).final_state_vector

    plot_wfs([init_state, final_state])

def test_verify_centered_fourier():
    """
    check that long and short circuit versions of Fc agree
    """

    def long_centeredQFT(qubits, inverse=False):
        """
        deprecated version of the centered Fourier transform
        """
        N = 2 ** len(qubits)
        circuit = kick_position(qubits, N / 2)
        circuit += QFT(qubits, inverse=inverse)
        circuit += kick_position(qubits, N / 2)
        return circuit

    for n in [2, 5, 6]:
        N = 2 ** n

        # set up operators
        register = [cirq.GridQubit(0, i) for i in range(n)]
        phi = PositionOp(register)

        # test QFT_c on a variety of initial states
        for j in range(N):
            # perform initialization and two cenetered QFT's

            # current implementation of centeredQFT
            F1 = ComputationalLayerInteger(j, register)
            F1 += centeredQFT(register)
            phi1 = cirq.Simulator().simulate(F1).final_state_vector

            # deprecated version of ceneteredQFT
            F2 = ComputationalLayerInteger(j, register)
            F2 += long_centeredQFT(register)
            phi2 = cirq.Simulator().simulate(F2).final_state_vector

            np.testing.assert_array_almost_equal(phi1, phi2)

def test_phi_fourier_consistency(quiet=True):
    """
    insist that Fc*Fc*|x> = |-x>
    """
    
    for n in [2, 5, 6]:
        N = 2 ** n
        # domain limits
        a, b = -np.sqrt(2 * np.pi * N) / 2, np.sqrt(2 * np.pi * N) / 2
        tau = np.sqrt(2 * np.pi / N)
        
        # set up operators
        register = [cirq.GridQubit(0, i) for i in range(n)]
        phi = PositionOp(register)
        
        for x_index in range(N):
            
            # choose one of the valid grid positions for the control register
            x = a + tau * x_index # true position in register A
            j = domain_bin(x, n) # str indical position representation of phi
            j_int = int(j, base=2)
            
            # perform initialization and two centered QFT's
            phiA_prep = ComputationalLayerBinary(j, register)
            phi0 = cirq.Simulator().simulate(phiA_prep).final_state_vector
            
            F1 = phiA_prep + centeredQFT(register)
            phi1 = cirq.Simulator().simulate(F1).final_state_vector
            
            F2 = F1 + centeredQFT(register)
            phi2 = cirq.Simulator().simulate(F2).final_state_vector
            
            # calculate the expected_state, |-x>
            x_ana = -1 * x
            k_ana = domain_bin(x_ana, n)
            phi_ana = prep_state_binary(k_ana)
            
            def quiet_print(*s):
                if not quiet:
                    print(*s)
                    
            # compare the expected, kicked result to actual
            quiet_print("domain=[%3.2f, %3.2f)" % (a, b))
            quiet_print("INITIAL_STATE: |x=%3.2f> = |%s>" % (x, j))
            quiet_print("EXPECTED FINAL STATE: |x=%3.2f> = |%s>" % (x_ana, k_ana))
            quiet_print("index    simu           expect")
            for k, (v2, v3) in enumerate(zip(phi2, phi_ana)):
                if math.isclose(v2.real, 0, abs_tol=0.1) and math.isclose(v3.real, 0, abs_tol=0.1):
                    continue
                quiet_print("  %2i    " % k,
                            "%4.2f+%4.2f   " % (v2.real, v2.imag),
                            "  %i   " % v3,
                )
                np.testing.assert_almost_equal(v2, v3, decimal=2)

def test_phi_eigenequation(quiet=True):
    """
    test that exp(-i*phi)|x> = exp(-i*x)|x>
    """

    def quiet_print(*s):
        if not quiet:
            print(*s)

    for precision in [2, 5]:
        N = float(2 ** precision)
        phi_register = [cirq.GridQubit(0, i) for i in range(precision)]
        # default domain: [-d/2, d/2] for both spaces
        phi = PositionOp(phi_register)
        tau = np.sqrt(2 * np.pi / N)
        a, b = [-np.sqrt(2 * np.pi * N) / 2, np.sqrt(2 * np.pi * N) / 2]

        quiet_print("index    simu                   expect")
        # prepare a definite position state |ph=x>
        for k in range(2 ** precision):
            x = np.sqrt(2 * np.pi / N) * (k - N / 2)

            init_phi = init_phi_circuit(x, phi_register)
            phi0 = cirq.Simulator().simulate(init_phi).final_state_vector

            # compute the expected outcome analytically
            x_ket_int = int(domain_bin(x, precision), base=2)
            x_ket = prep_state_integer(x_ket_int, precision)

            # we expect the state to be phased by exp(i*x)
            x_ket_ana = np.exp(1j * x) * np.array(x_ket)

            # do the eigenvalue equation sending |phi=x> -> exp(ix)|p=x>
            eigen_layer = init_phi + discrete_continuous(-1, [phi])
            phi1 = cirq.Simulator().simulate(eigen_layer).final_state_vector

            # compare results
            for d, (k2, k3) in enumerate(zip(phi1, x_ket_ana)):
                if k2 == 0 and k3 == 0:
                    continue
                quiet_print("    %i  " % d + "  |  ".join(
                    ["%6.5f + %6.5f j" % (v.real, v.imag) for v in [k2, k3]]
                    )
                )
                np.testing.assert_almost_equal(k2, k3, decimal=2)

def test_phi_clock(quiet=True):
    """
    Exponential of discretized momentum should push a state around like a clock:
    exp(-i*sqrt(2*pi/N)*s*Pi) |j> = (-1)^s |j-s>
    """
    for n in [2, 3]:
        N = 2 ** n
        # Create a Pi operator for kicking
        test_pi = MomentumOp([cirq.GridQubit(0, i) for i in range(n)])

        # Make a circuit for each momentum kick
        kick_angle = np.sqrt((2 * np.pi) / N)
        circuit_collect = [cirq.Circuit()]
        sim_wfs = []
        for circuit_n in range(N):
            new_circuit = circuit_collect[circuit_n] + discrete_continuous(kick_angle, [test_pi])
            circuit_collect.append(new_circuit)
            sim_wfs.append(cirq.Simulator().simulate(new_circuit).final_state_vector)

        ##### This was NOT what was in original, check this
        expected_wfs = [prep_state_integer((N - i - 1) % N, n) for i in range(N)]

        def quiet_print(*s):
            if not quiet:
                print(*s)

        # compare the expected, kicked result to actual
        for i_kick, (psi1, psi1_ana) in enumerate(zip(sim_wfs, expected_wfs)):
            quiet_print("KICK NUMBER %i" % i_kick)
            quiet_print("index    simu           expect")

            for k, (v2, v3) in enumerate(zip(psi1, psi1_ana)):
                # dense coding
                if math.isclose(v2, 0, abs_tol=0.1) and math.isclose(v3, 0, abs_tol=0.1):
                    continue
                quiet_print("  %2i    " % k,
                            "%7.4f+%7.4f   " % (v2.real, v2.imag),
                            "  %i   " % v3,
                            )
                np.testing.assert_almost_equal(v2, v3, decimal=2)

def test_phi_adder(quiet=True):
    """
    set up control and target, time evolve with phi on control and pi on target. If initial position was |a>|b>, final
    position should be |a>|a+b>
    """
    for n in [2, 3, 4]:
        N = 2 ** n

        # domain limits
        a, b = -np.sqrt(2 * np.pi * N) / 2, np.sqrt(2 * np.pi * N) /2

        control_register = [cirq.GridQubit(0, i) for i in range(n)]
        target_register = [cirq.GridQubit(0, i) for i in range(n, 2*n)]
        all_qubits = control_register + target_register

        tau = np.sqrt(2 * np.pi / N)

        # set up operators
        phi_control = PositionOp(control_register)
        pi_target = MomentumOp(target_register)

        s = 1 # kick angle will 'offset-add' xA+xB

        for xA_index in range(N):
            # choose one of the valid grid positions for the control register
            x_A = a + tau * xA_index # true position in register A
            j = domain_bin(x_A, n) # str indical position representation of phi
            j_int = int(j, base=2)
            for xB_index in range(N):

                x_B = a + tau * xB_index # true initial position in register B
                k = domain_bin(x_B, n) # str indicial momentum representation of pi
                k_int = int(k, base=2)

                # index representation of the final wf on target register
                phi_f_ind = int(k_int + s * (j_int - N / 2)) % N
                padded_k_f = "0" * (n - len(bin(phi_f_ind)[2:])) + bin(phi_f_ind)[2:]
                phi_f_ana = j + padded_k_f

                # proper padding for binary string state prep on two registers
                j_padded = j + "0" * len(target_register)
                k_padded = "0" * len(control_register) + k
                phiA_prep = ComputationalLayerBinary(j_padded, all_qubits)
                phiB_prep = phiA_prep + ComputationalLayerBinary(k_padded, all_qubits)

                phi_pi_operator = phiB_prep + discrete_continuous(s, [phi_control, pi_target])
                # observe the outcome at the target register
                phiB_i = cirq.Simulator().simulate(phiB_prep).final_state_vector
                phiB_f_sim = cirq.Simulator().simulate(phi_pi_operator).final_state_vector
                phiB_f_ana = prep_state_binary(phi_f_ana)

                def quiet_print(*s):
                    if not quiet:
                        print(*s)

                # compare the expected, kicked result to actual
                quiet_print("domain=[%3.2f, %3.2f)" % (a, b))
                quiet_print("INITIAL STATE: |xA=%3.2f>|xB=%3.2f>  = |%s>|%s>" % (x_A, x_B, j, k))
                quiet_print("EXPECTED FINAL STATE: |%s>|%s>"
                    % (phi_f_ana[0:n], phi_f_ana[n : 2 * n])
                )
                quiet_print("index    simu           expect")
                for k, (v2, v3) in enumerate(zip(phiB_f_sim, phiB_f_ana)):
                    if math.isclose(v2, 0, abs_tol=0.1) and math.isclose(v3, 0, abs_tol=0.1):
                        continue
                    quiet_print("  %2i    " % k,
                                "%4.2f+%4.2f   " % (v2.real, v2.imag),
                                "  %i   " % v3,
                    )
                    np.testing.assert_almost_equal(v2, v3, decimal=2)

def test_phi_adder_stepwise(quiet=True):
    """
    Exponential of discretized momentum should push a state around like a clock:
    exp(-i*sqrt(2*pi/N)*s*Pi) |j> = (-1)^s |j-s>
    """
    n = 2
    N = 2 ** n
    # domain limits
    a, b = -np.sqrt(2 * np.pi * N) / 2, np.sqrt(2 * np.pi * N) / 2

    control_register = [cirq.GridQubit(0, i) for i in range(n)]
    target_register = [cirq.GridQubit(0, i) for i in range(n, 2 * n)]
    all_qubits = control_register + target_register
    tau = np.sqrt(2 * np.pi / N)

    # set up operators
    phi_control = PositionOp(control_register)
    pi_target = MomentumOp(target_register)
    s = 1 # kick angle will 'offset-add' xA+xB

    # for every x-value in the first register, kick the second register by
    # that x-value computed directly!
    for xA_index in range(N):
        # choose one of the valid grid position for the control register
        x_A = a + tau + xA_index # true position in register A
        j = domain_bin(x_A, n) # str idicial position representation of phi
        j_int = int(j, base=2)
        for xB_index in range(N):

            x_B = a + tau * xB_index # true initial position in register B
            k = domain_bin(x_B, n) # str indicial momentum representation of pi
            k_int = int(k, base=2)

            # index representation of the final wf on target register
            phi_f_ind = int(k_int + s * (j_int - N / 2)) % N
            padded_k_f = "0" * (n - len(bin(phi_f_ind)[2:])) + bin(phi_f_ind)[2:]
            phi_f_ana = j + padded_k_f

            # proper padding for binary string state prep on two registers
            j_padded = j + "0" * len(target_register)
            k_padded = "0" * len(control_register) + k
            phiA_prep = ComputationalLayerBinary(j_padded, all_qubits)
            phiB_prep = phiA_prep + ComputationalLayerBinary(k_padded, all_qubits)

            # kick by the true position
            x_A_pi_B = phiB_prep + discrete_continuous(x_A, [pi_target])

            # observe the outcome at the target register
            phiB_i = cirq.Simulator().simulate(phiB_prep).final_state_vector
            phiB_f_sim = cirq.Simulator().simulate(x_A_pi_B).final_state_vector
            
            phiB_f_ana = prep_state_binary(phi_f_ana)

            def quiet_print(*s):
                if not quiet:
                    print(*s)

            # compare the expected, kicked result to actual
            quiet_print("domain=[%3.2f, %3.2f)" % (a, b))
            quiet_print("INITIAL STATE: |xA=%3.2f>|xB=%3.2f>  = |%s>|%s>" % (x_A, x_B, j, k))
            quiet_print("EXPECTED FINAL STATE: |%s>|%s>"
                % (phi_f_ana[0:n], phi_f_ana[n : 2 * n])
            )
            quiet_print("index    simu           expect")
            for k, (v2, v3) in enumerate(zip(phiB_f_sim, phiB_f_ana)):
                if math.isclose(v2, 0, abs_tol=0.1) and math.isclose(v3, 0, abs_tol=0.1):
                    continue
                quiet_print("  %2i    " % k,
                    "%4.2f+%4.2f   " % (v2.real, v2.imag),
                    "  %i   " % v3,
                )
                np.testing.assert_almost_equal(v2, v3, decimal=2)
                                                                
