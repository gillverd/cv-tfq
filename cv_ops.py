import cirq
import numpy as np

class BinaryOp(object):
    """
    Create a qubit register for storing a quantum binary number.
    Our operators are mapped to qubits in big-endian format, unlike in Verdon (2018) eqn 10
    Args:
        qubits (list): list of qubits which contain the bits
    """
    def __init__(self, qubits):
        for q in qubits:
            if not isinstance(q, cirq.Qid):
                raise TypeError("All entries of qubits must inherit from cirq.Qid")
        self.qubits = qubits
        self.precision = len(self.qubits)
        op = cirq.PauliSum()
        for n, q in enumerate(self.qubits):
            J = cirq.PauliString(-2**(self.precision - n - 1)/2 * cirq.Z(q))
            I = cirq.PauliString(2**(self.precision - n - 1)/2 * cirq.I(q))
            op += J
            op += I
        self.op = op

    def __str__(self):
        return str(self.op)

    def __eq__(self, o):
        return (self.__class__ == o.__class__) and (self.op == o.op)


class PositionOp(object):
    """
    Position operator
    Convention from https://arxiv.org/abs/1503.06319 (author R. D. Somma)
    Note that our operators are mapped to qubits in big-endian format (MSB at left-most position)
    Creates a position operator
        Args:
            qubits (list): indices of the qubits that the discretized CV is stored on 
    """
    def __init__(self, qubits):
        for q in qubits:
            if not isinstance(q, cirq.Qid):
                raise TypeError("All entries of qubits must inherit from cirq.Qid")
        self.qubits = qubits
        self.precision = len(self.qubits)
        op = cirq.PauliSum()
        for n, q in enumerate(self.qubits):
            op += cirq.PauliString(-2 ** (self.precision - n - 1), cirq.Z(q))
        op -= cirq.PauliString(cirq.I(qubits[-1]))
        op *= np.sqrt(2 * np.pi / (2 ** self.precision)) / 2

        self.op = op
        
    def __str__(self):
        return str(self.op)

    def __eq__(self, o):
        return (self.__class__ == o.__class__) and (self.op == o.op)

    def __add__(self, o):
        if isinstance(o, self.__class__):
            new_cv = self.__class__(sorted(set(self.qubits + o.qubits)))
            new_cv.op = self.op + o.op
            return new_cv
        elif isinstance(o, int) or isinstance(o, float):
            new_cv = self.__class__(self.qubits)
            new_cv.op = self.op
            for q in new_cv.qubits:
                new_cv.op += cirq.PauliString(o, cirq.I(q))
            return new_cv
        else:
            raise TypeError(
                "Adding is only defined between instances of the same CV class \
            or between a CV class instance and a number"
            )
    
    def __radd__(self, o):
        return self.__add__(o)
    
    def __sub__(self, o):
        return self.__add__(-1*o)
    
    def __rsub__(self, o):
        if isinstance(o, int) or isinstance(o, float):
            new_cv = self.__class__(self.qubits)
            new_cv.op = -1*self.op
            return new_cv.__add__(o)

    def __mul__(self, o):
        if isinstance(o, self.__class__):
            new_cv = self.__class__(sorted(set(self.qubits + o.qubits)))
            new_cv.op = self.op * o.op
            return new_cv
        elif isinstance(o, int) or isinstance(o, float):
            new_cv = self.__class__(self.qubits)
            new_cv.op = self.op * o
            return new_cv
        else:
            raise TypeError(
                "Multiplication is only defined between instances of the same CV class \
             or between a CV class instance and a number"
            )

    def __rmul__(self, o):
        if isinstance(o, int) or isinstance(o, float):
            return self.__mul__(o)

    def __truediv__(self, o):
        if not (isinstance(o, int) or isinstance(o, float)):
            raise TypeError(
                "Division is only defined between a CV class and a number"
            )
        if o == 0:
            return ValueError(
                "Division by 0 Error"
            )
        return self.__mul__(1.0 / o)

    def __pow__(self, power):
        new_cv = self.__class__(self.qubits)
        new_cv.op **= power
        return new_cv

class MomentumOp(PositionOp):
    """
    Momentum operator
    Convention from https://arxiv.org/abs/1503.06319 (author R. D. Somma)
    Note that our operators are mapped to qubits in big-endian format (MSB at left-most position)
    
    Creates a momentum operator
    Internally, this operator is the same as the position operator;
    it is a different class so that it will be padded with the proper
    centered Fourier transforms whenever it is exponentiated

    Args:
        qubits (list): qubits that the discretized CV is stored on [qubit0,qubit1,qubit2,...,qubitn]
    """
    def __init__(self, qubits):
        super().__init__(qubits)
