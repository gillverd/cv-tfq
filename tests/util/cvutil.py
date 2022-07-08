import numpy as np
import matplotlib.pyplot as plt


def domain_bin(v, precision, domain=None, lendian=False):
    """
    passed a float value and domain for discretization, convert it to a string
    that represents the binary representation of the value on the given domain
    with an implicit '.' before the MSB
    example usage:
        domain_bin(3.6, [3,4], 3) = '101' since .101 => 5/8 = .625 is the nearest
        decimal representation of 3.6 accurate to 3 bits of precision
        domain_bin(5.2, [3,4], 3) =
    args:
        v (float): float to convert to domain-specific binary
        domain (list): bounds of the discretized continuous variable [x_min,x_max)
        precision (int): number of bits of precision for output string
        lendian (bool): dictates if the result is in the 'little-endian' format
    return:
        padded_bin (string): string representation of the state representing v
        discretized on the given domain/precision scheme
    """
    if domain == None:
        domain = [
            -np.sqrt(2 * np.pi * 2 ** precision) / 2,
            np.sqrt(2 * np.pi * 2 ** precision) / 2,
        ]

    a, b = domain
    # no wrapping for now; just give a value in the range
    # convert this float to its decimal representation on the interval
    base = 1 / 2 ** precision
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


def plot_wfs(sim_wfs):

    d = np.size(sim_wfs[0])
    fig, (ax_re, ax_im, ax_norm) = plt.subplots(1, 3, sharey=True, figsize=(15, 30))
    for wf in sim_wfs:
        ax_re.plot(range(d), [v.real for v in np.nditer(wf.T)])
        ax_im.plot(range(d), [v.imag for v in np.nditer(wf.T)])
        ax_norm.plot(range(d), [np.abs(v) for v in np.nditer(wf.T)])

    ax_re.set_title("Re(psi)")
    ax_im.set_title("Im(psi)")
    ax_norm.set_title("norm(psi)")
    plt.legend(["kick %i" % k for k in range(d)])
    plt.show()
