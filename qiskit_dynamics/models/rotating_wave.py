##COPYRIGHT STUFF

"""Functions for performing the Rotating Wave Approximation
on Model classes."""


from typing import Union
import numpy as np
from qiskit_dynamics.models import GeneratorModel, HamiltonianModel, LindbladModel, Frame
from qiskit_dynamics.signals import SignalSum, Signal, SignalList
from qiskit_dynamics.dispatch import Array


def _get_new_operators(
    current_ops: Array,
    current_sigs: SignalList,
    frame: Frame,
    frame_freqs: Array,
    cutoff_freq: Union[float, int],
):
    r"""Given a set of operators as a (k,n,n) Array, a set of
    frequencies (frame_freqs)_{jk} = Im[-d_j+d_k] where d_i
    the i^{th} eigenlvalue of the frame operator F, the current
    signals of a model, and a cutoff frequency, returns
    the new operators and signals that should be passed to
    create a new Model class after the RWA.

    Args:
        current_ops: the current operator list, (k,n,n) Array
        current_sigs: (k) length SignalList
        frame: current Frame object of the pre-RWA model
        frame_freqs: the effective frequencies of different
            matrix elements due to the conjugation by e^{\pm Ft}
            in the rotating frame.
        cutoff_freq: maximum frequency allowed under the RWA.
    Returns:
        tuple[SignalList,Array] Tuple of Signal objects (post RWA)
        and (2k,n,n) Array of new operators post RWA.
    Raises:
        NotImplementedError: if components s_j(t) are not equivalent
        to pure Signal objects.
    """
    num_components = len(current_sigs)
    n = current_ops.shape[-1]

    frame_freqs = np.broadcast_to(frame_freqs, (num_components, n, n))

    carrier_freqs = []
    normal_signals = []
    abnormal_signals = []
    for sig_sum in current_sigs.components:
        if len(sig_sum.components) > 1:
            raise NotImplementedError(
                "RWA with coefficients s_j are not pure Signal objects is not currently supported."
            )
        sig = sig_sum.components[0]
        carrier_freqs.append(sig.carrier_freq)
        normal_signals.append(sig)
        abnormal_signals.append(
            SignalSum(Signal(sig.envelope, sig.carrier_freq, sig.phase - np.pi / 2))
        )
    carrier_freqs = np.array(carrier_freqs).reshape((num_components, 1, 1))

    pos_pass = np.abs(carrier_freqs + frame_freqs) < cutoff_freq
    neg_pass = np.abs(-carrier_freqs + frame_freqs) < cutoff_freq

    both_terms = current_ops * (pos_pass & neg_pass).astype(int)
    pos_terms = current_ops * (pos_pass & np.logical_not(neg_pass)).astype(int)
    neg_terms = current_ops * (np.logical_not(pos_pass) & neg_pass).astype(int)

    normal_operators = both_terms + pos_terms / 2 + neg_terms / 2
    abnormal_operators = 1j * pos_terms / 2 - 1j * neg_terms / 2

    new_signals = SignalList(normal_signals + abnormal_signals)
    new_operators = frame.operator_out_of_frame_basis(
        np.append(normal_operators, abnormal_operators, axis=0)
    )

    return new_signals, new_operators


def perform_rotating_wave_approximation(
    model: Union[GeneratorModel, HamiltonianModel], cutoff_freq: Union[float, int]
) -> Union[GeneratorModel, HamiltonianModel]:
    r"""Performs RWA on a GeneratorModel so that all terms that
    rotate with complex frequency larger than cutoff_freq are
    discarded. In particular, let G(t) = \sum_j s_j(t) G_j + G_d,
    and let us consider it in the rotating frame, so that the
    actual operator that is applied is given by e^{-tF}(G(t) - F)e^{tF}
    = \sum_js_j(t)e^{-tF}G_je^{tF} + e^{-tF}(G_d - F)e^{tF}
    Args:
        model: The GeneratorModel to which you
        wish to apply the RWA
        cutoff_freq: the maximum (magnitude) of
        frequency you wish to allow.
    Returns:
        GeneratorModel with twice as many terms
        and some signals with negative frequencies
    Raises:
        NotImplementedError: if components s_j(t) are not equivalent
        to pure Signal objects.

    Formalism: When we consider e^{-tF}A e^{tF} in the basis in which F
    is diagonal, we may write conjugation by e^{\pm tF} as elementwise
    multiplication s.t. (e^{-tF} A e^{tF})_{jk} = e^{(-d_j+d_k)t}*A_{jk}.
    When we write e^{-tF}(G(t)-F)e^{tF} = \sum_i e^{-tF} s_i(t)G_ie^{tF}
    + e^{-tF}(G_d-F)e^{tF} in order to take the RWA in the rotating frame,
    we must consider the effect of rotations caused by the frame, with
    frequency Im(-d_j-d_k)/(2*pi), as well as those due to the signal's
    rotation itself, where we write s_i(t) = Re[a_i(t)e^{i(2*pi*nu_i*t+phi_i)}]
    = a_i(t)e^{i(2*pi*nu_i*t-phi_i)}/2 + \bar{a_i(t)}e^{-i(2*pi*nu_i*t+phi_i)}/2. With this
    in mind, consider a term of the form (e^{-tF}s_i(t)G_ie^{tF})_{jk}
    = e^{(-d_j+d_k)t}(G_i)_{jk}[a_i(t)e^{i(2*pi*nu_i*t+phi_i)}
    +\bar{a_i(t)}e^{-i(2*pi*nu_i*t+phi_i)}]. The \pm i(...) term has
    effective frequency \pm nu_i + Im(-d_j+d_k)/(2*pi) = \pm nu_i + f_{jk},
    (we neglect any oscillation/rotation in a_i for now) and we wish to neglect
    all terms with effective frequency greater in magnitude than some cutoff
    frequency nu_*. As such, let us write G_i = A_i + B_i + C_i + D_i, with
    A_i the matrix representing terms/elements which have
    1) abs(nu_i+f_{jk})<nu_* and 2) abs(-nu_i+f_{jk})<nu_*, B_i the matrix
    of terms where 1) but not 2), C_i the matrix of terms where 2) but not 1),
    and D_i the terms where neither 1) nor 2) hold. Thus, after the RWA,
    we may write our term as s_i(t) G_i -> a_i(t)e^{i(2*pi*nu_i*t+phi_i)}(A_i+B_i)/2
    + \bar{a_i(t)}e^{-i(2*pi*nu_i*t+phi_i)}(A_i+C_i)/2 = s_i(t) A_i +
    [Re[a_i]cos(2*pi*nu_i*t+phi_i)-Im[a_i]sin(2*pi*nu_i*t+phi_i)]B_i/2 +
    [Im[a_i]cos(2*pi*nu_i*t+phi_i)+Re[a_i]sin(2*pi*nu_i*t+phi_i)]iB_i/2 +
    [Re[a_i]cos(2*pi*nu_i*t+phi_i)-Im[a_i]sin(2*pi*nu_i*t+phi_i)]C_i/2 +
    [Im[a_i]cos(2*pi*nu_i*t+phi_i)-Re[a_i]sin(2*pi*nu_i*t+phi_i)](-iC_i/2)
    = s_i(t)A_i + Re[a_i(t)e^{i(2*pi*nu_i*t+phi_i)}](B_i+C_i)/2 +
    Re[a_i(t)e^{i(2*pi*nu_i*t+phi_i-pi/2)}](iB_i-iC_i)/2 =
    s_i(t)(A_i + B_i/2 + C_i/2) + s'_i(t)(iB_i-iC_i)/2 where s'_i
    is a Signal with frequency nu_i, amplitude a_i, and phase phi_i - pi/2.

    Next, note that the drift terms (G_d)_{jk} have effective frequency
    Im[-d_j+d_k]/(2*pi). Note that this vanishes on the diagonal,
    so the -F term may be ignored for any nonzero cutoff_freq.

    """

    ##TODO Make function idempotent

    if model.frame is None or model.frame.frame_diag is None:
        return model

    n = model.dim
    diag = model.frame.frame_diag

    diff_matrix = np.broadcast_to(diag, (n, n)) - np.broadcast_to(diag, (n, n)).T
    frame_freqs = diff_matrix.imag / (2 * np.pi)

    curr_drift = model.frame.operator_out_of_frame_basis(
        model.drift + np.diag(model.frame.frame_diag)
    )
    new_drift = curr_drift * (abs(frame_freqs) < cutoff_freq).astype(int)

    num_components = len(model.signals)
    frame_freqs = np.broadcast_to(frame_freqs, (num_components, n, n))

    if isinstance(model, GeneratorModel):
        new_signals, new_operators = _get_new_operators(
            model.operators, model.signals, model.frame, frame_freqs, cutoff_freq
        )
        if isinstance(model, HamiltonianModel):
            new_model = HamiltonianModel(
                new_operators,
                drift=new_drift,
                signals=new_signals,
                frame=model.frame.frame_operator,
                evaluation_mode=model.evaluation_mode,
            )
        else:
            new_model = GeneratorModel(
                new_operators,
                drift=new_drift,
                signals=new_signals,
                frame=(model.frame.frame_operator),
                evaluation_mode=model.evaluation_mode,
            )
    elif isinstance(model, LindbladModel):
        cur_ham_ops, cur_dis_ops = model.operators
        cur_ham_sig, cur_dis_sig = model.signals

        new_ham_sig, new_ham_ops = _get_new_operators(
            cur_ham_ops, cur_ham_sig, model.frame, frame_freqs, cutoff_freq
        )
        if cur_dis_ops is not None and cur_dis_sig is not None:
            new_dis_sig, new_dis_ops = _get_new_operators(
                cur_dis_ops, cur_dis_sig, model.frame, frame_freqs, cutoff_freq
            )

        new_model = LindbladModel(
            new_ham_ops,
            new_ham_sig,
            new_dis_ops,
            new_dis_sig,
            new_drift,
            model.frame,
            model.evaluation_mode,
        )
    return new_model
