import numpy as np
import scipy.linalg as la


def _dense(block):
    if hasattr(block, "toarray"):
        block = block.toarray()
    return np.ascontiguousarray(block, dtype=np.float64)


def _blocks(B_j, n_q_i, n_q1_i):
    """Split B_j as [[B_i, E], [0, D]]."""
    B_i = _dense(B_j[:n_q_i, :n_q1_i])
    E = _dense(B_j[:n_q_i, n_q1_i:])
    D = _dense(B_j[n_q_i:, n_q1_i:])
    return B_i, E, D


def _rank(s, rcond):
    if s.size == 0 or s[0] == 0:
        return 0
    return int(np.count_nonzero(s > rcond * s[0]))


def prepare_up_persistent_laplacian(
    boundary_matrices, cuts, q, i, j
):
    if not 0 <= i <= j < len(cuts):
        raise ValueError("Require 0 <= i <= j < len(cuts)")

    n_q_i = int(cuts[i][q])
    n_q1_i = int(cuts[i][q + 1])

    n_q_j = int(cuts[j][q])
    n_q1_j = int(cuts[j][q + 1])

    B_j = np.ascontiguousarray(
        boundary_matrices[q + 1][:n_q_j, :n_q1_j],
        dtype=np.float64,
    )

    return B_j, n_q_i, n_q1_i


def compute_null_space_D(B_j, n_q_i, n_q1_i, rcond=1e-6):
    """Compute the persistent up-Laplacian using ker(D)."""
    B_i, E, D = _blocks(B_j, n_q_i, n_q1_i)

    L = B_i @ B_i.T
    if E.shape[1] == 0:
        return L
    if D.shape[0] == 0:
        return L + E @ E.T

    Z = la.null_space(D, rcond=rcond, check_finite=False)
    if Z.shape[1] != 0:
        EZ = E @ Z
        L += EZ @ EZ.T
    return L


def compute_reduced_svd_D(B_j, n_q_i, n_q1_i, rcond=1e-6):
    """Compute the persistent up-Laplacian using the row space of D."""
    B_i, E, D = _blocks(B_j, n_q_i, n_q1_i)

    L = B_i @ B_i.T
    if E.shape[1] == 0:
        return L

    L += E @ E.T
    if D.shape[0] == 0:
        return L

    _, s, Vh = la.svd(
        D,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesdd",
    )
    rank = _rank(s, rcond)
    if rank != 0:
        EV = E @ Vh[:rank].T
        L -= EV @ EV.T
    return L


def compute_schur_D(B_j, n_q_i, n_q1_i, rcond=1e-12):
    """Compute the persistent up-Laplacian by a Schur complement."""
    B_i, E, D = _blocks(B_j, n_q_i, n_q1_i)

    L = B_i @ B_i.T

    if E.shape[1] == 0:
        return L

    L += E @ E.T

    if D.shape[0] == 0 or D.shape[1] == 0:
        return L

    L_IO = E @ D.T
    L_OO = D @ D.T

    L_OO_pinv = np.linalg.pinv(
        L_OO,
        rcond=rcond,
        hermitian=True,
    )

    L -= (L_IO @ L_OO_pinv) @ L_IO.T

    return L


def compute_schur_D_lstsq(B_j, n_q_i, n_q1_i, rcond=1e-12):
    B_i, E, D = _blocks(B_j, n_q_i, n_q1_i)

    L = B_i @ B_i.T

    if E.shape[1] == 0:
        return L

    if D.shape[0] == 0:
        L += E @ E.T
        return L

    X, _, _, _ = la.lstsq(
        D.T,
        E.T,
        cond=rcond,
        lapack_driver="gelsy",
        check_finite=False,  # safe here if boundary matrices are known finite
    )

    residual = E.T - D.T @ X
    L += residual.T @ residual
    return L

def compute_null_space_clearing_D(
    B_j,
    n_q_i,
    n_q1_i,
    higher_output=None,
    rcond=1e-6,
):
    """Compute one dimension using the clearing output from the dimension above.

    Process dimensions from high to low.  Call the highest dimension with
    ``higher_output=None``.  The second returned value is passed unchanged as
    ``higher_output`` at the next lower dimension.
    """
    B_i, E, D = _blocks(B_j, n_q_i, n_q1_i)

    if higher_output is None:
        known_image = None
        complement = None
    else:
        higher_U, higher_rank = higher_output
        if higher_U.shape != (D.shape[1], D.shape[1]):
            raise ValueError(
                "higher_output is incompatible with the current D block"
            )
        known_image = higher_U[:, :higher_rank]
        complement = higher_U[:, higher_rank:]

        if higher_rank == 0:
            known_image = None
            complement = None

    if complement is None:
        compressed_D = D
        compressed_E = E
    else:
        compressed_D = D @ complement
        compressed_E = E @ complement

    U, s, Vh = la.svd(
        compressed_D,
        full_matrices=True,
        check_finite=False,
        lapack_driver="gesdd",
    )
    rank = _rank(s, rcond)

    L = B_i @ B_i.T

    if known_image is not None:
        EW = E @ known_image
        L += EW @ EW.T

    if compressed_D.shape[1] > rank:
        EZ = compressed_E @ Vh[rank:].T
        L += EZ @ EZ.T

    return L, (U, rank)