import numpy as np
import scipy.linalg as la
from contextlib import contextmanager
from time import perf_counter


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

def prepare_up_persistent_relative_cochain_laplacian(
    boundary_matrices,
    cuts,
    q,
    i,
    j,
    l,
):
    """ the persistent down term in cohomological
    degree q for

        C^*(K_l, K_j) subset C^*(K_l, K_i).

    The returned matrix is ordered as

        rows:
            q-cochains in K_l \\ K_j,
            q-cochains in K_j \\ K_i,

        columns:
            (q-1)-cochains in K_l \\ K_j,
            (q-1)-cochains in K_j \\ K_i.

    Hence it has block form

        [[B, E],
         [0, A]],

    where A is the matrix whose nullspace must be computed.

    Returns
    -------
    coboundary : ndarray
        Reordered matrix of delta_{q-1}^{K_l, K_i}.
    n_q_relative : int
        Dimension of C^q(K_l, K_j).
    n_qm1_relative : int
        Dimension of C^(q-1)(K_l, K_j).
    """
    if not 0 <= i <= j <= l < len(cuts):
        raise ValueError("Require 0 <= i <= j <= l < len(cuts)")

    max_dim = cuts.shape[1] - 1
    if not 0 <= q <= max_dim:
        raise ValueError(f"Require 0 <= q <= {max_dim}")

    n_q_i = int(cuts[i][q])
    n_q_j = int(cuts[j][q])
    n_q_l = int(cuts[l][q])

    # V_q = C^q(K_l, K_j)
    n_q_relative = n_q_l - n_q_j

    # S_q = C^q(K_j, K_i)
    n_q_middle = n_q_j - n_q_i

    # There is no incoming coboundary in degree zero.
    if q == 0:
        coboundary = np.zeros(
            (n_q_relative + n_q_middle, 0),
            dtype=np.float64,
        )
        return coboundary, n_q_relative, 0

    n_qm1_i = int(cuts[i][q - 1])
    n_qm1_j = int(cuts[j][q - 1])
    n_qm1_l = int(cuts[l][q - 1])

    n_qm1_relative = n_qm1_l - n_qm1_j
    n_qm1_middle = n_qm1_j - n_qm1_i

    # Boundary C_q(K_l) -> C_(q-1)(K_l).
    Dq = boundary_matrices[q]

    coboundary = np.zeros(
        (
            n_q_relative + n_q_middle,
            n_qm1_relative + n_qm1_middle,
        ),
        dtype=np.float64,
    )

    # B = delta_(q-1) on C^*(K_l, K_j).
    coboundary[
        :n_q_relative,
        :n_qm1_relative,
    ] = Dq[
        n_qm1_j:n_qm1_l,
        n_q_j:n_q_l,
    ].T

    # E: middle (q-1)-cochains mapping into outer q-cochains.
    coboundary[
        :n_q_relative,
        n_qm1_relative:,
    ] = Dq[
        n_qm1_i:n_qm1_j,
        n_q_j:n_q_l,
    ].T

    # Lower-left block is zero because K_j is a subcomplex.

    # A = delta_(q-1) on C^*(K_j, K_i).
    coboundary[
        n_q_relative:,
        n_qm1_relative:,
    ] = Dq[
        n_qm1_i:n_qm1_j,
        n_q_i:n_q_j,
    ].T

    return coboundary, n_q_relative, n_qm1_relative


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

def compute_null_space_D_profiling(
    B_j,
    n_q_i,
    n_q1_i,
    rcond=1e-6,
):
    """Compute the persistent up-Laplacian using ker(D), with profiling."""
    timings = {}
    shapes = {}
    total_start = perf_counter()

    @contextmanager
    def measured(label):
        start = perf_counter()
        try:
            yield
        finally:
            timings[label] = perf_counter() - start

    def finish(L, exit_reason):
        timings["total"] = perf_counter() - total_start
        shapes["L"] = L.shape

        return L, {
            "timings": timings,
            "shapes": shapes,
            "exit_reason": exit_reason,
        }

    with measured("_blocks"):
        B_i, E, D = _blocks(B_j, n_q_i, n_q1_i)

    shapes.update({
        "B_i": B_i.shape,
        "E": E.shape,
        "D": D.shape,
    })

    with measured("B_i @ B_i.T"):
        L = B_i @ B_i.T

    if E.shape[1] == 0:
        return finish(L, "E has no columns")

    if D.shape[0] == 0:
        with measured("E @ E.T"):
            E_term = E @ E.T

        with measured("add E term"):
            L += E_term

        return finish(L, "D has no rows")

    with measured("null_space(D)"):
        Z = la.null_space(
            D,
            rcond=rcond,
            check_finite=False,
        )

    shapes["Z"] = Z.shape

    if Z.shape[1] == 0:
        return finish(L, "ker(D) is zero")

    with measured("E @ Z"):
        EZ = E @ Z

    shapes["EZ"] = EZ.shape

    with measured("EZ @ EZ.T"):
        null_space_term = EZ @ EZ.T

    with measured("add null-space term"):
        L += null_space_term

    return finish(L, "full null-space computation")

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

def compute_schur_D_profiling(
    B_j,
    n_q_i,
    n_q1_i,
    rcond=1e-12
):
    """Compute the persistent up-Laplacian by a Schur complement using profiling.
    """
    timings = {}
    total_start = perf_counter()

    @contextmanager
    def measured(label):
        start = perf_counter()
        try:
            yield
        finally:
            timings[label] = perf_counter() - start

    def finish(L, exit_reason):
        timings["total"] = perf_counter() - total_start

        return L, {
            "timings": timings,
            "shapes": shapes,
            "exit_reason": exit_reason,
        }

    with measured("_blocks"):
        B_i, E, D = _blocks(B_j, n_q_i, n_q1_i)

    shapes = {
        "B_i": B_i.shape,
        "E": E.shape,
        "D": D.shape,
    }

    with measured("B_i @ B_i.T"):
        L = B_i @ B_i.T

    if E.shape[1] == 0:
        return finish(L, "E has no columns")

    with measured("E @ E.T"):
        E_term = E @ E.T

    with measured("add E term"):
        L += E_term

    del E_term

    if D.shape[0] == 0 or D.shape[1] == 0:
        return finish(L, "D is empty")

    with measured("E @ D.T"):
        L_IO = E @ D.T

    with measured("D @ D.T"):
        L_OO = D @ D.T

    shapes.update({
        "L_IO": L_IO.shape,
        "L_OO": L_OO.shape,
        "L": L.shape,
    })

    with measured("pinv(L_OO)"):
        L_OO_pinv = np.linalg.pinv(
            L_OO,
            rcond=rcond,
            hermitian=True,
        )

    # Split the two products so they receive separate timings.
    with measured("L_IO @ pinv"):
        schur_left = L_IO @ L_OO_pinv

    with measured("schur_left @ L_IO.T"):
        correction = schur_left @ L_IO.T

    with measured("subtract correction"):
        L -= correction

    return finish(L, "full Schur computation")


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