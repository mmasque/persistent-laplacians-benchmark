import numpy as np


def boundary_matrices(simplex_tree, dtype=np.int8):
    max_dim = simplex_tree.dimension()
    records = {q: [] for q in range(max_dim + 1)}

    # GUDHI returns simplices in increasing filtration order.
    for simplex, value in simplex_tree.get_filtration():
        simplex = tuple(sorted(simplex))
        q = len(simplex) - 1
        records[q].append((float(value), simplex))

    # Filtration order, with deterministic ordering of ties.
    for q in records:
        records[q].sort(key=lambda item: (item[0], item[1]))

    simplices = {
        q: [simplex for _, simplex in records[q]]
        for q in records
    }

    births = {
        q: np.asarray([value for value, _ in records[q]])
        for q in records
    }

    levels = np.asarray(sorted({
        value
        for q in records
        for value, _ in records[q]
    }))

    D = {}

    for q in range(1, max_dim + 1):
        row_index = {
            simplex: row
            for row, simplex in enumerate(simplices[q - 1])
        }

        Dq = np.zeros(
            (len(simplices[q - 1]), len(simplices[q])),
            dtype=dtype,
        )

        for column, simplex in enumerate(simplices[q]):
            for k in range(q + 1):
                face = simplex[:k] + simplex[k + 1:]
                Dq[row_index[face], column] = (-1) ** k

        D[q] = Dq
    
    # cuts[i, q] = number of q-simplices present at filtration index i
    cuts = np.column_stack([
        np.searchsorted(births[q], levels, side="right")
        for q in range(max_dim + 1)
    ])

    return D, levels, cuts 

def picker(cuts, i):
    """Indices selecting each boundary matrix at filtration index i."""
    return {
        q: np.s_[:cuts[i, q - 1], :cuts[i, q]]
        for q in range(1, cuts.shape[1])
    }


def print_boundary_dimensions(D, cuts, i, j):
    """
    Print the dimensions of every boundary matrix in K_i and K_j.
    """
    if not 0 <= i <= j < len(cuts):
        raise ValueError("Require 0 <= i <= j < len(cuts)")

    print(f"{'matrix':<8} {'K_' + str(i):>16} {'K_' + str(j):>16}")

    for q in sorted(D):
        shape_i = (int(cuts[i][q - 1]), int(cuts[i][q]))
        shape_j = (int(cuts[j][q - 1]), int(cuts[j][q]))

        print(f"D_{q:<6} {str(shape_i):>16} {str(shape_j):>16}")
