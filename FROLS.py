import numpy as np


def FROLS(Theta, y, tol=1e-2, max_terms=None):
    """
    Forward Regression Orthogonal Least Squares (FROLS)

    Selects terms from Theta greedily to explain y, stopping when the
    Error Reduction Ratio (ERR) of remaining terms falls below tol.

    Parameters
    ----------
    Theta : np.ndarray, shape (N, M)
        Library matrix of candidate terms.
    y : np.ndarray, shape (N,)
        Target (e.g. a_dot[:, i]).
    tol : float
        Stopping threshold on cumulative ESR (stop when unexplained
        variance < tol). Default 1e-2.
    max_terms : int or None
        Maximum number of terms to select. Defaults to M.

    Returns
    -------
    xi : np.ndarray, shape (M,)
        Sparse coefficient vector (zeros for unselected terms).
    """
    N, M = Theta.shape
    if max_terms is None:
        max_terms = M

    sigma = y @ y  # yTy

    selected = []  # indices of selected columns (in original Theta)
    Q = np.zeros((N, M))  # orthogonalised bases, filled as we go
    A = np.zeros((M, M))  # upper triangular matrix of inner products
    g = np.zeros(M)  # g[s] = (q_s^T y) / (q_s^T q_s)

    remaining = list(range(M))
    ERR_cumulative = 0.0

    # --- Orthogonal candidates from the current step's residual bases ---
    # P holds the current orthogonalised versions of all remaining candidates
    # We re-orthogonalise against selected bases at each step.

    for s in range(max_terms):
        best_err = -1.0
        best_m = None
        best_q = None

        # For each remaining candidate, orthogonalise against selected bases
        # and compute its ERR contribution
        for m in remaining:
            q = Theta[:, m].copy()

            # Orthogonalise against all previously selected bases
            for j in range(s):
                q_j = Q[:, j]
                q = q - (q_j @ Theta[:, m]) / (q_j @ q_j) * q_j

            qTq = q @ q
            if qTq < 1e-14:
                # Numerically degenerate — skip
                continue

            err_m = (q @ y) ** 2 / (qTq * sigma)

            if err_m > best_err:
                best_err = err_m
                best_m = m
                best_q = q

        if best_m is None:
            break

        # Accept the best candidate
        Q[:, s] = best_q
        g[s] = (best_q @ y) / (best_q @ best_q)

        # Store the a_{js} coefficients needed to recover beta later
        # a_{j, s} = (q_j^T p_{best_m}) / (q_j^T q_j)  for j < s
        for j in range(s):
            q_j = Q[:, j]
            A[j, s] = (q_j @ Theta[:, best_m]) / (q_j @ q_j)
        A[s, s] = 1.0

        selected.append(best_m)
        remaining.remove(best_m)

        ERR_cumulative += best_err

        # Stopping criterion: ESR (explained sum ratio) >= 1 - tol
        if 1.0 - ERR_cumulative < tol:
            s_final = s + 1
            break
    else:
        s_final = len(selected)

    # --- Recover coefficients beta from A beta = g (back substitution) ---
    # A is upper triangular with shape (s_final, s_final)
    s_final = len(selected)
    A_sub = A[:s_final, :s_final]
    g_sub = g[:s_final]

    # Back substitution: A_sub @ beta_sub = g_sub
    beta_sub = np.zeros(s_final)
    for i in range(s_final - 1, -1, -1):
        beta_sub[i] = g_sub[i] - A_sub[i, i + 1:s_final] @ beta_sub[i + 1:s_final]
        # A[i,i] == 1.0 always, so no division needed

    # Place coefficients back into full-length xi vector
    xi = np.zeros(M)
    for idx, m in enumerate(selected):
        xi[m] = beta_sub[idx]

    return xi