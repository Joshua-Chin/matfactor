#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

from cpython cimport bool

from cython.parallel import prange

import numpy as np
cimport numpy as np

from libc.math cimport sqrt, log, pow

cdef packed struct Entry:
    int row, col
    double val

cdef inline int max(int a, int b) nogil: return a if a > b else b

def factorize(x,
              int dim=40,
              int epochs=5,
              bool shuffle=True,
              double learning_rate=0.05,
              int x_max=100,
              double alpha=0.75,
              int threads=8,
              bool verbose=True):
    """
    factorize(x, dim=40, epochs=5, learning_rate=0.05, x_max, alpha=0.75, threads=8)

    Factorizes a sparse matrix `x` as `e ** (u * v)`

    Solves the equation `x = e ** (u * v)` by minimizing
    `sum_{i,j} f(X_ij)(u_i \dot v_j - log X_ij)^2` where
    `f(x) = min((x / x_max) ** alpha, 1)`. This function
    uses the adagrad gradient descent algorithm.

    This function was inspired by GloVe (Pennington et Al).
    You can view their site at `nlp.stanford.edu/projects/glove/`
    This function differs by omitting the bias vector and
    allowing an arbitrarily shaped sparse matrix as input.

    Parameters
    ----------
    x : a scipy.sparse matrix
        A real matrix of shape (M, N)
    dim : int, optional
        The number of columns of u and v
    shuffle : bool, optional
        Whether to shuffle the input matrix
    learning_rate : float, optional
        The initial learning rate
    epochs : int, optional
        The number of epochs to run
    x_max : int, optional
        Each entry is weighted by min((x / x_max) ** alpha, 1)
    alpha : double, optional
        Each entry is weighted by min((x / x_max) ** alpha, 1)
    threads : int, optional
        The number of threads to use
    verbose: bool, optional
        Whether to print the error for each epoch

    Returns
    -------
    u:(M, dim) array
    v:(M, dim) array
    """

    cdef int n_rows, n_cols, epoch, i, j
    cdef int[:] rows, cols
    cdef double total_error, prediction, diff, fdiff, t1, t2
    cdef double[:] vals, error
    cdef double[:, :] row_vecs, col_vecs, row_grad_sq, col_grad_sq
    cdef Entry elem
    cdef Entry[:] coo_matrix

    x = x.tocoo()
    rows = x.row
    cols = x.col
    vals = x.data

    coo_matrix = np.zeros(
        vals.shape,
        dtype=np.dtype([('row', np.int32), ('col', np.int32), ('val', np.float64)])
    )

    for i in range(vals.shape[0]):
        coo_matrix[i].row = rows[i]
        coo_matrix[i].col = cols[i]
        coo_matrix[i].val = vals[i]

    if shuffle:
        np.random.shuffle(coo_matrix)

    n_rows = 0
    n_cols = 0
    for i in range(coo_matrix.shape[0]):
        n_rows = max(n_rows, coo_matrix[i].row)
        n_cols = max(n_cols, coo_matrix[i].col)

    n_rows += 1
    n_cols += 1

    row_vecs = (np.random.rand(n_rows, dim) - 0.5) / dim
    col_vecs = (np.random.rand(n_cols, dim) - 0.5) / dim

    row_grad_sq = np.ones_like(row_vecs)
    col_grad_sq = np.ones_like(col_vecs)

    error = np.zeros(coo_matrix.shape[0])

    for epoch in range(epochs):

        for i in prange(coo_matrix.shape[0],
                        num_threads=threads,
                        nogil=True,
                        chunksize=coo_matrix.shape[0] // threads + 1,
                        schedule='static'):

            elem = coo_matrix[i]

            prediction = 0.0
            for j in range(dim):
                prediction = prediction + row_vecs[elem.row, j] * col_vecs[elem.col, j]

            diff = prediction - log(elem.val)
            fdiff = diff if elem.val > x_max else pow(elem.val / x_max, alpha) * diff

            error[i] = 0.5 * fdiff * diff
            fdiff = fdiff * learning_rate

            for j in range(dim):
                t1 = fdiff * row_vecs[elem.row, j]
                t2 = fdiff * col_vecs[elem.col, j]
                row_vecs[elem.row, j] -= t2 / sqrt(row_grad_sq[elem.row, j])
                col_vecs[elem.col, j] -= t1 / sqrt(col_grad_sq[elem.col, j])

                row_grad_sq[elem.row, j] += t2 * t2
                col_grad_sq[elem.col, j] += t1 * t1

        total_error = 0.0
        for i in range(coo_matrix.shape[0]):
            total_error += error[i]
        if verbose:
          print(total_error / error.shape[0])

    return np.asarray(row_vecs), np.asarray(col_vecs)
