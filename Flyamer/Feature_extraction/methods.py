import numpy as np

# This function is used to calculate the total number of contacts on a single chromatin, since the matrix is symmetric, counting the total number of contacts on a chromatin only requires counting the upper triangular elements + diagonal elements.
def sum_matrix(matrix):
    U = np.triu(matrix, 1)
    D = np.diag(np.diag(matrix))
    return sum(sum(U + D))


# Get the sum of the contacts of the elements on the main diagonal of the matrix matrix
def sum_diagonal_matrix(matrix):
    # Take only the diagonal elements of the matrix, the rest of the elements are 0
    D = np.diag(np.diag(matrix))
    return sum(sum(D))


# This function is used to return all contact information for the current bin
def find_contact_list(list):
    con = []
    for i in list:
        if i != 0:
            con.append(i)
    return con

# This function is used to perform smoothing calculations on the current bin
def smooth_mean(list):
    # In numpy's sum function axis is 0 is compressed rows, that is, the elements of each column are added, the matrix will be compressed into a line
    # to m = numpy.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]) for example
    # print(numpy.sum(m,axis = 0)) results in [22 26 30].
    # In the following line of code the function sums each column of a matrix consisting of contact information from spatially adjacent (linear + interacting) bins of the current bin.
    # The sum is then divided by the length of the List list, which is b+1, where b is the number of neighbors, plus itself.
    return np.sum(np.array(list), axis=0) / len(list)


# This function is used to implement the restart randomized wandering algorithm, where the argument rp is the restart probability.
def random_walk_imp(matrix, rp):
    row, _ = matrix.shape
    row_sum = np.sum(matrix, axis=1)
    for i in range(row_sum.shape[0]):
        if row_sum[i] == 0:
            row_sum[i] = 0.001
    nor_matrix = np.divide(matrix.T, row_sum).T
    Q = np.eye(row)
    I = np.eye(row)
    for i in range(30):
        # The random wandering process can be represented by the following matrix operation.
        Q_new = rp * np.dot(Q, nor_matrix) + (1 - rp) * I
        delta = np.linalg.norm(Q - Q_new)
        Q = Q_new.copy()
        # When delta is less than the threshold, the restart random walk process converges and the restart random walk process terminates.
        if delta < 1e-6:
            break
    return Q