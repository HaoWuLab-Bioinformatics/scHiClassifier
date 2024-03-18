import numpy as np

# 该函数用于计算单个染色质上的总接触数，因为矩阵是对称的，所以算染色质上的总接触数只需要计算上三角元素+对角线元素。
def sum_matrix(matrix):
    # U取了matrix矩阵的上三角部分，不包括对角线：
    # 若matrix为[[3,2,1],  那么U为[[0 2 1],   D为[[3 0 0],
    #           [1,2,1],         [0 0 1],       [0 2 0],
    #           [1,2,1]]         [0 0 0]]       [0 0 1]]
    U = np.triu(matrix, 1)
    # D取了matrix矩阵的对角线元素，
    D = np.diag(np.diag(matrix))
    return sum(sum(U + D))  # 先按列相加，最后再把所有列都加起来
    # 假设总和为198,那么sum(U + D)的结果为(198,)
    # sum(sum(U + D)) 就得到198了


# 获取矩阵matrix中主对角线上的元素的接触数总和
def sum_diagonal_matrix(matrix):
    # 只取matrix的对角线元素，其余元素均为0
    D = np.diag(np.diag(matrix))
    return sum(sum(D))


# 该函数用来返回当前bin所有的接触信息
def find_contact_list(list):
    con = []
    for i in list:
        if i != 0:
            con.append(i)
    return con

# 该函数用来对当前bin进行平滑计算
def smooth_mean(list):
    # 在numpy的sum函数中axis为0是压缩行，即将每一列的元素相加,将矩阵压缩为一行
    # 以m = numpy.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])为例
    # print(numpy.sum(m,axis = 0))结果为[22 26 30]
    # 在下面那一行代码中该函数对当前bin的空间相邻（线性+有相互作用的）bin的接触信息组成的矩阵的每一列进行了求和
    # 求和后再除以List列表的长度，它的长度为b+1，其中b为邻居的个数，再加上自己。
    return np.sum(np.array(list), axis=0) / len(list)

