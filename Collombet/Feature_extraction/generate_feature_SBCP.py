import numpy as np
from methods import find_contact_list, sum_matrix, smooth_mean

# 该函数用来计算单条染色质对应的SBCP特征
def calculate_SBCP(contact_matrix, max_length):
    # 先平滑
    matrix1 = contact_matrix
    # len(contact_matrix)为接触矩阵的行数
    for i in range(len(contact_matrix)):
        # 列表space_neighbour_contact_infor用来存放当前bin的空间相邻bin(线性相邻+相互作用即空间相邻)的接触信息。
        space_neighbour_contact_infor = []
        # 列表index1用来存放与当前bin线性相邻的bin的编号
        index1 = []
        # 先把当前bin自己的接触信息放进去
        space_neighbour_contact_infor.append(contact_matrix[i, :])

        # 该循环用于取得与当前bin线性相邻的bin(左右邻居)的接触信息
        # m的取值为i-1、i、i+1。
        for m in range(i - 1, i + 2):
            # 因为数组是从0号元素开始的(即接触矩阵是从0行开始的)所以m不能<0
            # 因为数组最多到len(matrix)-1号元素(即接触矩阵最多到len(matrix)-1行），所以m不能>len(matrix)-1
            # 因为此循环外，上边一行代码已经取了染色体片段i的接触信息了，此处不能再取一次，所以m不能等于i。
            if m < 0 or m > len(contact_matrix) - 1 or m == i:
                continue
            space_neighbour_contact_infor.append(contact_matrix[m, :])
            # 记录与当前bin线性相邻的bin的编号
            index1.append(m)

        # 此循环的作用是取得与当前bin相互作用(即空间相邻)的bin的接触信息。
        for n in range(len(contact_matrix[0])):
            # 如果n已经是当前bin的线性邻居的话就跳过
            if n in index1:
                continue
            # matrix[i, n] ！= 0则说明当前bin与bin_n发生相互作用，即空间相邻。
            # 而且因为此前已经取了当前bin的接触信息了，此处不能再取一次，所以i!=n。
            if contact_matrix[i, n] != 0 and i != n:
                space_neighbour_contact_infor.append(contact_matrix[n, :])
        # 用平滑处理过后的bin接触信息替换原来bin对应的接触信息
        matrix1[i, :] = smooth_mean(space_neighbour_contact_infor)
    # 为什么要进行以下三行代码生成final_matrix？
    # 答：因为matrix1它是一个染色体接触矩阵，上三角形和下三角形必须是对称的，也就是说matrix1[i,j]和matrix1[j,i]必须保持一致。
    #    但是由于每个bin的空间邻近bin是不同的，所以我们一行一行生成的matrix1矩阵它是不对称的。
    #    为了保持矩阵是对称的，我们就把上三角映射到下三角。（其实把下三角映射到上三角也是可以的）。
    # 已知matrix1是一个方阵（行数和列数相同），那么以m = np.array([[1,2,3],[4,5,6],[7,8,9]])这个3×3的方阵为例
    # a = np.triu(m, 1)   b = np.triu(m, 1).T   print(a)和print(b)的结果分别为：
    # [[0 2 3]     [[0 0 0]
    #  [0 0 6]      [2 0 0]
    #  [0 0 0]]     [3 6 0]]
    # d = np.diag(m)   d2 = np.diag(np.diag(m))  print(d)和print(d2)的结果分别为
    #  [1 5 9]     [[1 0 0]
    #               [0 5 0]
    #               [0 0 9]]
    # m是取方阵matrix1的上三角的所有元素（不包含主对角线），并将主对角线和下三角的所有元素赋值为0。
    m = np.triu(matrix1, 1)
    n = m.T
    # np.diag(matrix1)提取了方阵matrix1的主对角线上的元素。
    # np.diag(np.diag(matrix1))将以np.diag(matrix1)提取出的方阵matrix1的主对角线上的元素作为主对角线，构建一个除主对角线外的其他元素都为0的方阵。
    smoothed_matrix = m + n + np.diag(np.diag(matrix1))
    # ------------------------------------------------------------
    # 再求特征
    SBCP = []
    # sum_matrix用于计算单个染色质上的总接触数，因为矩阵是对称的，所以算染色质上的总接触数只需要计算上三角元素+对角线元素。
    chr_total = sum_matrix(smoothed_matrix)
    for i in range(max_length):
        con_list = find_contact_list(smoothed_matrix[i])
        if len(con_list) == 0:
            SBCP.append(0)
        else:
            con_pro = sum(con_list) / chr_total
            SBCP.append(con_pro)
    return SBCP

def con_ran(cell_id,type,chr_name,max_length):
    file_path = "../Collombet_Data/%s/cell_%s_%s.txt" % (type, str(cell_id), chr_name)
    chr_file = open(file_path)
    lines = chr_file.readlines()
    # 初始化接触矩阵为零矩阵
    contact_matrix = np.zeros((max_length, max_length))
    for line in lines:
        # bin1，bin2是两个染色体片段的编号，num是bin1和bin2的接触数
        bin1, bin2, num = line.split()
        contact_matrix[int(bin1), int(bin2)] += int(float(num))
        # 因为这个矩阵是对称阵，所以如果bin1不等于bin2的话，就需要对称一下值
        if bin1 != bin2:
            contact_matrix[int(bin2), int(bin1)] += int(float(num))
    SBCP = calculate_SBCP(contact_matrix, max_length)
    return SBCP


def main():
    # types存的是
    types = {'1CSE': 85, '2CSE': 114, '4CSE': 121, '64CSE': 205, '8CSE': 123}  # 一共648个细胞
    # 分辨率为1mbp就是指，1M（1000000）个碱基对作为分辨率
    resolution = 1000000
    tps = sorted(types)
    # print(tps) ['1CSE', '2CSE', '4CSE', '64CSE', '8CSE']
    f = open('../mm10.main.nochrM.chrom.sizes')
    index = {}
    lines = f.readlines()
    for line in lines:
        chr_name, length = line.split()
        # max_len+1是指一个长度为length的染色体在resolution分辨率下能分成max_len+1块
        # 为什么要+1？因为int(10/3)=3，是向下取整的，多出来的那一截，也要做一块。
        max_len = int(int(length) / resolution)
        # index字典中index[chr_name]存的是染色体chr_name在分别率为resolution时能分出的块数
        index[chr_name] = max_len + 1
        # f.seek(0)用于将光标移到开头
    f.seek(0)
    # 关闭文件流
    f.close()
    print(index)
    cell_dict = {}
    cell_number = 0
    chr_list = sorted(index.keys())
    for type in tps:
        for c_num in range(types[type]):
            cell_id = c_num + 1
            cell_number += 1
            cell_dict[str(cell_number)] = {}
            print(c_num)
            for chr in chr_list:
                max_len = index[chr]
                SBCP = con_ran(cell_id, type, chr, max_len)
                cell_dict[str(cell_number)][chr] = SBCP
    print(cell_dict)
    out_path = '../Collombet_features/SBCP.npy'
    np.save(out_path, cell_dict)

if __name__ == '__main__':
    main()
