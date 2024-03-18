import numpy as np
from methods import sum_matrix

# 该函数用来计算单条染色质对应的PSDCP特征
def calculate_PSDCP(contact_matrix, index, scale):
    contact_matrix = np.array(contact_matrix)
    new_matrix = np.zeros((index + 2 * scale, index + 2 * scale))
    PSDCP = []
    chr_total = sum_matrix(contact_matrix)
    for i in range(index):
        for j in range(index):
            new_matrix[i + scale, j + scale] = contact_matrix[i, j]
    for i in range(index):
        bin = i + scale
        a = sum_matrix(new_matrix[bin - scale:bin + scale + 1, bin - scale:bin + scale + 1])
        if a == 0:
            PSDCP.append(float(0))
        else:
            PSDCP.append(float(a / chr_total))
    return PSDCP


def con_ran(cell_id, type, chr_name, max_length):
    file_path = "../Lee_Data/%s/cell_%s_%s.txt" % (type, str(cell_id), chr_name)
    chr_file = open(file_path)
    # scale为PSDCP的区域规模
    scale = 2
    lines = chr_file.readlines()
    # 初始化接触矩阵为零矩阵
    contact_matrix = np.zeros((max_length, max_length))
    for line in lines:
        # bin1，bin2是两个染色体片段的编号，num是bin1和bin2的接触数
        bin1, bin2, num = line.split()
        # 这个跟其他数据集不一样，因为其他数据集如果是1号染色体能切250个bin，那么这些bin的编号是0到249，但是本数据集是1到250
        bin1 = int(bin1) - 1
        bin2 = int(bin2) - 1
        contact_matrix[int(bin1), int(bin2)] += int(float(num))
        # 因为这个矩阵是对称阵，所以如果bin1不等于bin2的话，就需要对称一下值
        if bin1 != bin2:
            contact_matrix[int(bin2), int(bin1)] += int(float(num))
    PSDCP = calculate_PSDCP(contact_matrix, max_length, scale)
    return PSDCP


def main():
    # types存的是
    types = {'Astro': 449, 'Endo': 204, 'L4': 131, 'L5': 180, 'L6': 86, 'L23': 551, 'MG': 422, 'Ndnf': 144,
             'NN1': 100, 'ODC': 1243, 'OPC': 203, 'Pvalb': 134, 'Sst': 216, 'Vip': 171}
    # 分辨率为1mbp就是指，1M（1000000）个碱基对作为分辨率
    resolution = 1000000
    tps = sorted(types)
    # print(tps) ['1CSE', '2CSE', '4CSE', '64CSE', '8CSE']
    f = open('../combo_hg19.genomesize')
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
                PSDCP = con_ran(cell_id, type, chr, max_len)
                cell_dict[str(cell_number)][chr] = PSDCP
    out_path = '../Lee_features/PSDCP.npy'
    np.save(out_path, cell_dict)


if __name__ == '__main__':
    main()

