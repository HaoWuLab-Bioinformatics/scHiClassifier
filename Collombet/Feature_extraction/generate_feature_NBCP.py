import numpy as np
from methods import sum_matrix, find_contact_list


# 该函数用来计算单条染色质对应的NBCP特征
def calculate_NBCP(contact_matrix, max_length):
    NBCP = []
    # sum_matrix用于计算单个染色质上的总接触数，因为矩阵是对称的，所以算染色质上的总接触数只需要计算上三角元素+对角线元素。
    chr_total = sum_matrix(contact_matrix)
    # 0号行没有上邻居所以，只放它自己和它的下邻居
    con_list1 = find_contact_list(contact_matrix[0])
    con_list2 = find_contact_list(contact_matrix[1])
    sum_list = con_list1 + con_list2
    # bin对应的邻居列表
    if len(sum_list) == 0:
        NBCP.append(0)
    else:
        con_pro = sum(sum_list) / chr_total
        NBCP.append(con_pro)

    # 一共有max_length行数据，因为这个是NBCP，0号行没有上邻居，max_length-1行数据没有下邻居，所以真实范围应该是1到max_length-2行
    for i in range(1, max_length-1):
        con_list1 = find_contact_list(contact_matrix[i])
        con_list2 = find_contact_list(contact_matrix[i-1])
        con_list3 = find_contact_list(contact_matrix[i+1])
        final_list = con_list1 + con_list2 + con_list3
        # bin对应的邻居列表
        if len(final_list) == 0:
            NBCP.append(0)
        else:
            con_pro = sum(final_list)/chr_total
            NBCP.append(con_pro)
    # max_length-1行数据没有下邻居，所以只放它和它的上邻居
    con_list1 = find_contact_list(contact_matrix[max_length-2])
    con_list2 = find_contact_list(contact_matrix[max_length-1])
    sum_list = con_list1 + con_list2
    # bin对应的邻居列表
    if len(sum_list) == 0:
        NBCP.append(0)
    else:
        con_pro = sum(sum_list) / chr_total
        NBCP.append(con_pro)
    return NBCP

def con_ran(cell_id, type, chr_name, max_length):
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
    nbcp = calculate_NBCP(contact_matrix, max_length)

    return nbcp


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
                NBCP = con_ran(cell_id, type, chr, max_len)
                cell_dict[str(cell_number)][chr] = NBCP
    print(cell_dict)
    out_path = '../Collombet_features/NBCP.npy'
    np.save(out_path, cell_dict)


if __name__ == '__main__':
    main()
    # bcp是算一行占矩阵总和的。nbcp算3行5行、7行占矩阵的总和，这是他的邻居。
    # with0就是带首行和最后一行的

