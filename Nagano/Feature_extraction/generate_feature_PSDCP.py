import numpy as np
from methods import sum_matrix

# This function is used to compute the PSDCP features corresponding to a single chromatin
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
    file_path = "../Nagano_Data/%s/%s_%s/%s.txt" % (type, type,str(cell_id), chr_name)
    chr_file = open(file_path)
    # scale is the regional scale of the PSDCP
    scale = 2
    lines = chr_file.readlines()
    # Initialize the contact matrix as a zero matrix
    contact_matrix = np.zeros((max_length, max_length))
    for line in lines:
        # bin1, bin2 are the numbers of the two chromosome segments and num is the number of contacts in bin1 and bin2
        bin1, bin2, num = line.split()
        # The first line is the column name, not needed
        if bin1 == "bin1":
            continue
        contact_matrix[int(bin1), int(bin2)] += int(float(num))
        # Because this matrix is symmetric, if bin1 is not equal to bin2, then you need to symmetrize the values
        if bin1 != bin2:
            contact_matrix[int(bin2), int(bin1)] += int(float(num))
    PSDCP = calculate_PSDCP(contact_matrix, max_length, scale)
    return PSDCP


def main():
    types = {'G1':280,'early_S':303,'mid_S':262,'late_S':326}
    # A resolution of 1mbp means that 1M (1,000,000) base pairs are used as the resolution.
    resolution = 1000000
    tps = sorted(types)
    f = open('../mm10.main.nochrM.chrom.sizes')
    index = {}
    lines = f.readlines()
    for line in lines:
        chr_name, length = line.split()
        max_len = int(int(length) / resolution)
        # index[chr_name] stores the number of bins that the chromosome with the number chr_name can be divided into when the resolution is resolution
        index[chr_name] = max_len + 1
        # Why +1, because the division will round down, and the extra piece, too, will make a bin.
    f.seek(0)
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
    print(cell_dict)
    out_path = '../Nagano_features/PSDCP.npy'
    np.save(out_path, cell_dict)


if __name__ == '__main__':
    main()
