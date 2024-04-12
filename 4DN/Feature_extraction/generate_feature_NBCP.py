import numpy as np
from methods import sum_matrix, find_contact_list


# This function is used to compute the NBCP features corresponding to a single chromatin
def calculate_NBCP(contact_matrix, max_length):
    NBCP = []
    # sum_matrix is used to calculate the total number of contacts on a single chromatin, since the matrix is symmetric, counting the total number of contacts on a chromatin only requires counting the upper triangular elements + diagonal elements。
    chr_total = sum_matrix(contact_matrix)
    # Row 0 has no upper neighbors, so it only puts itself and its lower neighbors
    con_list1 = find_contact_list(contact_matrix[0])
    con_list2 = find_contact_list(contact_matrix[1])
    sum_list = con_list1 + con_list2
    # List of neighbors corresponding to the current bin
    if len(sum_list) == 0:
        NBCP.append(0)
    else:
        con_pro = sum(sum_list) / chr_total
        NBCP.append(con_pro)

    # There are max_length rows of data, row 0 has no upper neighbor, and max_length-1 rows of data have no lower neighbor, so the range here should be 1 to max_length-2 rows
    for i in range(1, max_length-1):
        con_list1 = find_contact_list(contact_matrix[i])
        con_list2 = find_contact_list(contact_matrix[i-1])
        con_list3 = find_contact_list(contact_matrix[i+1])
        final_list = con_list1 + con_list2 + con_list3
        if len(final_list) == 0:
            NBCP.append(0)
        else:
            con_pro = sum(final_list)/chr_total
            NBCP.append(con_pro)
    # max_length-1 row of data has no lower neighbors, so only put it and its upper neighbors
    con_list1 = find_contact_list(contact_matrix[max_length-2])
    con_list2 = find_contact_list(contact_matrix[max_length-1])
    sum_list = con_list1 + con_list2
    if len(sum_list) == 0:
        NBCP.append(0)
    else:
        con_pro = sum(sum_list) / chr_total
        NBCP.append(con_pro)
    return NBCP

def con_ran(cell_id, type, chr_name, max_length):
    file_path = "../4DN_Data/%s/cell_%s_%s.txt" % (type, str(cell_id), chr_name)
    chr_file = open(file_path)
    lines = chr_file.readlines()
    # Initialize the contact matrix as a zero matrix
    contact_matrix = np.zeros((max_length, max_length))
    for line in lines:
        # bin1, bin2 are the numbers of the two chromosome segments and num is the number of contacts in bin1 and bin2
        bin1, bin2, num = line.split()
        contact_matrix[int(bin1), int(bin2)] += int(float(num))
        # Since this matrix is a symmetric array, you need to symmetrize the values if they are not diagonal elements
        if bin1 != bin2:
            contact_matrix[int(bin2), int(bin1)] += int(float(num))
    nbcp = calculate_NBCP(contact_matrix, max_length)
    return nbcp


def main():
    types = {'GM12878': 1591, 'H1Esc': 1240, 'HAP1': 916, 'HFF': 356, 'IMR90': 12}
    # A resolution of 1mbp means that 1M (1,000,000) base pairs are used as the resolution.
    resolution = 1000000
    tps = sorted(types)
    f = open('../combo_hg19.genomesize')
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
                NBCP = con_ran(cell_id, type, chr, max_len)
                cell_dict[str(cell_number)][chr] = NBCP
    # print(cell_dict)
    out_path = '../4DN_features/NBCP.npy'
    np.save(out_path, cell_dict)


if __name__ == '__main__':
    main()


