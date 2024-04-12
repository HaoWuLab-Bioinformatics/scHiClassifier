import numpy as np
from methods import  sum_matrix, smooth_mean


# This function is used to compute the SSDCP features corresponding to a single piece of chromatin
def calculate_SSDCP(contact_matrix, max_length, scale):
    matrix1 = contact_matrix
    # len(contact_matrix) is the number of rows in the contact matrix
    for i in range(len(contact_matrix)):
        # The list space_neighbour_contact_infor is used to store the contact information of the spatially neighboring bin (linear neighbor + interaction i.e. spatially neighboring) of the current bin.
        space_neighbour_contact_infor = []
        # The list index1 is used to hold the numbers of the bins that are linearly adjacent to the current bin.
        index1 = []
        # Put in the current bin's own contact info first #
        space_neighbour_contact_infor.append(contact_matrix[i, :])

        # This loop is used to obtain contact information for the bins (left and right neighbors) that are linearly adjacent to the current bin
        # The values of m are i-1, i, i+1.
        for m in range(i-1, i+2):
            # Since the array starts at element 0 (i.e., the contact matrix starts at row 0), m cannot be <0
            # Since the array goes up to element len(matrix)-1 (i.e., the contact matrix goes up to row len(matrix)-1), m cannot be > len(matrix)-1.
            # Because outside of this loop, the previous line of code has already taken the contact information for chromosome segment i, it cannot be taken again here, so m cannot be equal to i.
            if m < 0 or m > len(contact_matrix) - 1 or m == i:
                continue
            space_neighbour_contact_infor.append(contact_matrix[m, :])
            # Record the number of the bin that is linearly adjacent to the current bin
            index1.append(m)

        # The function of this loop is to obtain contact information for a bin that interacts with (i.e., is spatially adjacent to) the current bin.
        for n in range(len(contact_matrix[0])):
            # Skip if n is already a linear neighbor of the current bin
            if n in index1:
                continue
            # matrix[i, n]! = 0 means that the current bin interacts with bin_n, i.e., it is spatially adjacent.
            # And since the contact information for the current bin has already been fetched previously, it cannot be fetched again here, so i!=n.
            if contact_matrix[i, n] != 0 and i != n:
                space_neighbour_contact_infor.append(contact_matrix[n, :])
        # Replace the original bin contact information with the smoothed bin contact information.
        matrix1[i, :] = smooth_mean(space_neighbour_contact_infor)
     # Why do the following three lines of code to generate final_matrix?
    # A: Because matrix1 it is a chromosome contact matrix, the upper and lower triangles must be symmetric, that is, matrix1[i,j] and matrix1[j,i] must be consistent.
    # But since the spatial neighborhood bin is different for each bin, the matrix1 matrix we generate row by row it is not symmetric.
    # To keep the matrix symmetric, we map the upper triangles to the lower triangles. (In fact, it is possible to map the lower triangles to the upper triangles).
    # m is to take all the elements of the upper triangle of the square matrix matrix1 (excluding the main diagonal) and assign all the elements of the main diagonal and the lower triangle to zero.。
    m = np.triu(matrix1, 1)
    n = m.T
    # np.diag(matrix1) extracts the elements on the main diagonal of the square matrix1.
    # np.diag(np.diag(matrix1)) will use the elements on the main diagonal of square matrix1 extracted by np.diag(matrix1) as the main diagonal to construct a square matrix with all elements except the main diagonal as zero.
    smoothed_matrix = m + n + np.diag(np.diag(matrix1))
    # ------------------------------------------------------------
    # Calculate SSDCP from smoothed chromatin contact matrix

    # Amplify the contact matrix
    new_matrix = np.zeros((max_length + 2 * scale, max_length + 2 * scale))
    SSDCP = []
    # sum_matrix is used to calculate the total number of contacts on a single chromatin, since the matrix is symmetric, counting the total number of contacts on a chromatin only requires counting the upper triangular elements + diagonal elements.
    chr_total = sum_matrix(smoothed_matrix)
    for i in range(max_length):
        for j in range(max_length):
            # Put the data from the original contact matrix into new_matrix, augmented by 0
            new_matrix[i + scale, j + scale] = smoothed_matrix[i, j]

    for i in range(max_length):
        bin = i + scale
        # Suppose a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
        # Then a[1:4,1:4] doesn't actually take 4 and returns
        # [[ 1 2 3]]
        # [ 5 6 7]
        # [ 9 10 11]]
        # That's why there's an extra 1 in the next line #
        a = sum_matrix(new_matrix[bin - scale:bin + scale + 1, bin - scale:bin + scale + 1])  # 3 * 3 matrices at a time
        if a == 0:
            SSDCP.append(float(0))
        else:
            SSDCP.append(float(a / chr_total))
    return SSDCP

def con_ran(cell_id, type, chr_name, max_length):
    file_path = "../Lee_Data/%s/cell_%s_%s.txt" % (type, str(cell_id), chr_name)
    chr_file = open(file_path)
    # scale is the regional scale of the SSDCP
    scale = 1
    lines = chr_file.readlines()
    # Initialize the contact matrix as a zero matrix
    contact_matrix = np.zeros((max_length, max_length))
    for line in lines:
        # bin1, bin2 are the numbers of the two chromosome segments and num is the number of contacts in bin1 and bin2
        bin1, bin2, num = line.split()
        # This is different from other datasets because other datasets if chromosome 1 cuts 250 bins then those bins are numbered 0 to 249 but this dataset is 1 to 250
        bin1 = int(bin1) - 1
        bin2 = int(bin2) - 1

        contact_matrix[int(bin1), int(bin2)] += int(float(num))
        # Because this matrix is symmetric, if bin1 is not equal to bin2, then you need to symmetrize the values
        if bin1 != bin2:
            contact_matrix[int(bin2), int(bin1)] += int(float(num))
    SSDCP = calculate_SSDCP(contact_matrix, max_length, scale)
    return SSDCP


def main():
    types = {'Astro': 449, 'Endo': 204, 'L4': 131, 'L5': 180, 'L6': 86, 'L23': 551, 'MG': 422, 'Ndnf': 144,
             'NN1': 100, 'ODC': 1243, 'OPC': 203, 'Pvalb': 134, 'Sst': 216, 'Vip': 171}
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
                SSDCP = con_ran(cell_id, type, chr, max_len)
                cell_dict[str(cell_number)][chr] = SSDCP
    out_path = '../Lee_features/SSDCP.npy'
    np.save(out_path, cell_dict)


if __name__ == '__main__':
    main()

