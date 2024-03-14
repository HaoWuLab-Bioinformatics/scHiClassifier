
if __name__ == '__main__':
    cdx1 = []
    cdx2 = []
    cdx3 = []
    cdx4 = []
    f = open("./new_cell_inf.txt", 'r')
    lines = f.readlines()
    for line in lines:
        b1, b2 = line.split()
        # print(b1)
        b1 = b1.replace('_','.',1)
        b1 = b1 + "_reads"
        # print(b1)
        if "1CDX1" in b1:
            cdx1.append(b1)
        elif "1CDX2" in b1:
            cdx2.append(b1)
        elif "1CDX3" in b1:
            cdx3.append(b1)
        elif "1CDX4" in b1:
            cdx4.append(b1)
    f.close
    print(cdx1)
    print(cdx2)
    print(cdx3)
    print(cdx4)
