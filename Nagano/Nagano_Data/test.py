import os
import shutil
if __name__ == '__main__':
    # shutil.rmtree("../b")
    file_dir = "late_S"
    p = []
    for root, dirs, files in os.walk(file_dir, topdown=False):
        p.append(root)
    for i in range(len(p)):
        p[i] = p[i][8:]
    print(p)

    cdx1 = []
    cdx2 = []
    cdx3 = []
    cdx4 = []
    f = open("new_cell_inf.txt", 'r')
    lines = f.readlines()
    for line in lines:
        b1, b2 = line.split()
        # print(b1)
        b1 = b1.replace('_', '.', 1)
        b1 = b1 + "_reads"
        # print(b1)
        if "G1" in b1:
            cdx1.append(b1)
        elif "early_S" in b1:
            cdx2.append(b1)
        elif "mid_S" in b1:
            cdx3.append(b1)
        elif "late_S" in b1:
            cdx4.append(b1)
    f.close


    for i in p:
        if i not in cdx4 and i!="":
            print(i+"mmmmm")
            shutil.rmtree("./late_S/%s" % i)
