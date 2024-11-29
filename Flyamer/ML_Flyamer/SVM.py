import xlsxwriter
import numpy as np
from sklearn.model_selection import train_test_split
import random,os, torch

from Machine_learning import machinelearning_SVM


def seed_torch(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    types = {'oocytes': 114, 'ZygM': 32, 'ZygP': 32}

    Y = []
    X = []
    resolution = 1000000
    tps = sorted(types)
    f = open('./mm10.main.nochrM.chrom.sizes')
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
    cell_number = 0
    for type in tps:
        for c_num in range(types[type]):
            cell_number += 1
            X.append(str(cell_number))
            if type == 'oocytes':
                Y.append(0)
            elif type == 'ZygM':
                Y.append(1)
            elif type == 'ZygP':
                Y.append(2)
    X = np.array(X).reshape(cell_number, 1)
    Y = np.array(Y)


    # for testseed in [10, 541, 800, 1654, 8666]:
    random.seed(2023)
    testseed_list = [735715, 884807, 640905, 598764, 978791, 353617,
                     664118, 104908, 822596, 129898, 682656, 318109,
                     730637, 557627, 753260, 553225, 315788, 186666,
                     243967, 668572, 17889, 380684, 132226, 739264,
                     890591, 209523, 162059, 503280, 557513, 527263,
                     68755, 79616, 831588, 600368, 777091, 645281, 668028,
                     755110, 111881, 609102, 72548, 147381, 558246, 221929,
                     304937, 857099, 751662, 603207, 781213, 392217]

    zuhes = {1: ['nbcp3_with0', 'sbcp', 'nsicp2', 'ssicp'],
             }
    for zuhe in zuhes.values():
        # 读取特征集
        f1 = zuhe[0]
        f1_path = "../Flyamer_features/%s.npy" % f1
        f1_Data = np.load(f1_path, allow_pickle=True).item()  # 返回的长度为细胞数量
        f2 = zuhe[1]
        f2_path = "../Flyamer_features/%s.npy" % f2
        f2_Data = np.load(f2_path, allow_pickle=True).item()  # 返回的长度为细胞数量
        f3 = zuhe[2]
        f3_path = "../Flyamer_features/%s.npy" % f3
        f3_Data = np.load(f3_path, allow_pickle=True).item()  # 返回的长度为细胞数量
        f4 = zuhe[3]
        f4_path = "../Flyamer_features/%s.npy" % f4
        f4_Data = np.load(f4_path, allow_pickle=True).item()  # 返回的长度为细胞数量

        file = './test_result/SVM.xlsx'
        workbook = xlsxwriter.Workbook(file)
        worksheet1 = workbook.add_worksheet('model')
        worksheet1.write(0, 0, 'test_seed')
        worksheet1.write(0, 1, 'acc')
        worksheet1.write(0, 2, 'micro_F1')
        worksheet1.write(0, 3, 'macro_F1')
        worksheet1.write(0, 4, 'micro_Precision')
        worksheet1.write(0, 5, 'macro_Precision')
        worksheet1.write(0, 6, 'bacc')
        worksheet1.write(0, 7, 'mcc')
        worksheet1.write(0, 8, 'micro_Recall')
        worksheet1.write(0, 9, 'macro_Recall')

        row = 0
        for testseed in testseed_list:
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=testseed, stratify=Y)
            row += 1
            test_acc, micro_F1, macro_F1, micro_Precision, macro_Precision, micro_Recall, macro_Recall, bacc, mcc = machinelearning_SVM(f1_Data, f2_Data, f3_Data, f4_Data, x_train, y_train, x_test, y_test)
            print("=============")
            print("testseed:", testseed)
            print("row:",row)
            print("acc:", test_acc)
            print("bacc:", bacc)
            worksheet1.write(row, 0, testseed)
            worksheet1.write(row, 1, test_acc)
            worksheet1.write(row, 2, micro_F1)
            worksheet1.write(row, 3, macro_F1)
            worksheet1.write(row, 4, micro_Precision)
            worksheet1.write(row, 5, macro_Precision)
            worksheet1.write(row, 6, bacc)
            worksheet1.write(row, 7, mcc)
            worksheet1.write(row, 8, micro_Recall)
            worksheet1.write(row, 9, macro_Recall)
        workbook.close()

if __name__ == '__main__':
    main()