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
    types = {'GM12878': 1591, 'H1Esc': 1240, 'HAP1': 916, 'HFF': 356, 'IMR90': 12}
    Y = []
    X = []
    resolution = 1000000
    tps = sorted(types)
    f = open('./combo_hg19.genomesize')
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
            if type == 'GM12878':
                Y.append(0)
            elif type == 'H1Esc':
                Y.append(1)
            elif type == 'HAP1':
                Y.append(2)
            elif type == 'HFF':
                Y.append(3)
            elif type == 'IMR90':
                Y.append(4)
    X = np.array(X).reshape(cell_number, 1)
    Y = np.array(Y)

    random.seed(2023)
    testseed_list = [884807, 408141, 640905, 353617, 129898, 682656, 318109, 879136,
                     730637, 557627, 753260, 553225, 972593, 185272, 221168, 380684,
                     820294, 209523, 162059, 503280, 468642, 557513, 527263, 522696, 32041,
                     668028, 111881, 503384, 677611, 631852, 221929, 304937, 914569, 282637,
                     664421, 751662, 124891, 954318, 917249, 5748, 2624, 5008, 820, 6427, 5334,
                     2486, 5885, 3662, 2975, 576]

    zuhes = {1: ['nbcp3_with0', 'sbcp', 'nsicp2', 'ssicp']
             }
    for zuhe in zuhes.values():
        # 读取特征集
        f1 = zuhe[0]
        f1_path = "../4DN_features/%s.npy" % f1
        f1_Data = np.load(f1_path, allow_pickle=True).item()  # 返回的长度为细胞数量
        f2 = zuhe[1]
        f2_path = "../4DN_features/%s.npy" % f2
        f2_Data = np.load(f2_path, allow_pickle=True).item()  # 返回的长度为细胞数量
        f3 = zuhe[2]
        f3_path = "../4DN_features/%s.npy" % f3
        f3_Data = np.load(f3_path, allow_pickle=True).item()  # 返回的长度为细胞数量
        f4 = zuhe[3]
        f4_path = "../4DN_features/%s.npy" % f4
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
            test_acc, micro_F1, macro_F1, micro_Precision, macro_Precision, micro_Recall, macro_Recall, bacc, mcc = machinelearning_SVM_gai(
                f1_Data, f2_Data, f3_Data, f4_Data, x_train, y_train, x_test, y_test)
            print("=============")
            print("testseed:", testseed)
            print("row:", row)
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