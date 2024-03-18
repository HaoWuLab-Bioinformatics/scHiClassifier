import pandas as pd
import xlsxwriter
from scipy import stats
import numpy as np
from method import CNN_1D_montage
from sklearn.model_selection import train_test_split
import random,os, torch
from sklearn.metrics import f1_score, precision_score, balanced_accuracy_score,recall_score,matthews_corrcoef
from collections import Counter

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
    # alpha用于计算focal_loss
    # 每个类别对应的alpha=该类别出现频率的倒数
    alpha = []
    for value in types.values():
        ds = 1 / value
        alpha.append(ds)
    # print(alpha)

    Y = []
    X = []
    resolution = 1000000
    tps = sorted(types)
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

    num_features = 4
    num_types = 3
    key_size, query_size, value_size = 2660, 2660, 2660
    Con_layer = [2, 2, 2, 2]
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


    zuhes = {1: ['NBCP', 'SBCP', 'PSDCP', 'SSDCP'] }
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

        for epoch in [150]:
            file = './train&test_result/train&test_result.xlsx'
            workbook = xlsxwriter.Workbook(file)
            worksheet1 = workbook.add_worksheet('111')
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
            worksheet1.write(0, 10, 'counter')
            row = 0
            for testseed in testseed_list:
                x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=testseed, stratify=Y)
                row += 1
                num_hiddens = 2660
                norm_shape = [num_features, num_hiddens]
                ffn_num_input = num_hiddens
                ffn_num_hiddens = 2660
                num_heads = 20
                num_layers = 1
                dp = 0.1
                use_bias = False
                kernel_size = 7
                out_channels = 64
                out_feature = 1330
                linear_layer = 1
                lr = 0.0001
                gamma = 1
                batch_size = 32
                model_para = [num_features, key_size, query_size,
                              value_size,
                              num_hiddens, norm_shape, ffn_num_input,
                              ffn_num_hiddens, num_heads, num_layers, dp,
                              use_bias,
                              kernel_size, out_channels, out_feature,
                              Con_layer,
                              linear_layer, num_types]
                test_acc, test_label, real_label, test_result_matrix = CNN_1D_montage(f1_Data, f2_Data, f3_Data, f4_Data,
                                                                                      x_train,
                                                                                      y_train,
                                                                                      x_test,
                                                                                      y_test,
                                                                                      lr,
                                                                                      model_para,
                                                                                      alpha,
                                                                                      gamma, batch_size,epoch)
                label_count = []
                for i, j in zip(test_label, real_label):
                    if i == j:
                        label_count.append(i)
                micro_F1 = f1_score(real_label, test_label, average='micro')
                macro_F1 = f1_score(real_label, test_label, average='macro')
                micro_Precision = precision_score(real_label, test_label,
                                                  average='micro')
                macro_Precision = precision_score(real_label, test_label,
                                                  average='macro')
                micro_Recall = recall_score(real_label, test_label,
                                            average='micro')
                macro_Recall = recall_score(real_label, test_label,
                                            average='macro')
                bacc = balanced_accuracy_score(real_label, test_label)
                mcc = matthews_corrcoef(real_label, test_label)
                print("=============")
                print("testseed:", testseed)
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
                worksheet1.write(row, 10, str(Counter(label_count)))
            workbook.close()

if __name__ == '__main__':
    main()