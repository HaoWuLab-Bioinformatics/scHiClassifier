import os
if __name__ == '__main__':

    filelist = os.listdir("late_S")  # 该文件夹下所有的文件（包括文件夹）\
    for file in filelist:
        old = "./late_S/"+file
        new = "./late_S/late_S_"+file[6:]
        os.rename(old, new)