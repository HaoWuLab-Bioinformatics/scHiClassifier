import os
if __name__ == '__main__':

    filelist = os.listdir("mid_S")  # 该文件夹下所有的文件（包括文件夹）\
    i = 1
    for file in filelist:
        old = "./mid_S/"+file
        print(old)
        new = "./mid_S/"+file[:6]+str(i)
        print(new)
        i+=1
        os.rename(old, new)