import scipy.io as scio
dataFile = 'D:\\TCSS555\\project\\training\\wiki_crop\\wiki.mat'
data = scio.loadmat(dataFile)

a = data['wiki'][0][0]
lable =[]
lable_sex =[]
filenames = []
genders = []
for i in range(len(a[2][0])):
    name = a[2][0][i]
    sex = a[3][0][i]
    print(type(str(name[0])))
    print(str(name[0]).split("/")[1])

    # name = list(name)
    # filenames.append(name[0])
    # genders.append(sex)

# print(filenames[0])

# for i in lable:
#     print(i[0])
#     print(i[1])
#     with open("lable2.txt", "a") as f:  # 格式化字符串还能这么用！
#         f.write(i[0])
#         f.write('   ')
#         f.write(i[1])
#         f.write('\n')