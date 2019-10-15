import os
import random
import shutil
path = '/home/zeyuzhang/PycharmProjects/scienceexam_ner/finaldata_preprocessed'
train_path = '/home/zeyuzhang/PycharmProjects/scienceexam_ner/data/wordtree_ner/train'
valid_path = '/home/zeyuzhang/PycharmProjects/scienceexam_ner/data/wordtree_ner/valid'
test_path = '/home/zeyuzhang/PycharmProjects/scienceexam_ner/data/wordtree_ner/test'
fileNameList = []
for file in os.listdir(path):
    fileNameList.append(file.split('.')[0])
fileNameList = set(fileNameList)
train_list = random.sample(fileNameList, int(0.7*len(fileNameList)))
valid_test_list = [i for i in fileNameList if i not in train_list]
valid_list = random.sample(valid_test_list, int(0.1*len(fileNameList)))
test_list = [i for i in valid_test_list if i not in valid_list]
print(len(train_list))
print(len(valid_list))
print(test_list)
for item in test_list:
    if item in train_list:
        print(item)
for train_file in train_list:
    sourFile_ann = os.path.join(path, train_file+'.ann')
    sourFile_txt = os.path.join(path, train_file+'.txt')
    targetFile_ann = os.path.join(train_path, train_file + '.ann')
    targetFile_txt = os.path.join(train_path, train_file + '.txt')
    shutil.copyfile(sourFile_ann, targetFile_ann)
    shutil.copyfile(sourFile_txt, targetFile_txt)

for valid_file in valid_list:
    sourFile_ann = os.path.join(path, valid_file+'.ann')
    sourFile_txt = os.path.join(path, valid_file+'.txt')
    targetFile_ann = os.path.join(valid_path, valid_file + '.ann')
    targetFile_txt = os.path.join(valid_path, valid_file + '.txt')
    shutil.copyfile(sourFile_ann, targetFile_ann)
    shutil.copyfile(sourFile_txt, targetFile_txt)

for test_file in test_list:
    sourFile_ann = os.path.join(path, test_file+'.ann')
    sourFile_txt = os.path.join(path, test_file+'.txt')
    targetFile_ann = os.path.join(test_path, test_file + '.ann')
    targetFile_txt = os.path.join(test_path, test_file + '.txt')
    shutil.copyfile(sourFile_ann, targetFile_ann)
    shutil.copyfile(sourFile_txt, targetFile_txt)