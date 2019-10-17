import os
path = '/home/zeyuzhang/PycharmProjects/scienceexam_ner/finaldata'
def get_brat_label(path):
    label_list = []
    for file in os.listdir(path):
        if file.split('.')[-1]=='ann':
            with open(os.path.join(path, file), 'r+') as f:
                for line in f.readlines():
                    anno = line.split()
                    # print('anno: ',anno)
                    if anno == []:
                        continue
                    id_anno = anno[0]
                    # parse entity
                    if id_anno[0] == 'T':
                        label_list.append(anno[1])
    label_set = set(label_list)
    print(len(label_set))
    print(label_set)
def get_conll_bio_label(path):
    label_list = []
    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r+') as f:
            for line in f.readlines():
                if len(line) == 0 or line[0]=='\n':
                    continue
                splits = line.split(' ')
                label_list.append(splits[-1][:-1])
    label_set = set(label_list)
    if 'O' in label_set:
        print(1)
    print(len(label_set))
    print(label_set)
if __name__ == '__main__':
    brat_path = '/home/zeyuzhang/PycharmProjects/scienceexam_ner/finaldata'
    conll_bio_path = '/home/zeyuzhang/PycharmProjects/scienceexam_ner/bert_data'
    #brat_label_set = get_brat_label(path)
    conll_bio_label_set = get_conll_bio_label(conll_bio_path)