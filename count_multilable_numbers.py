import glob
import os
path = '/home/zeyuzhang/PycharmProjects/scienceexam_ner/finaldata'
ann_filepaths = sorted(glob.glob(os.path.join(path, '*.ann')))
print(ann_filepaths)
count_number = 0
total_number = 0
total_same_label = []
for ann_filepath in ann_filepaths:
    all_tokens = []
    with open(ann_filepath, 'r+') as f:
        for line in f.readlines():
            if len(line.strip()) == 0:
                continue
            if '----------------------------------------------------------------------------------------------' in line:
                continue
            ann = line.split()
            all_tokens.append((int(ann[2]), int(ann[3]), ann[1],' '.join(ann[4:])))
    all_tokens_set = set(all_tokens)

    # print(len(all_tokens))
    # print(len(all_tokens_set))
    all_tokens_set = list(all_tokens_set)
    total_number += len(all_tokens_set)
    #print(all_tokens_set)
    for i in range(len(all_tokens_set)):
        for k in range(i+1, len(all_tokens_set)):
            if all_tokens_set[i][0] == all_tokens_set[k][0]:
                # print(all_tokens_set[i])
                # print(all_tokens_set[k])
                total_same_label.append(tuple(sorted((all_tokens_set[i][2], all_tokens_set[k][2]))))
                count_number +=1
            elif (all_tokens_set[i][0] < all_tokens_set[k][0]<all_tokens_set[k][1]<=all_tokens_set[i][1]) or (all_tokens_set[k][0] < all_tokens_set[i][0] < all_tokens_set[i][1] <= all_tokens_set[k][1]) or \
                    (all_tokens_set[i][0] < all_tokens_set[k][0] < all_tokens_set[i][1] <= all_tokens_set[k][1]) or \
                (all_tokens_set[k][0] < all_tokens_set[i][0] < all_tokens_set[k][1] <= all_tokens_set[i][1]):
                count_number+=1
                # print(all_tokens_set[i])
                # print(all_tokens_set[k])
                total_same_label.append(tuple(sorted((all_tokens_set[i][2], all_tokens_set[k][2]))))
print(total_same_label)
total_same_label_set = set(total_same_label)
print('label number: ', len(total_same_label_set))
print('count number: ', count_number)
print('total number: ', total_number)

print(count_number/(total_number*1.0))