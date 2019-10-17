import os
import pandas as pd
import random
def chunk_dataset(file_name):
    token2label = {}
    offset_list = []
    with open(file_name + '.ann', 'r+') as fn:
        for line_ann in fn.readlines():
            if len(line_ann.strip()) == 0:
                continue
            if '----------------------------------------------------------------------------------------------' in line_ann:
                continue
            ann = line_ann.split()
            # print('filename: ',file_name)
            # print('ann: ',ann)
            if (int(ann[2]), int(ann[3])) in token2label.keys():
                token2label[(int(ann[2]), int(ann[3]))].append([ann[1], ' '.join(ann[4:])])
                print(file_name)
                print((int(ann[2]), int(ann[3])))
                #print([ann[1], ' '.join(ann[4:])])
            else:
                token2label[(int(ann[2]), int(ann[3]))] = [[ann[1], ' '.join(ann[4:])]]
                #print([ann[1], ' '.join(ann[4:])])
            offset_list.append((int(ann[2]), int(ann[3])))
    offset_list_set = set(offset_list)
    sorted_offset_list = sorted(offset_list_set, key=lambda x: x[0], reverse=False)
    questionID = []
    with open(file_name + '.txt', 'r+') as ft:
        # text = ft.read()
        # print(text)
        for line_txt in ft.readlines():
            # print(line_txt)
            if '[questionID]' in line_txt:
                continue
            if '----------------------------------------------------------------------------------------------' in line_txt:
                continue
            if line_txt.strip() == 'Question':
                continue
            # print(line_txt)
            if ('[' and ']') in line_txt and '(' not in line_txt and (line_txt.find(']',0)-line_txt.find('[',0))>2:
                splits = line_txt.strip().split(' ')
                questionID.append(splits[0])
    #print('questionID: ', questionID)
    question_location = []
    with open(file_name + '.txt', 'r+') as ft1:
        text = ft1.read()
        final_label = len(text)
        for id in questionID[1:]:
            question_location.append(text.find(id, 0))
        #print(len(question_location))
    pre = 0
    question_location_list = []
    for location in question_location:
        for id, (offset1, offset2) in enumerate(sorted_offset_list):
            if offset1 > location:
                question_location_list.append(sorted_offset_list[pre:id].copy())
                pre = id
                break
        if location == question_location[-1]:
            question_location_list.append(sorted_offset_list[pre:].copy())
    question_location.append(final_label+1)
    questionID_location = {}
    questionID_location_list = {}
    for idx, ID in enumerate(questionID):
        questionID_location[ID] = question_location[idx]
        questionID_location_list[ID] = question_location_list[idx]

    return questionID_location, questionID_location_list, token2label

def get_questionsID(challenge_path, easy_path):
    pd_train_challenge = pd.read_csv(os.path.join(challenge_path,'ARC-Challenge-Train.csv'))
    pd_train_easy = pd.read_csv(os.path.join(easy_path,'ARC-Easy-Train.csv'))
    pd_dev_challenge = pd.read_csv(os.path.join(challenge_path, 'ARC-Challenge-Dev.csv'))
    pd_dev_easy = pd.read_csv(os.path.join(easy_path, 'ARC-Easy-Dev.csv'))
    trainChallengeList = pd_train_challenge['questionID'].tolist()
    trainEasyList = pd_train_easy['questionID'].tolist()
    devChallengeList = pd_dev_challenge['questionID'].tolist()
    devEasyList = pd_dev_easy['questionID'].tolist()
    trainChallengeEasy = trainChallengeList + trainEasyList
    devChallengeEasy = devChallengeList + devEasyList
    random.shuffle(trainChallengeEasy)
    train_questionID_list = trainChallengeEasy[:int(len(trainChallengeEasy)*0.8)]
    valid_questionID_list = trainChallengeEasy[int(len(trainChallengeEasy)*0.8):]
    test_questionID_list = devChallengeEasy
    return train_questionID_list, valid_questionID_list, test_questionID_list

def write_dataset(path, file_name, questionID_location, questionID_location_list, token2label,
                  train_set_qeustionID, valid_set_questionsID, test_set_questionsID):
    with open(os.path.join(path,file_name) + '.txt', 'r+') as f:
        text = f.read()
        pre=0
        for qid in questionID_location.keys():
            #print(qid[1:-1])
            if qid[1:-1] in train_set_qeustionID:
                with open('train/'+str(qid[1:-1])+ '.txt', 'w+') as f_train:
                    #f_train.write(text[pre:questionID_location[qid]])
                    temp_content = text[pre:questionID_location[qid]]
                    for temp_line in temp_content.split('\n'):
                        # if '[questionID]' in temp_line:
                        #     continue
                        if '----------------------------------------------------------------------------------------------' in temp_line:
                            continue
                        # if temp_line.strip() == 'Question':
                        #     continue
                        f_train.write(temp_line)
                        f_train.write('\n')
                    pre = questionID_location[qid]

                with open('train/'+str(qid[1:-1])+ '.txt', 'r+') as fn:
                    text_single = fn.read()
                with open('train/'+str(qid[1:-1])+ '.ann', 'w+') as f_train_ann:
                    pre_ = 0
                    token2label_list = []
                    all_tokens_list = []
                    question_location_list = questionID_location_list[qid]
                    for item in question_location_list:
                        token2label_list.append(token2label[item])
                    for each_token_label in token2label_list:
                        for inside_each_token_label in each_token_label:
                            if text_single.find(inside_each_token_label[1], pre_) == -1:
                                print(file_name)
                                print(qid)
                            all_tokens_list.append((inside_each_token_label[0], text_single.find(inside_each_token_label[1], pre_),
                                                    text_single.find(inside_each_token_label[1], pre_)+len(inside_each_token_label[1]),inside_each_token_label[1]))
                            pre_temp = text_single.find(inside_each_token_label[1], pre_)
                        pre_ = pre_temp
                    for idxx, each in enumerate(all_tokens_list):
                        f_train_ann.write('{}\t{}\t{}\t{}\t{}\n'.format('T'+str(idxx+1),each[0], each[1],each[2],each[3]))
            if qid[1:-1] in valid_set_questionsID:
                with open('valid/'+str(qid[1:-1])+ '.txt', 'w+') as f_valid:
                    temp_content_valid = text[pre:questionID_location[qid]]
                    for temp_line_valid in temp_content_valid.split('\n'):
                        # if '[questionID]' in temp_line_valid:
                        #     continue
                        if '----------------------------------------------------------------------------------------------' in temp_line_valid:
                            continue
                        # if temp_line_valid.strip() == 'Question':
                        #     continue
                        f_valid.write(temp_line_valid)
                        f_valid.write('\n')
                    pre = questionID_location[qid]

                with open('valid/' + str(qid[1:-1]) + '.txt', 'r+') as fn_valid:
                    text_single_valid = fn_valid.read()
                with open('valid/' + str(qid[1:-1]) + '.ann', 'w+') as f_valid_ann:
                    pre_ = 0
                    token2label_list = []
                    all_tokens_list = []
                    question_location_list = questionID_location_list[qid]
                    for item in question_location_list:
                        token2label_list.append(token2label[item])
                    for each_token_label in token2label_list:
                        for inside_each_token_label in each_token_label:
                            if text_single_valid.find(inside_each_token_label[1], pre_) == -1:
                                print(file_name)
                                print(qid)
                            all_tokens_list.append((inside_each_token_label[0],
                                                    text_single_valid.find(inside_each_token_label[1], pre_),
                                                    text_single_valid.find(inside_each_token_label[1],
                                                                     pre_) + len(inside_each_token_label[1]),
                                                    inside_each_token_label[1]))
                            pre_temp = text_single_valid.find(inside_each_token_label[1], pre_)
                        pre_ = pre_temp
                    for idxx, each in enumerate(all_tokens_list):
                        f_valid_ann.write(
                            '{}\t{}\t{}\t{}\t{}\n'.format('T' + str(idxx + 1), each[0], each[1], each[2], each[3]))
            if qid[1:-1] in test_set_questionsID:
                with open('test/'+str(qid[1:-1])+ '.txt', 'w+') as f_test:
                    temp_content_test = text[pre:questionID_location[qid]]
                    for temp_line_test in temp_content_test.split('\n'):
                        # if '[questionID]' in temp_line_test:
                        #     continue
                        if '----------------------------------------------------------------------------------------------' in temp_line_test:
                            continue
                        # if temp_line_test.strip() == 'Question':
                        #     continue
                        f_test.write(temp_line_test)
                        f_test.write('\n')
                    pre = questionID_location[qid]
                #print(qid)
                with open('test/' + str(qid[1:-1]) + '.txt', 'r+') as fn_test:
                    text_single_test = fn_test.read()
                with open('test/' + str(qid[1:-1]) + '.ann', 'w+') as f_test_ann:
                    pre_ = 0
                    token2label_list = []
                    all_tokens_list = []
                    question_location_list = questionID_location_list[qid]
                    for item in question_location_list:
                        token2label_list.append(token2label[item])
                    for each_token_label in token2label_list:
                        for inside_each_token_label in each_token_label:
                            if text_single_test.find(inside_each_token_label[1], pre_) == -1:
                                print(file_name)
                                print(qid)
                            all_tokens_list.append((inside_each_token_label[0],
                                                    text_single_test.find(inside_each_token_label[1], pre_),
                                                    text_single_test.find(inside_each_token_label[1],
                                                                     pre_) + len(inside_each_token_label[1]),
                                                    inside_each_token_label[1]))
                            pre_temp = text_single_test.find(inside_each_token_label[1], pre_)
                        pre_ = pre_temp
                    for idxx, each in enumerate(all_tokens_list):
                        f_test_ann.write(
                            '{}\t{}\t{}\t{}\t{}\n'.format('T' + str(idxx + 1), each[0], each[1], each[2], each[3]))

if __name__ == '__main__':
    path_arc_challenge = '/home/zeyuzhang/PycharmProjects/scienceexam_ner/ARC-V1-Feb2018-2/ARC-Challenge'
    path_arc_easy = '/home/zeyuzhang/PycharmProjects/scienceexam_ner/ARC-V1-Feb2018-2/ARC-Easy'
    train_set_qeustionID, valid_set_questionsID, test_set_questionsID = get_questionsID(path_arc_challenge, path_arc_easy)
    #print(len(train_set_qeustionID))
    path = '/home/zeyuzhang/PycharmProjects/scienceexam_ner/finaldata'
    fileNameList = []
    for file in os.listdir(path):
        fileNameList.append(file.split('.')[0])
    fileNameList = set(fileNameList)
    #fileNameList = ['ARC-science-3325-3350']
    for file_name in fileNameList:
        questionID_location, questionID_location_list, token2label = chunk_dataset(os.path.join(path,file_name))
        # write_dataset(path, file_name, questionID_location, questionID_location_list, token2label,
        #               train_set_qeustionID, valid_set_questionsID, test_set_questionsID)

