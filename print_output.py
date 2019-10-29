import os
import numpy as np
import pickle
def pickle_load_large_file(filepath):
    max_bytes = 2**31 - 1
    input_size = os.path.getsize(filepath)
    bytes_in = bytearray(0)
    with open(filepath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    obj = pickle.loads(bytes_in)
    return obj
def get_sentence_tokens(filepath):
    with open(filepath, 'r+') as file:
        data = []
        sentence = []
        label = []
        for line in file.readlines():
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    data.append((sentence,label))
                    sentence = []
                    label = []
                continue
            splits = line.split(' ')
            sentence.append(splits[0])
            label.append([i.strip() for i in splits[4:]])
        if len(sentence) > 0:
            data.append((sentence,label))
            sentence = []
            label = []
    return data

def token_predition_write(y_pred, sentence_tokens_labels, output_path, total_precision):
    with open(output_path, 'w+') as file:
        for idx, (sentence,label) in enumerate(sentence_tokens_labels):
            for tokenid, token in enumerate(sentence):
                file.write('{}\t{}\t{}\t{}\n'.format(token, total_precision[idx][tokenid], ' '.join(y_pred[idx][tokenid]), ' '.join(['GOLD-'+item for item in label[tokenid]])))
            file.write('\n')

def calculate_precision(sentence_tokens_labels, y_pred):
    total_precision = []
    for idx, (sentence, label) in enumerate(sentence_tokens_labels):
        sentence_precision = []
        for labelsid, labels in enumerate(label):
            labels_set = set(labels)
            y_pred_set = set(y_pred[idx][labelsid])
            if labels_set == y_pred_set:
                sentence_precision.append(1)
            else:
                sentence_precision.append(0)
        total_precision.append(sentence_precision)
    return total_precision
if __name__ == '__main__':

    y_pred = pickle_load_large_file('y_pred_dev.pkl')
    print(len(y_pred))
    sentence_tokens_labels = get_sentence_tokens('/home/zeyuzhang/PycharmProjects/scienceexam_ner/bert_data/valid_spacy.txt')
    print(len(sentence_tokens_labels))
    output_path = './entity_prediction/dev_entity_prediction.txt'
    total_precision = calculate_precision(sentence_tokens_labels, y_pred)
    token_predition_write(y_pred, sentence_tokens_labels, output_path, total_precision)