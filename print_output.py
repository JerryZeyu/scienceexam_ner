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
        # label = []
        for line in file.readlines():
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0:
                    data.append(sentence)
                    sentence = []
                    # label = []
                continue
            splits = line.split(' ')
            sentence.append(splits[0])
            # label.append([i.strip() for i in splits[4:]])
        if len(sentence) > 0:
            data.append(sentence)
            sentence = []
            # label = []
    return data

def token_predition_write(y_pred, sentence_tokens, output_path):
    with open(output_path, 'w+') as file:
        for idx, sentence in enumerate(sentence_tokens):
            for tokenid, token in enumerate(sentence):
                file.write('{}\t{}\n'.format(token, '\t'.join(y_pred[idx][tokenid])))
            file.write('\n')
if __name__ == '__main__':

    y_pred = pickle_load_large_file('y_pred_dev.pkl')
    print(len(y_pred))
    sentence_tokens = get_sentence_tokens('/home/zeyuzhang/PycharmProjects/scienceexam_ner/bert_data/valid_spacy.txt')
    print(len(sentence_tokens))
    output_path = './token_prediction/dev_token_prediction.txt'
    token_predition_write(y_pred, sentence_tokens, output_path)