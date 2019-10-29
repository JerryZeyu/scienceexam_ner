import pandas as pd
import spacy
import codecs
import os
import glob

def get_ARC_test_data(ARC_Easy_path, ARC_Challenge_path, plain_text_path):
    pd_easy = pd.read_csv(ARC_Easy_path)
    pd_challenge = pd.read_csv(ARC_Challenge_path)
    #df_easy = pd_easy[pd_easy['flags'].str.contains('SUCCESS')].copy()
    #df_challenge = pd_challenge[pd_challenge['flags'].str.contains('SUCCESS')].copy()
    #answer_number = []
    # with open(plain_text_path, 'w+') as file:
    for _, row in pd_easy.iterrows():
        question_answer = row['question']
        if '(3)' in question_answer and '(4)' not in question_answer:
            question = question_answer.split('(1)')[0].strip()
            answerA = question_answer.split('(1)')[1].split('(2)')[0].strip()
            answerB = question_answer.split('(2)')[1].split('(3)')[0].strip()
            answerC = question_answer.split('(3)')[1].strip()
            #print(question_answer)
            answerD = None
            answerE = None
        elif '(1)' and '(2)' and '(3)' in question_answer:
            question = question_answer.split('(1)')[0].strip()
            answerA = question_answer.split('(1)')[1].split('(2)')[0].strip()
            answerB = question_answer.split('(2)')[1].split('(3)')[0].strip()
            answerC = question_answer.split('(3)')[1].split('(4)')[0].strip()
            #print(question_answer)
            answerD = question_answer.split('(4)')[1].strip()
            answerE = None
        elif '(D)' not in question_answer:
            question = question_answer.split('(A)')[0].strip()
            answerA = question_answer.split('(A)')[1].split('(B)')[0].strip()
            answerB = question_answer.split('(B)')[1].split('(C)')[0].strip()
            answerC = question_answer.split('(C)')[1].strip()
            #print(question_answer)
            answerD = None
            answerE = None
        elif '(E)' not in question_answer:
            question = question_answer.split('(A)')[0].strip()
            answerA = question_answer.split('(A)')[1].split('(B)')[0].strip()
            answerB = question_answer.split('(B)')[1].split('(C)')[0].strip()
            answerC = question_answer.split('(C)')[1].split('(D)')[0].strip()
            #print(question_answer)
            answerD = question_answer.split('(D)')[1].strip()
            answerE = None
        else:
            question = question_answer.split('(A)')[0].strip()
            answerA = question_answer.split('(A)')[1].split('(B)')[0].strip()
            answerB = question_answer.split('(B)')[1].split('(C)')[0].strip()
            answerC = question_answer.split('(C)')[1].split('(D)')[0].strip()
            answerD = question_answer.split('(D)')[1].split('(E)')[0].strip()
            answerE = question_answer.split('(E)')[1].strip()
        with open(os.path.join(plain_text_path, row['questionID'])+'.txt','w+') as file:
            file.write(question + '\n')
            file.write(answerA + '\n')
            file.write(answerB + '\n')
            file.write(answerC + '\n')
            if answerD is not None:
                file.write(answerD + '\n')
            if answerE is not None:
                file.write(answerE + '\n')
            #file.write('\n')
    for _, row in pd_challenge.iterrows():
        question_answer = row['question']
        if '(3)' in question_answer and '(4)' not in question_answer:
            question = question_answer.split('(1)')[0].strip()
            answerA = question_answer.split('(1)')[1].split('(2)')[0].strip()
            answerB = question_answer.split('(2)')[1].split('(3)')[0].strip()
            answerC = question_answer.split('(3)')[1].strip()
            #print(question_answer)
            answerD = None
            answerE = None
        elif '(1)' and '(2)' and '(3)' in question_answer:
            question = question_answer.split('(1)')[0].strip()
            answerA = question_answer.split('(1)')[1].split('(2)')[0].strip()
            answerB = question_answer.split('(2)')[1].split('(3)')[0].strip()
            answerC = question_answer.split('(3)')[1].split('(4)')[0].strip()
            #print(question_answer)
            answerD = question_answer.split('(4)')[1].strip()
            answerE = None
        elif '(D)' not in question_answer:
            question = question_answer.split('(A)')[0].strip()
            answerA = question_answer.split('(A)')[1].split('(B)')[0].strip()
            answerB = question_answer.split('(B)')[1].split('(C)')[0].strip()
            answerC = question_answer.split('(C)')[1].strip()
            #print(question_answer)
            answerD = None
            answerE = None
        elif '(E)' not in question_answer:
            question = question_answer.split('(A)')[0].strip()
            answerA = question_answer.split('(A)')[1].split('(B)')[0].strip()
            answerB = question_answer.split('(B)')[1].split('(C)')[0].strip()
            answerC = question_answer.split('(C)')[1].split('(D)')[0].strip()
            #print(question_answer)
            answerD = question_answer.split('(D)')[1].strip()
            answerE = None
        else:
            question = question_answer.split('(A)')[0].strip()
            answerA = question_answer.split('(A)')[1].split('(B)')[0].strip()
            answerB = question_answer.split('(B)')[1].split('(C)')[0].strip()
            answerC = question_answer.split('(C)')[1].split('(D)')[0].strip()
            answerD = question_answer.split('(D)')[1].split('(E)')[0].strip()
            answerE = question_answer.split('(E)')[1].strip()

        with open(os.path.join(plain_text_path, row['questionID'])+'.txt', 'w+') as file:
            file.write(question + '\n')
            file.write(answerA + '\n')
            file.write(answerB + '\n')
            file.write(answerC + '\n')
            if answerD is not None:
                file.write(answerD + '\n')
            if answerE is not None:
                file.write(answerE + '\n')
            #file.write('\n')

def get_start_and_end_offset_of_token_from_spacy(temp, token):
    start = temp + token.idx
    #print('token idx: ', token.idx)
    end = start + len(token)
    return start, end

def get_sentences_and_tokens_from_spacy(plain_text_path, spacy_nlp):
    with codecs.open(plain_text_path, 'r', 'UTF-8') as f:
        text = f.read()
    sentences = []
    temp = 0
    for line in text.split('\n')[:-1]:
        #print(line)
        document = spacy_nlp(line)

        for span in document.sents:
            sentence = [document[i] for i in range(span.start, span.end)]
        sentence_tokens = []
        for token in sentence:
            token_dict = {}
            token_dict['start'], token_dict['end'] = get_start_and_end_offset_of_token_from_spacy(temp, token)
            token_dict['text'] = text[token_dict['start']:token_dict['end']]

            if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
                continue
            # Make sure that the token text does not contain any space
            if len(token_dict['text'].split(' ')) != 1:
                print(plain_text_path)
                print(token_dict['text'])
                print(
                    "WARNING: the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(
                        token_dict['text'],
                        token_dict['text'].replace(' ', '-')))
                token_dict['text'] = token_dict['text'].replace(' ', '-')
            sentence_tokens.append(token_dict)
            if token == sentence[-1]:
                temp = token_dict['end'] + 1
        sentences.append(sentence_tokens)
    return sentences


if __name__ == '__main__':
    ARC_Easy_Test_path = '/home/zeyuzhang/PycharmProjects/scienceexam_ner/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Test.csv'
    ARC_Challenge_Test_path = '/home/zeyuzhang/PycharmProjects/scienceexam_ner/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Test.csv'
    plain_text_path = '/home/zeyuzhang/PycharmProjects/scienceexam_ner/ARC-V1-Feb2018-2/ARC_text_questions'
    output_filepath = '/home/zeyuzhang/PycharmProjects/scienceexam_ner/ARC-V1-Feb2018-2/ARC_test_spacy.txt'
    get_ARC_test_data(ARC_Easy_Test_path,ARC_Challenge_Test_path, plain_text_path)
    spacy_nlp = spacy.load('en')
    text_filepaths = sorted(glob.glob(os.path.join(plain_text_path, '*.txt')))
    output_file = codecs.open(output_filepath, 'w', 'utf-8')
    for text_filepath in text_filepaths:
        base_filename = os.path.splitext(os.path.basename(text_filepath))[0]
        sentences = get_sentences_and_tokens_from_spacy(text_filepath, spacy_nlp)
        for sentence in sentences:
            for token in sentence:
                output_file.write(
                    '{0} {1}\n'.format(token['text'], base_filename))
            output_file.write('\n')

    output_file.close()
    print('Done.')
    del spacy_nlp