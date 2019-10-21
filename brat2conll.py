# -*- coding: utf-8 -*-
import codecs
import glob
import json
import os
import spacy

def replace_unicode_whitespaces_with_ascii_whitespace(string):
    return ' '.join(string.split())

def get_start_and_end_offset_of_token_from_spacy(temp, token):
    start = temp + token.idx
    end = start + len(token)
    return start, end

def get_sentences_and_tokens_from_spacy(text, spacy_nlp):
    sentences = []
    temp = 0
    for line in text.split('\n')[:-1]:
        # print(line)
        document = spacy_nlp(line)

        for span in document.sents:
            sentence = [document[i] for i in range(span.start, span.end)]
        # print('sentence: ', sentence)
        sentence_tokens = []
        for token in sentence:
            # print('token: ',token)
            token_dict = {}
            # print('temp: ', temp)
            token_dict['start'], token_dict['end'] = get_start_and_end_offset_of_token_from_spacy(temp, token)
            # print(token_dict['start'], token_dict['end'])
            token_dict['text'] = text[token_dict['start']:token_dict['end']]
            # print('token_dict text: ', token_dict['text'])
            if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
                continue
            # Make sure that the token text does not contain any space
            if len(token_dict['text'].split(' ')) != 1:
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


def get_entities_from_brat(text_filepath, annotation_filepath, verbose=False):
    # load text
    with codecs.open(text_filepath, 'r', 'UTF-8') as f:
        text = f.read()
    if verbose: print("\ntext:\n{0}\n".format(text))

    # parse annotation file
    entities = []
    with codecs.open(annotation_filepath, 'r', 'UTF-8') as f:
        for line in f.read().splitlines():
            anno = line.split()
            id_anno = anno[0]
            # parse entity
            if id_anno[0] == 'T':
                entity = {}
                entity['id'] = id_anno
                entity['type'] = anno[1]
                entity['start'] = int(anno[2])
                entity['end'] = int(anno[3])
                entity['text'] = ' '.join(anno[4:])
                if verbose:
                    print("entity: {0}".format(entity))
                # Check compatibility between brat text and anootation
                if replace_unicode_whitespaces_with_ascii_whitespace(text[entity['start']:entity['end']]) != \
                        replace_unicode_whitespaces_with_ascii_whitespace(entity['text']):
                    print("Warning: brat text and annotation do not match.")
                    print('text_filepath: ', text_filepath)
                    print('annotation_filepath: ', annotation_filepath)
                    print('entity ID: ', entity['id'])
                    print('line: ', line)
                    print('anno: ', anno)
                    print("\ttext: {0}".format(text[entity['start']:entity['end']]))
                    print("\tanno: {0}".format(entity['text']))
                    # entity['end']=entity['end']-95
                # add to entitys data
                entities.append(entity)
    if verbose: print("\n\n")

    return text, entities


def brat_to_conll(input_folder, output_filepath, tokenizer, language):
    if tokenizer == 'spacy':
        spacy_nlp = spacy.load(language)
    else:
        raise ValueError("tokenizer should be 'spacy'.")
    verbose = False
    dataset_type = os.path.basename(input_folder)
    print("Formatting {0} set from BRAT to CONLL... ".format(dataset_type), end='')
    text_filepaths = sorted(glob.glob(os.path.join(input_folder, '*.txt')))
    output_file = codecs.open(output_filepath, 'w', 'utf-8')
    for text_filepath in text_filepaths:
        base_filename = os.path.splitext(os.path.basename(text_filepath))[0]
        annotation_filepath = os.path.join(os.path.dirname(text_filepath), base_filename + '.ann')
        # create annotation file if it does not exist
        if not os.path.exists(annotation_filepath):
            codecs.open(annotation_filepath, 'w', 'UTF-8').close()
        text, entities = get_entities_from_brat(text_filepath, annotation_filepath)
        entities = sorted(entities, key=lambda entity: entity["start"])
        # temp_entities = [entity['start'] for entity in entities]
        # temp_entities_set = set(temp_entities)
        # if len(temp_entities) != temp_entities_set:
        #     print(1)

        if tokenizer == 'spacy':
            sentences = get_sentences_and_tokens_from_spacy(text, spacy_nlp)
        for sentence in sentences:
            for token in sentence:
                token['label'] = []
                for entity in entities:
                    if entity['start'] == token['start'] and token['end']<= entity['end']:
                        token['label'].append('B-{0}'.format(entity['type'].replace('-','_')))
                    elif entity['start'] < token['start'] and token['end']<= entity['end']:
                        token['label'].append('I-{0}'.format(entity['type'].replace('-','_')))
                    elif token['end'] < entity['start']:
                        break
                if token['label'] == []:
                    token['label'] = ['O']

                if len(entities) == 0:
                    entity = {'end': 0}
                if verbose: print(
                    '{0} {1} {2} {3} {4}\n'.format(token['text'], base_filename, token['start'], token['end'],
                                                   ' '.join(token['label'])))
                output_file.write(
                    '{0} {1} {2} {3} {4}\n'.format(token['text'], base_filename, token['start'], token['end'],
                                                   ' '.join(token['label'])))
            if verbose: print('\n')
            output_file.write('\n')

    output_file.close()
    print('Done.')
    if tokenizer == 'spacy':
        del spacy_nlp

if __name__ == '__main__':
    path_dataset = '/home/zeyuzhang/PycharmProjects/scienceexam_ner/bert_data'
    dataset_brat_folders = {}
    dataset_conll_filepaths = {}
    tokenizer = 'spacy'
    for dataset_type in ['train', 'valid', 'test']:
        dataset_brat_folders[dataset_type] = os.path.join(path_dataset,
                                                          dataset_type)
        if os.path.exists(dataset_brat_folders[dataset_type]) \
                and len(glob.glob(os.path.join(dataset_brat_folders[dataset_type], '*.txt'))) > 0:
            dataset_filepath_for_tokenizer = os.path.join(path_dataset,
                                                          '{0}_{1}.txt'.format(dataset_type, tokenizer))

            brat_to_conll(dataset_brat_folders[dataset_type],
                                        dataset_filepath_for_tokenizer, tokenizer, 'en')
            dataset_conll_filepaths[dataset_type] = dataset_filepath_for_tokenizer
