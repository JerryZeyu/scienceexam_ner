"""Metrics to assess performance on sequence labeling task given prediction
Functions named as ``*_score`` return a scalar value to maximize: the higher
the better
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import pickle
import os
import numpy as np

def pickle_load_large_file(filepath):
    max_bytes = 2**31 - 1
    input_size = os.path.getsize(filepath)
    bytes_in = bytearray(0)
    with open(filepath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    obj = pickle.loads(bytes_in)
    return obj
# for idx, singlechunk in enumerate(chunk):
#     tag_temp = tag[idx]
#     type__temp = type_[idx]
#     for idx__, prev_tag_temp in enumerate(prev_tag):
#         prev_type_temp = prev_type[idx__]
#         if end_of_chunk(prev_tag_temp, tag_temp, prev_type_temp, type__temp):
#             #print((prev_type_temp, previous_begin_offset[0], i - 1))
#             chunks.append((prev_type_temp, previous_begin_offset[0], i - 1))
#         if start_of_chunk(prev_tag_temp, tag_temp, prev_type_temp, type__temp):
#             begin_offset.append(i)

def get_entities(seq, suffix=False):
    """Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        # >>> from seqeval.metrics.sequence_labeling import get_entities
        # >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        # >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list ['B-LevelOfInclusion'], ['B-ScientificTools'],
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + [['O']]]
    #seq = [['B-Examine', 'B-Pressure'], ['B-Measurements', 'I-Pressure'],['B-Examine', 'I-Pressure'],['I-Examine', 'I-Pressure']]
    #print(seq)
    print('seq: ',seq[0:100])
    prev_tag = ['O']
    prev_type = ['']
    previous_begin_offset = {}
    chunks = []
    #begin_offset = []
    for i, chunk in enumerate(seq + [['O']]):
        if chunk == []:
            chunk = ['O']
        tag = [singlechunk[0] for singlechunk in chunk]
        type_ = [singlechunk.split('-')[-1] for singlechunk in chunk]
        # print('chunk: ',chunk)
        # print(begin_offset)
        # print(prev_tag)
        # print(prev_type)
        for idx, prev_tag_temp in enumerate(prev_tag):
            prev_type_temp = prev_type[idx]
            if end_of_chunk(prev_tag_temp, tag, prev_type_temp, type_):
                #print((prev_type_temp, begin_offset[idx], i - 1))
                chunks.append((prev_type_temp, begin_offset[prev_type_temp], i - 1))
        #begin_offset=[]
        begin_offset = {}
        for idx, singlechunk in enumerate(chunk):
            tag_temp = tag[idx]
            type__temp = type_[idx]
            if start_of_chunk(tag_temp, type__temp):
                begin_offset[type__temp] = i
            else:
                if type__temp not in previous_begin_offset.keys():
                    begin_offset[type__temp] = i
                else:
                    begin_offset[type__temp]=previous_begin_offset[type__temp]

            #print('begin offset: ',begin_offset)
        previous_begin_offset=begin_offset.copy()
        prev_tag = tag
        prev_type = type_
    print(len(chunks))
    print('chunks: ',sorted(chunks[0:100],key=lambda x: x[1]))
    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False


    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and 'I' not in tag: chunk_end = True
    if prev_tag == 'B' and 'I' in tag:
        for idx, tag_temp in enumerate(tag):
            if tag_temp == 'I':
                if type_[idx] == prev_type:
                    chunk_end = False
                    break
            chunk_end = True
    if prev_tag == 'I' and 'I' not in tag: chunk_end = True
    if prev_tag == 'I' and 'I' in tag:
        for idx, tag_temp in enumerate(tag):
            if tag_temp == 'I':
                if type_[idx] == prev_type:
                    chunk_end = False
                    break
            chunk_end = True
    # if prev_tag == 'B' and tag == 'B': chunk_end = True
    # if prev_tag == 'B' and tag == 'S': chunk_end = True
    # if prev_tag == 'B' and tag == 'O': chunk_end = True
    # if prev_tag == 'I' and tag == 'B': chunk_end = True
    # if prev_tag == 'I' and tag == 'S': chunk_end = True
    # if prev_tag == 'I' and tag == 'O': chunk_end = True

    # if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
    #     chunk_end = True

    return chunk_end


def start_of_chunk(tag, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    # if prev_tag == 'E' and tag == 'E': chunk_start = True
    # if prev_tag == 'E' and tag == 'I': chunk_start = True
    # if prev_tag == 'S' and tag == 'E': chunk_start = True
    # if prev_tag == 'S' and tag == 'I': chunk_start = True
    # if prev_tag == 'O' and tag == 'E': chunk_start = True
    # if prev_tag == 'O' and tag == 'I': chunk_start = True
    #
    # if tag != 'O' and tag != '.' and prev_type != type_:
    #     chunk_start = True

    return chunk_start


def f1_score(y_true, y_pred, average='micro', suffix=False):
    """Compute the F1 score.

    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import f1_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score


def accuracy_score(y_true, y_pred):
    """Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import accuracy_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> accuracy_score(y_true, y_pred)
        0.80
    """
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]

    nb_correct = sum(y_t==y_p for y_t, y_p in zip(y_true, y_pred))
    nb_true = len(y_true)

    score = nb_correct / nb_true

    return score


def precision_score(y_true, y_pred, average='micro', suffix=False):
    """Compute the precision.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample.

    The best value is 1 and the worst value is 0.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import precision_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> precision_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)

    score = nb_correct / nb_pred if nb_pred > 0 else 0

    return score


def recall_score(y_true, y_pred, average='micro', suffix=False):
    """Compute the recall.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is 0.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import recall_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> recall_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)

    score = nb_correct / nb_true if nb_true > 0 else 0

    return score


def classification_report(y_true, y_pred, digits=2, suffix=False):
    """Build a text report showing the main classification metrics.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a classifier.
        digits : int. Number of digits for formatting output floating point values.

    Returns:
        report : string. Text summary of the precision, recall, F1 score for each class.

    Examples:
        # >>> from seqeval.metrics import classification_report
        # >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        # >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        # >>> print(classification_report(y_true, y_pred))
                     precision    recall  f1-score   support
        <BLANKLINE>
               MISC       0.00      0.00      0.00         1
                PER       1.00      1.00      1.00         1
        <BLANKLINE>
        avg / total       0.50      0.50      0.50         2
        <BLANKLINE>
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))
    #pred_entities = pickle_load_large_file('/home/zeyuzhang/PycharmProjects/scienceexam_ner_siglelabel/pred_entities.pkl')
    print(len(true_entities))
    print(len(pred_entities))
    name_width = 0
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))
        name_width = max(name_width, len(e[0]))
    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))
    print(len(d1))
    print(len(d2))
    last_line_heading = 'avg / total'
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    ps, rs, f1s, s = [], [], [], []
    for type_name, true_entities in d1.items():
        pred_entities = d2[type_name]
        nb_correct = len(true_entities & pred_entities)
        nb_pred = len(pred_entities)
        nb_true = len(true_entities)

        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        report += row_fmt.format(*[type_name, p, r, f1, nb_true], width=width, digits=digits)

        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    report += u'\n'

    # compute averages
    report += row_fmt.format(last_line_heading,
                             np.average(ps, weights=s),
                             np.average(rs, weights=s),
                             np.average(f1s, weights=s),
                             np.sum(s),
                             width=width, digits=digits)

    return report
    #return None