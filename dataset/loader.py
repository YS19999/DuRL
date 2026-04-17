import os
import itertools
import collections
import json
from collections import defaultdict
from copy import copy

import numpy as np
import pandas as pd
import torch
from dataset.utils import tprint

from transformers import BertTokenizer, AutoTokenizer
# from pytorch_transformers import BertModel

def _get_Liu_classes(args):

    train_classes = list(range(20))
    val_classes = list(range(20, 30))
    test_classes = list(range(30, 54))

    return train_classes, val_classes, test_classes


def _get_hwu64_classes(args):
    '''
        @return list of classes associated with each split
    '''

    train_classes = list(range(23))
    val_classes = list(range(23, 39))
    test_classes = list(range(39, 64))

    return train_classes, val_classes, test_classes


def _get_clinic150_classes(args):
    '''
        @return list of classes associated with each split
    '''

    train_classes = list(range(50))
    val_classes = list(range(50, 100))
    test_classes = list(range(100, 150))

    return train_classes, val_classes, test_classes


def _get_banking77_classes(args):
    '''
        @return list of classes associated with each split
    '''

    train_classes = list(range(25))
    val_classes = list(range(25, 50))
    test_classes = list(range(50, 77))

    return train_classes, val_classes, test_classes


def _load_csv(path):
    label = {}
    text_len = []

    dataset = pd.read_csv(path, encoding='UTF-8')
    texts, intents = dataset['content'], dataset['label']

    data = []
    for i, line in enumerate(texts):

        if int(intents[i]) not in label:
            label[int(intents[i])] = 1
        else:
            label[int(intents[i])] += 1

        item = {
            'label': int(intents[i]),
            'text': line,
        }

        text_len.append(len(line))
        data.append(item)

    tprint('Class balance:')

    print(label)

    tprint('Avg len: {}'.format(sum(text_len) / (len(text_len))))

    return data

def _read_words(data):
    '''
        Count the occurrences of all words
        @param data: list of examples
        @return words: list of words (with duplicates)
    '''
    words = []
    for example in data:
        words += example['text']
    return words


def _meta_split(all_data, train_classes, val_classes, test_classes, args):

    train_data, val_data, test_data = [], [], []

    if all_data is not None:
        for example in all_data:
            if example['label'] in train_classes:
                train_data.append(example)
            if example['label'] in val_classes:
                val_data.append(example)
            if example['label'] in test_classes:
                test_data.append(example)
    else:
        if args.dataset == 'cb':
            train_data = _load_csv(path="../MetaAE/clinic150.csv")
            val_data = _load_csv(path="../MetaAE/liu.csv")
            test_data = _load_csv(path="../MetaAE/banking77.csv")
        elif args.dataset == 'ca':
            train_data = _load_csv(path="../MetaAE/clinic150.csv")
            val_data = _load_csv(path="../MetaAE/liu.csv")
            test_data = _load_csv(path="../MetaAE/acid.csv")
        elif args.dataset == 'ch':
            train_data = _load_csv(path="../MetaAE/clinic150.csv")
            val_data = _load_csv(path="../MetaAE/liu.csv")
            test_data = _load_csv(path="../MetaAE/hwu64.csv")
        elif args.dataset == 'bc':
            train_data = _load_csv(path="../MetaAE/banking77.csv")
            val_data = _load_csv(path="../MetaAE/liu.csv")
            test_data = _load_csv(path="../MetaAE/clinic150.csv")
        elif args.dataset == 'ba':
            train_data = _load_csv(path="../MetaAE/banking77.csv")
            val_data = _load_csv(path="../MetaAE/liu.csv")
            test_data = _load_csv(path="../MetaAE/acid.csv")
        elif args.dataset == 'bh':
            train_data = _load_csv(path="../MetaAE/banking77.csv")
            val_data = _load_csv(path="../MetaAE/liu.csv")
            test_data = _load_csv(path="../MetaAE/hwu64.csv")
        elif args.dataset == 'cl':
            train_data = _load_csv(path="../MetaAE/clinic150.csv")
            val_data = _load_csv(path="../MetaAE/hwu64.csv")
            test_data = _load_csv(path="../MetaAE/liu.csv")
        else:
            AssertionError

    return train_data, val_data, test_data


def _del_by_idx(array_list, idx, axis):
    '''
        Delete the specified index for each array in the array_lists

        @params: array_list: list of np arrays
        @params: idx: list of int
        @params: axis: int

        @return: res: tuple of pruned np arrays
    '''
    if type(array_list) is not list:
        array_list = [array_list]

    # modified to perform operations in place
    for i, array in enumerate(array_list):
        array_list[i] = np.delete(array, idx, axis)

    if len(array_list) == 1:
        return array_list[0]
    else:
        return array_list


def _data_to_nparray(data, args):
    '''
        Convert the data into a dictionary of np arrays for speed.
    '''
    doc_label = np.array([x['label'] for x in data], dtype=np.int64)

    raw = np.array([e['text'] for e in data], dtype=object)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)

    for e in data:
        tokenize = tokenizer(e['text'], return_tensors="pt")
        e['bert_id'], e['attn_mask'] = tokenize['input_ids'][0].numpy(), tokenize['attention_mask'][0].numpy()

    text_len = np.array([len(e['bert_id']) for e in data])
    max_text_len = max(text_len)

    text = np.zeros([len(data), max_text_len], dtype=np.int64)
    text_mask = np.zeros([len(data), max_text_len], dtype=np.int64)

    del_idx = []
    # convert each token to its corresponding id
    for i in range(len(data)):
        text[i, :len(data[i]['bert_id'])] = data[i]['bert_id']
        text_mask[i, :len(data[i]['attn_mask'])] = data[i]['attn_mask']

        # filter out document with only special tokens
        # unk (100), cls (101), sep (102), pad (0)
        if np.max(text[i]) < 103:
            del_idx.append(i)

    text_len, text, doc_label, raw = _del_by_idx([text_len, text, doc_label, raw], del_idx, 0)

    new_data = {
        'text': text,
        'text_len': text_len,
        'attn_mask': text_mask,
        'label': doc_label,
        'raw': raw,
    }

    return new_data



def load_dataset_DG(args):

    tprint("current: {}, {}-way {}-shot".format(args.dataset, args.way, args.shot))

    # Split into meta-train, meta-val, meta-test data
    train_data, val_data, test_data = _meta_split(None, None, None, None, args)
    tprint('#train {}, #val {}, #test {}'.format(
        len(train_data), len(val_data), len(test_data)))

    # Convert everything into np array for fast data loading
    train_data = _data_to_nparray(train_data, args)
    val_data = _data_to_nparray(val_data, args)
    test_data = _data_to_nparray(test_data, args)

    train_data['is_train'] = True
    # this tag is used for distinguishing train/val/test when creating source pool

    return train_data, val_data, test_data

