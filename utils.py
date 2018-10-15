import numpy as np
from nltk import ngrams
import string
import os
import random


def ot2bio(tag_sequence):
    """
    OT tag sequence to BIO tag sequence
    :param tag_sequence:
    :return:
    """
    new_sequence = []
    prev_tag = '$$$'
    for t in tag_sequence:
        assert t == 'O' or t == 'T'
        if t == 'O':
            new_sequence.append(t)
        elif t == 'T':
            if prev_tag == 'T':
                new_sequence.append('I')
            else:
                new_sequence.append('B')
        prev_tag = t
    assert len(new_sequence) == len(tag_sequence)
    return new_sequence


def ot2bieos(tag_sequence):
    """
    convert OT sequence to BIEOS tag sequence
    OT and BIEOS denote tagging schema
    """
    new_sequence = []
    prev = ''
    n_tag = len(tag_sequence)
    for i in range(n_tag):
        cur = tag_sequence[i]
        assert cur == 'O' or cur == 'T'
        if cur == 'O':
            new_sequence.append('O')
        else:
            # current tag is T, that is, part of an aspect or a singleton
            if prev != cur:
                # previous tag is not T, current word can only be head word of an aspect or a singleton
                if i == (n_tag - 1):
                    new_sequence.append('S')
                elif tag_sequence[i + 1] == cur:
                    new_sequence.append('B')
                elif tag_sequence[i + 1] != cur:
                    new_sequence.append('S')
                else:
                    raise ValueError('Unexpected tagging case!!')
            else:
                # previous tag is T, current word can only be internal word or the end word of an aspect
                if i == (n_tag - 1):
                    new_sequence.append('E')
                elif tag_sequence[i + 1] == cur:
                    new_sequence.append('I')
                elif tag_sequence[i + 1] != cur:
                    new_sequence.append('E')
                else:
                    raise ValueError('Unexpected tagging case!!')
        prev = cur
    assert len(new_sequence) == len(tag_sequence)
    return new_sequence


def bio2ot(tag_sequence):
    """
    BIO tag sequence to OT tag sequence
    :param tag_sequence:
    :return:
    """
    ot_tags = []
    for t in tag_sequence:
        if t == 'O':
            ot_tags.append("O")
        else:
            ot_tags.append("T")
    return ot_tags


def sent2tags(sent):
    return [t for t in sent['raw_tags']]


def read_data(path, opi_path):
    """
    construct dataset with aspect tags and opinion tags
    :param path: path of data file
    :param opi_path: path of data file
    :return:
    """
    # load opinion annotations
    opinions = []
    with open(opi_path) as fp:
        for line in fp:
            opi_record = {}
            items = line.strip().split(', ')
            #print(items)
            for item in items:
                eles = item.split()
                polarity = eles[-1]
                word = ' '.join(eles[:-1])
                opi_record[word] = polarity
            opinions.append(opi_record)
    dataset = []
    idx = 0
    with open(path) as fp:
        for line in fp:
            record = {}
            opi_record = opinions[idx]
            sent, tag_string = line.strip().split("####")
            record['sentence'] = sent
            tag_sequence = tag_string.split(' ')
            words, tags, opi_tags = [], [], []
            for item in tag_sequence:
                eles = item.split('=')
                if len(eles) == 2:
                    word, tag = eles
                else:
                    n_ele = len(eles)
                    tag = eles[-1]
                    word = ''
                    for k in range(n_ele):
                        ele = eles[k]
                        if ele == '' and k == 0:
                            continue
                        elif ele == '':
                            word += '='
                        else:
                            word += ele
                #words.append(word.lower())
                if word not in string.punctuation:
                    words.append(word.lower())
                else:
                    words.append('PUNCT')
                tags.append(tag)
                # opinion tagging schema: OT
                if word in opi_record:
                    opi_tags.append('T')
                else:
                    opi_tags.append('O')
            record['words'] = words.copy()
            # origin aspect tags
            record['raw_tags'] = tags.copy()
            record['opinion_tags'] = opi_tags.copy()
            dataset.append(record)
            idx += 1
    print("N opinion:", len(opinions))
    print("N dataset:", len(dataset))
    assert len(opinions) == len(dataset)
    return dataset


def read_lexicon(path):
    """
    load sentiment lexicon from the disk
    :param path:
    :return:
    """
    lexicon = {}
    with open(path) as fp:
        for line in fp:
            word, polarity = line.strip().split('\t')
            lexicon[word] = polarity
    return lexicon


def build_vocab(trainset, testset):
    """
    build vocabulary from the training set and the testing set
    :param trainset:
    :param testset:
    :return:
    """
    wid = 0
    vocab, inv_vocab = {}, {}
    for record in trainset + testset:
        for w in record['words']:
            if w not in vocab:
                vocab[w] = wid
                inv_vocab[wid] = w
                wid += 1
    vocab['PADDING'] = wid
    inv_vocab[wid] = 'PADDING'
    return vocab, inv_vocab


def obtain_labels(trainset, testset, schema='OT'):
    """
    transform tags to integer labels
    :param trainset:
    :param testset:
    :return:
    """
    if schema == 'OT':
        tag_vocab = {'O': 0, 'T': 1}
    elif schema == 'BIO':
        tag_vocab = {'O': 0, 'B': 1, 'I': 2}
    elif schema == 'BIEOS':
        tag_vocab = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
    else:
        raise Exception("Invalid tagging schema!!!")
    tag_inv_vocab = {}
    for t in tag_vocab:
        label = tag_vocab[t]
        tag_inv_vocab[label] = t
    n_train = len(trainset)
    n_test = len(testset)
    for i in range(n_train):
        raw_tags = trainset[i]['raw_tags']
        opinion_tags = trainset[i]['opinion_tags']
        if schema == 'OT':
            tags = [ele for ele in raw_tags]
        elif schema == 'BIO':
            tags = ot2bio(tag_sequence=raw_tags)
        elif schema == 'BIEOS':
            tags = ot2bieos(tag_sequence=raw_tags)
        else:
            raise Exception("Invalid value")
        labels = [tag_vocab[t] for t in tags]
        opinion_labels = [int(t == 'T') for t in opinion_tags]
        trainset[i]['tags'] = tags.copy()
        trainset[i]['labels'] = labels.copy()
        trainset[i]['opinion_labels'] = opinion_labels.copy()

    for i in range(n_test):
        raw_tags = testset[i]['raw_tags']
        opinion_tags = testset[i]['opinion_tags']
        if schema == 'OT':
            tags = [ele for ele in raw_tags]
        elif schema == 'BIO':
            tags = ot2bio(tag_sequence=raw_tags)
        elif schema == 'BIEOS':
            tags = ot2bieos(tag_sequence=raw_tags)
        else:
            raise Exception("Invalid value")
        labels = [tag_vocab[t] for t in tags]
        opinion_labels = [int(t == 'T') for t in opinion_tags]
        testset[i]['tags'] = tags.copy()
        testset[i]['labels'] = labels.copy()
        testset[i]['opinion_labels'] = opinion_labels.copy()

    return trainset, testset, tag_vocab, tag_inv_vocab


def obtain_word_id(dataset, vocab, win):
    """
    transform word to word index
    :param dataset:
    :param vocab:
    :param win: context window, should be an odd number
    :return:
    """
    n_records = len(dataset)
    for i in range(n_records):
        words = dataset[i]['words']
        sent_len = len(words)
        n_pad_token = win // 2
        padded_left = ["PADDING" for _ in range(n_pad_token)]
        padded_right = ["PADDING" for _ in range(n_pad_token)]
        padded_words = padded_left + words + padded_right
        n_grams = list(ngrams(padded_words, win))
        assert len(n_grams) == sent_len
        full_words = []
        for t in n_grams:
            full_words.append(t)
        # the window-based input
        wids = [[vocab[w] for w in ngram] for ngram in full_words]
        dataset[i]['wids'] = np.array(wids, dtype='int32')
    return dataset


def load_embedding(path, vocab):
    """
    load pre-trained word embedding from the disk
    :param path:
    :return:
    """
    vocab_lower = {}
    for w in vocab:
        if w == 'PADDING' or w == 'PUNCT':
            continue
        if not w.islower():
            vocab_lower[w] = 1
    raw_embeddings = {}
    with open(path) as fp:
        for line in fp:
            eles = line.strip().split()
            word = eles[0]
            if word in vocab:
                raw_embeddings[word] = eles[1:]
            #if word in vocab_lower:
            #    raw_embeddings[word] = eles[1:]
    dim_w = len(list(raw_embeddings.items())[0][1])
    n_words = len(vocab)
    # embeddings = np.zeros(shape=(n_words, 2 * dim_w))
    # only use case-insensitive word embeddings
    embeddings = np.zeros(shape=(n_words, dim_w))
    print(embeddings.shape)
    for w in vocab:
        if w == 'PADDING' or w == 'PUNCT':
            wid = vocab[w]
            embeddings[wid] = np.random.uniform(-0.25, 0.25, dim_w)
            continue
        # vec_cs: case-sensitive vector
        # vec_ci: case-insensitive vector
        if w in raw_embeddings:
            #print(raw_embeddings[w])
            try:
                vec_cs = [float(ele) for ele in raw_embeddings[w]]
            except ValueError:
                vec_cs = np.random.uniform(-0.25, 0.25, dim_w)
        else:
            # sample word embedding from uniform distribution
            vec_cs = np.random.uniform(-0.25, 0.25, dim_w)
        w_lower = w.lower()
        if w_lower in raw_embeddings:
            #print(raw_embeddings[w])
            try:
                vec_ci = [float(ele) for ele in raw_embeddings[w_lower]]
            except ValueError:
                vec_ci = np.random.uniform(-0.25, 0.25, dim_w)
        else:
            # sample word embedding from uniform distribution
            vec_ci = np.random.uniform(-0.25, 0.25, dim_w)
        wid = vocab[w]
        # use the case-sensitive and case-insensitive word embeddings
        # embeddings[wid] = np.concatenate([vec_cs, vec_ci])
        # only use the case-insensitive word embeddings
        embeddings[wid] = vec_ci
    return np.array(embeddings, dtype='float32')


def build_dataset(ds_name, win=1, mode="train-test", test_ids=None):
    """

    :param ds_name: dataset name
    :param win: context window
    :param mode: running mode, either train-test or cross-validation
    :param test_ids: list of training sample id for testing, only used in cross-validation
    :return:
    """
    # dataset for the task of aspect term extraction
    train_path = './dataset/%s_train.txt' % ds_name
    test_path = './dataset/%s_test.txt' % ds_name

    # dataset for the task of opinion word detection, we do not use gold standard labels
    # but distant supervision from existing opinion lexicon
    train_opi_path = './dataset/%s_train_opi_ds.txt' % ds_name
    test_opi_path = './dataset/%s_test_opi_ds.txt' % ds_name

    train_set = read_data(train_path, train_opi_path)
    if mode == 'train-test':
        test_set = read_data(test_path, test_opi_path)
    elif mode == 'cross-validation':
        dataset = [r for r in train_set]
        train_set, test_set = [], []
        for i in range(len(dataset)):
            if i in test_ids:
                test_set.append(dataset[i])
            else:
                train_set.append(dataset[i])
        print("In the cross validation mode: %s training documents, %s testing documents" % (len(train_set), len(test_set)))
    else:
        raise Exception("Invalid running mode!!!")

    vocab, inv_vocab = build_vocab(trainset=train_set, testset=test_set)

    train_set, test_set, tag_vocab, tag_inv_vocab = obtain_labels(trainset=train_set, testset=test_set, schema='BIO')

    train_set = obtain_word_id(dataset=train_set, vocab=vocab, win=win)
    test_set = obtain_word_id(dataset=test_set, vocab=vocab, win=win)

    return train_set, test_set, vocab, inv_vocab, tag_vocab, tag_inv_vocab


def tag2aspect(tag_sequence):
    """
    convert BIEOS tag sequence to the aspect sequence
    :param tag_sequence: tag sequence in BIEOS tagging schema
    :return:
    """
    n_tag = len(tag_sequence)
    chunk_sequence = []
    beg, end = -1, -1
    # number of multi-word and single-word aspect
    n_mult, n_s = 0, 0
    for i in range(n_tag):
        if tag_sequence[i] == 'S':
            # start position and end position are kept same for the singleton
            chunk_sequence.append((i, i))
            n_s += 1
        elif tag_sequence[i] == 'B':
            beg = i
        elif tag_sequence[i] == 'E':
            end = i
            if end > beg:
                # only valid chunk is acceptable
                chunk_sequence.append((beg, end))
                n_mult += 1
    return chunk_sequence, n_s, n_mult

