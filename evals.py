from utils import *
from tabulate import tabulate
import os


def evaluate_chunk(test_Y, pred_Y, dataset):
    """
    evaluate function for aspect term extraction, generally, it can also be used to evaluate the NER, NP-chunking task
    :param test_Y: gold standard tags (i.e., post-processed labels)
    :param pred_Y: predicted tags
    :param dataset: dataset
    :return:
    """
    assert len(test_Y) == len(pred_Y) == len(dataset)
    length = len(test_Y)
    TP, FN, FP = 0, 0, 0
    # hit count of mult-word aspect and singleton
    n_mult, n_s = 0, 0
    # gold count of mult-word aspect and singleton
    n_mult_gold, n_s_gold = 0, 0
    # predicted count of mult-word aspect and singleton
    n_mult_pred, n_s_pred = 0, 0
    # number of errors in sentences not having aspect
    n_error_nsubj = 0
    n_error_nsubj_pred = 0

    # system output, for case study
    output_lines = []

    for i in range(length):
        gold = test_Y[i]
        pred = pred_Y[i]
        record = dataset[i]
        if 'T' in pred and 'T' not in gold:
            n_error_nsubj += 1
        assert len(gold) == len(pred)
        gold_aspects, n_s_g, n_mult_g = tag2aspect(tag_sequence=ot2bieos(tag_sequence=gold))
        pred_aspects, n_s_p, n_mult_p = tag2aspect(tag_sequence=ot2bieos(tag_sequence=pred))
        n_hit, n_hit_s, n_hit_mult, n_e_nsubj, error_type = match_aspect(pred=pred_aspects, gold=gold_aspects)

        sent = record['sentence']
        #words = sent.split()
        words = record['words']
        # gold aspect strings, predicted aspect strings
        ga, pa = [], []
        for (b, e) in gold_aspects:
            aspect_words = words[b:(e+1)]
            ga.append(' '.join(aspect_words))
        for (b, e) in pred_aspects:
            aspect_words = words[b:(e+1)]
            pa.append(' '.join(aspect_words))
        line = '%s\tGOLD:%s\tPRED:%s\n' % (sent, ', '.join(ga), ', '.join(pa))
        output_lines.append(line)
        n_error_nsubj_pred += n_e_nsubj

        n_s += n_hit_s
        n_s_gold += n_s_g
        n_s_pred += n_s_p
        n_mult += n_hit_mult
        n_mult_gold += n_mult_g
        n_mult_pred += n_mult_p
        TP += n_hit
        FP += (len(pred_aspects) - n_hit)
        FN += (len(gold_aspects) - n_hit)
    #print(tabulate([['singleton', '%s / %s' % (n_s, n_s_gold), '%s / %s' % (n_s, n_s_pred)], ['multi-words', '%s / %s' % (n_mult, n_mult_gold), '%s / %s' % (n_mult, n_mult_pred)], ['total', '%s / %s' % (TP, TP + FN), '%s / %s' % (TP, TP + FP)]], headers={'##', 'recall', 'precision'}))
    precision = float(TP) / float(TP + FP + 0.0001)
    recall = float(TP) / float(TP + FN + 0.0001)
    F1 = 2 * precision * recall / (precision + recall + 0.0001)
    return precision, recall, F1, output_lines


def match_aspect(pred, gold):
    """

    :param pred:
    :param gold:
    :return:
    """
    true_count = 0
    n_mult, n_s = 0, 0
    # number of error predictions in the sentence not having aspects
    n_error_nsubj = 0
    if gold == [] and pred != []:
        n_error_nsubj = len(pred)
    n_error_s, n_error_mult = 0, 0
    for t in pred:
        if t in gold:
            true_count += 1
            if t[1] > t[0]:
                n_mult += 1
            else:
                n_s += 1
        else:
            if t[1] > t[0]:
                n_error_mult += 1
            else:
                n_error_s += 1
    error_type = 'GOOD'
    if n_error_nsubj:
        # the ground truth has no aspects but the model predict some aspects for sentence
        error_type = 'NON_OT'
    elif n_mult + n_s != len(gold):
        # do not predict all of the tags
        error_type = 'ERROR'

    return true_count, n_s, n_mult, n_error_nsubj, error_type


def output(predictions, ds_name, model_name, result):
    """
    output system predictions onto the disk
    :param predictions: system predictions together with ground truth
    :param ds_name: dataset name
    :param result: result score (p, r, f1)
    :return:
    """
    if not os.path.exists('./output'):
        os.mkdir('./output')
    if not os.path.exists('./output/%s' % ds_name):
        os.mkdir('./output/%s' % ds_name)
    p, r, f1 = result
    assert f1 > 1.0 <= 100.0
    f1_string = int(f1 * 100)
    p_string = int(p * 100)
    r_string = int(r * 100)
    result_string = "%s_%s_%s" % (f1_string, p_string, r_string)
    with open('./output/%s/%s_%s.txt' % (ds_name, model_name, result_string), 'w+') as fp:
        fp.writelines(predictions)
