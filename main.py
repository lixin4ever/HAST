import argparse
import dynet_config
dynet_config.set_gpu(True) # actually, this statement does not work.

dy_seed = 1314159
#dy_seed = 78996
dynet_config.set(mem='4096', random_seed=dy_seed)
import dynet as dy
import random
from utils import *
from evals import *
import os
import pickle
from model import Model


def run(args, flag2embedding_path, test_ids):
    """

    :param args: user-specific arguments
    :param flag2embedding_path: flag to word embedding path
    :param test_ids: list of sample id for testing, only used for cross-validation
    :return:
    """
    win_size = args.win
    ds_name = args.ds_name
    train_set, test_test, vocab, _, tag_vocab, tag_inv_vocab = build_dataset(ds_name=ds_name, win=win_size,
                                                                             mode=args.running_mode, test_ids=test_ids)
    
    embedding_flag = args.flag
    embedding_path = flag2embedding_path[embedding_flag]
    if args.running_mode == 'cross-validation':
        embedding_pkl_path = './embeddings/%s_%s_cv.pkl' % (ds_name, embedding_flag)
    else:
        embedding_pkl_path = './embeddings/%s_%s.pkl' % (ds_name, embedding_flag)
        
    if not os.path.exists('./embeddings'):
        os.mkdir('./embeddings')

    if not os.path.exists(embedding_pkl_path):
        print("Load embedding from %s..." % embedding_path)
        embeddings = load_embedding(path=embedding_path, vocab=vocab)
        pickle.dump(embeddings, open(embedding_pkl_path, 'wb'))
    else:
        print("Use the saved word embeddings")
        embeddings = pickle.load(open(embedding_pkl_path, 'rb'))
        # if len(embeddings) != len(vocab) or len(embeddings[0]) == 200 or len(embeddings[0] == 300):
        if len(embeddings) != len(vocab):
            print("vocabulary dis-match!! reload the word embeddings")
            print("Load embedding from %s..." % embedding_path)
            embeddings = load_embedding(path=embedding_path, vocab=vocab)
            pickle.dump(embeddings, open(embedding_pkl_path, 'wb'))

    args.dim_w = len(embeddings[0])
    args.n_asp_tags = len(tag_vocab)
    args.n_opi_tags = 2  # opinion tags follow the OT schema

    print("Embeddings shape:", embeddings.shape)
    if embeddings.shape[1] == 300:
        print("Use case-insensitive word embeddings")
    else:
        print("Use case-sensitive+case-insensitive word embeddings")
    print("Parameters:", args)

    n_epoch = args.n_epoch

    # model = MODEL(params=args, vocab=vocab, label2tag=tag_inv_vocab, pretrained_embeddings=embeddings)
    model = Model(params=args, vocab=vocab, label2tag=tag_inv_vocab, pretrained_embeddings=embeddings)

    results_strings = []
    for i in range(1, n_epoch + 1):
        print("In Epoch %s / %s:" % (i, n_epoch))
        # shuffle training dataset
        random.shuffle(train_set)
        # ---------------training----------------
        loss, Y_pred_asp, Y_pred_opi, _, _ = model(dataset=train_set, is_train=True)
        Y_gold_asp = [sent2tags(sent) for sent in train_set]
        p, r, f1, _ = evaluate_chunk(test_Y=Y_gold_asp, pred_Y=Y_pred_asp, dataset=train_set)
        print('\ttrain loss: %.2f, train precision: %.2f, train recall: %.2f, train f1: %.2f' % (loss, \
                                                                                                 p * 100, r * 100,
                                                                                                 f1 * 100))
        # ---------------prediction----------------
        loss, Y_pred_asp, Y_pred_opi, aspect_attention, opinion_attention = model(dataset=test_test, is_train=False)
        Y_gold_asp = [sent2tags(sent) for sent in test_test]
        p, r, f1, output_lines = evaluate_chunk(test_Y=Y_gold_asp, pred_Y=Y_pred_asp, dataset=test_test)

        print("\tCurrent results: precision: %.2f, recall: %.2f, f1: %.2f" % (p * 100, r * 100, f1 * 100))
        results_strings.append("In Epoch %s: precision: %.2f, recall: %.2f, f1: %.2f\n" % (i, p * 100, r * 100, f1 * 100))

    # log information consist of settings of hyper-parameters and intermediate results
    result_logs = ['-------------------------------------------------------\n']
    params_string = str(args)
    if embeddings.shape[1] == 300:
        result_logs.append("Use case-insensitive word embeddings\n")
    else:
        result_logs.append("Use case-sensitive+case-insensitive word embeddings\n")
    result_logs.append("Running model: %s\n" % args.model_name)
    result_logs.append(params_string + "\n")
    result_logs.extend(results_strings)
    result_logs.append('-------------------------------------------------------\n')
    with open("./log/%s.txt" % ds_name, 'a') as fp:
        fp.writelines(result_logs)


if __name__ == '__main__':
    random_seed = 1234
    random.seed(random_seed)
    parser = argparse.ArgumentParser(description="Aspect Term Extraction")
    parser.add_argument("-ds_name", type=str, default='14semeval_rest', help="dataset name")
    parser.add_argument("-dim_asp", type=int, default=100, help="dimension aspect")
    parser.add_argument("-dim_opi", type=int, default=30, help="dimension opinion")
    parser.add_argument("-win", type=int, default=3, help="window size")
    parser.add_argument("-n_steps", type=int, default=5, help="number of steps in truncated self attention")
    parser.add_argument("-optimizer", type=str, default="sgd", help="optimizer (or, trainer)")
    parser.add_argument("-n_epoch", type=int, default=100, help="number of training epoch")
    parser.add_argument("-dropout", type=float, default=0.5, help="dropout rate for final representations")
    parser.add_argument("-dropout_asp", type=float, default=0.5, help="dropout rate for ASP-LSTM")
    parser.add_argument("-dropout_opi", type=float, default=0.5, help="dropout rate for OPI-LSTM")
    parser.add_argument("-flag", type=str, default="glove_840B", help="word embedding flag")
    parser.add_argument("-rnn_type", type=str, default="LSTM", help="type of rnn unit, currently only LSTM and GRU are supported")
    parser.add_argument("-sgd_lr", type=float, default=0.07, help="learning rate for sgd, only used when the optimizer is sgd")
    parser.add_argument("-model_name", type=str, default="full", help="model name")
    parser.add_argument("-attention_type", type=str, default="bilinear", help="attention type")
    parser.add_argument("-running_mode", type=str, default="train-test", help="running mode")
    args = parser.parse_args()
    # seed number for dynet libary
    args.dynet_seed = dy_seed
    # seed number for random module
    args.random_seed = random_seed
    model_name = "full"

    # NOTE: we consistently use glove_840B when reporting the benchmark results
    flag2embedding_path = {
        'glove_6B': '/projdata9/info_fil/lixin/Research/OTE/embeddings/glove_6B_300d.txt',
        'glove_42B': '/projdata9/info_fil/lixin/Research/OTE/embeddings/glove_42B_300d.txt',
        'glove_840B': '/projdata9/info_fil/lixin/Research/OTE/embeddings/glove_840B_300d.txt',
        # mainly for 15semeval rest
        'yelp_rest1': '/projdata9/info_fil/lixin/Research/yelp/yelp_vec_200_2_win5_sent.txt',
        # mainly for 14semeval_rest and 16semeval_rest
        'yelp_rest2': '/projdata9/info_fil/lixin/Research/yelp/yelp_vec_200_2_new.txt',
        'amazon_laptop': '/projdata9/info_fil/lixin/Resources/amazon_full/vectors/amazon_laptop_vec_200_5.txt'
    }

    dataset2train_num = {
        '14semeval_rest': 3041,
        '14semeval_laptop': 3045,
        '15semeval_rest': 1315,
        '16semeval_rest': 2000
    }

    if args.running_mode == 'cross-validation':
        n_train = dataset2train_num[args.ds_name]
        total_ids = list(range(n_train))
        random.shuffle(total_ids)
        #print(total_ids)
        n_fold = int(n_train / 5)
        for i in range(5):
            print("In %s-th fold of cross-validation..." % (i + 1))
            if i == 4:
                test_ids = total_ids[4*n_fold:]
            else:
                test_ids = total_ids[i*n_fold:(i+1)*n_fold]
            run(args=args, flag2embedding_path=flag2embedding_path, test_ids=test_ids)
    else:
        run(args=args, flag2embedding_path=flag2embedding_path, test_ids=None)







