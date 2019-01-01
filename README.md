# HAST
Aspect Term Extraction with **H**istory **A**ttention and **S**elective **T**ransformation.

## Requirements
* Python 3.6
* [DyNet 2.0.2](https://github.com/clab/dynet) (For building DyNet and enabling the python bindings, please follow the instructions in this [link](http://dynet.readthedocs.io/en/latest/python.html#manual-installation))
* nltk 3.2.2
* numpy 1.13.3

## External Linguistic Resources
* [Glove Word Embeddings](https://nlp.stanford.edu/projects/glove/) (840B, 2.2M vocab).
* [MPQA Subjectivity Lexicon](http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/)

## Preprocessing
* Window-based input (window size is 3, as done in Pengfei's [work](http://www.aclweb.org/anthology/D15-1168)).
* Replacing the punctuations with the same token `PUNCT`.
* Only the sentimental words with strong subjectivity are employed to provide distant supervision.

## Running
```
python main.py -ds_name [YOUR_DATASET_NAME] -sgd_lr [YOUR_LEARNING_RATE_FOR_SGD] -win [YOUR_WINDOW_SIZE] -optimizer [YOUR_OPTIMIZER] -rnn_type [LSTM|GRU] -attention_type [bilinear|concat]
```

## Environment
* OS: REHL Server 6.4 (Santiago)
* CPU: Intel Xeon CPU E5-2620 (Yes, we do not use GPU)

## Citation
If the code is used in your research, please star this repo and cite our paper as follows:
```
@inproceedings{li2018aspect,
  title={Aspect Term Extraction with History Attention and Selective Transformation},
  author={Li, Xin and Bing, Lidong and Li, Piji and Lam, Wai and Yang, Zhimou},
  booktitle={IJCAI},
  pages={4194--4200}
  year={2018}
}
```

