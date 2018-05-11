# HAST
Aspect Term Extraction with **H**istory **A**ttention and **S**elective **T**ransformation. (Currently, code is only available upon request)

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
* Replacing the OOVs with the same token `PUNCT`.
* Only the sentimental words with strong subjectivity are employed to provide distant supervision.


## Citation
If the code is used in your research, please cite our paper as follows:
```
@article{li2018aspect,
  title={Aspect Term Extraction with History Attention and Selective Transformation},
  author={Li, Xin and Bing, Lidong and Li, Piji and Lam, Wai and Yang, Zhimou},
  journal={arXiv preprint arXiv:1805.00760},
  year={2018}
}
```

