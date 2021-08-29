# Semantic Hashing models

This repo contains six well-known semantic hashing models: [VDSH](https://arxiv.org/pdf/1708.03436.pdf), [NASH](https://aclanthology.org/P18-1190.pdf), [BMSH](https://aclanthology.org/D19-1526.pdf), [WISH](https://aclanthology.org/2020.findings-emnlp.233.pdf), [AMMI](http://proceedings.mlr.press/v119/stratos20a/stratos20a.pdf), and [corrSH](https://aclanthology.org/2020.acl-main.71.pdf).

### Datasets

Please download the data from [here](https://github.com/unsuthee/VariationalDeepSemanticHashing/tree/master/dataset) and move them into the `./data/` directory.

### How to Run

Unsupervised document hashing on 20Newsgroups using 64 bits

```
python main.py ng64 data/ng20.tfidf.mat --train --cuda
```

