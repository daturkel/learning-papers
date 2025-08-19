### Landmark Papers in Machine Learning

*This document attempts to collect the papers which developed important techniques in machine learning. Research is a collaborative process, discoveries are made independently, and the difference between the original version and a precursor can be subtle, but I’ve done my best to select the papers that I think are novel or significant.*

*My opinions are by no means the final word on these topics. Please create an issue or pull request if you have a suggestion.*

- [Landmark Papers in Machine Learning](#landmark-papers-in-machine-learning)
  - [Key](#key)
  - [Association Rule Learning](#association-rule-learning)
  - [Datasets](#datasets)
    - [Enron](#enron)
    - [ImageNet](#imagenet)
  - [Decision Trees](#decision-trees)
  - [Deep Learning](#deep-learning)
    - [AlexNet (image classification CNN)](#alexnet-image-classification-cnn)
    - [Convolutional Neural Network](#convolutional-neural-network)
    - [DeepFace (facial recognition)](#deepface-facial-recognition)
    - [Generative Adversarial Network](#generative-adversarial-network)
    - [GPT](#gpt)
    - [Inception (classification/detection CNN)](#inception-classificationdetection-cnn)
    - [Long Short-Term Memory (LSTM)](#long-short-term-memory-lstm)
    - [Residual Neural Network (ResNet)](#residual-neural-network-resnet)
    - [Transformer (sequence to sequence modeling)](#transformer-sequence-to-sequence-modeling)
    - [U-Net (image segmentation CNN)](#u-net-image-segmentation-cnn)
    - [VGG (image recognition CNN)](#vgg-image-recognition-cnn)
  - [Ensemble Methods](#ensemble-methods)
    - [AdaBoost](#adaboost)
    - [Bagging](#bagging)
    - [Gradient Boosting](#gradient-boosting)
    - [Random Forest](#random-forest)
  - [Games](#games)
    - [AlphaGo](#alphago)
    - [Deep Blue](#deep-blue)
  - [Optimization](#optimization)
    - [Adam](#adam)
    - [Expectation Maximization](#expectation-maximization)
    - [Stochastic Gradient Descent](#stochastic-gradient-descent)
  - [Miscellaneous](#miscellaneous)
    - [Non-negative Matrix Factorization](#non-negative-matrix-factorization)
    - [PageRank](#pagerank)
    - [DeepQA (Watson)](#deepqa-watson)
  - [Natural Language Processing](#natural-language-processing)
    - [Latent Dirichlet Allocation](#latent-dirichlet-allocation)
    - [Latent Semantic Analysis](#latent-semantic-analysis)
    - [Word2Vec](#word2vec)
  - [Neural Network Components](#neural-network-components)
    - [Autograd](#autograd)
    - [Back-propagation](#back-propagation)
    - [Batch Normalization](#batch-normalization)
    - [Dropout](#dropout)
    - [Gated Recurrent Unit](#gated-recurrent-unit)
    - [Perceptron](#perceptron)
  - [Recommender Systems](#recommender-systems)
    - [Collaborative Filtering](#collaborative-filtering)
    - [Matrix Factorization](#matrix-factorization)
    - [Implicit Matrix Factorization](#implicit-matrix-factorization)
  - [Regression](#regression)
    - [Elastic Net](#elastic-net)
    - [Lasso](#lasso)
  - [Software](#software)
    - [MapReduce](#mapreduce)
    - [TensorFlow](#tensorflow)
    - [Torch](#torch)
  - [Supervised Learning](#supervised-learning)
    - [k-Nearest Neighbors](#k-nearest-neighbors)
    - [Support Vector Machine](#support-vector-machine)
  - [Statistics](#statistics)
    - [The Bootstrap](#the-bootstrap)
- [Credits](#credits)

#### Key

| Icon |                                                              |
| ---- | ------------------------------------------------------------ |
| 🔒    | Paper behind paywall. In some cases, I provide an alternative link to the paper *if* it comes directly from one of the authors. |
| 🔑    | Freely available version of paywalled paper, directly from the author. |
| 💽    | Code associated with the paper.                              |
| 🏛️    | Precursor or historically relevant paper. This may be a fundamental breakthrough that paved the way for the concept in question to be developed. |
| 🔬    | Iteration, advancement, elaboration, or major popularization of a technique. |
| 📔    | Blog post or something other than a formal publication.      |
| 🌐    | Website associated with the paper.                           |
| 🎥    | Video associated with the paper.                             |
| 📊    | Slides or images associated with the paper.                  |

Papers proceeded by “See also” indicate either additional historical context or else major developments, breakthroughs, or applications.

#### Association Rule Learning

- **Scalable Algorithms for Association Mining (2000)**. Zaki, [@IEEE](https://ieeexplore.ieee.org/document/846291/metrics#metrics) 🔒.

- **Mining Frequent Patterns without Candidate Generation (2000)**. Han, Pei, and Yin, [@acm](https://dl.acm.org/doi/10.1145/335191.335372) .

- **Mining Association Rules between Sets of Items in Large Databases (1993)**, Agrawal, Imielinski, and Swami, [@CiteSeerX](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.40.6984) 🏛️.

- See also: **The GUHA method of automatic hypotheses determination (1966)**, Hájek, Havel, and Chytil, [@Springer](https://link.springer.com/article/10.1007/BF02345483) 🔒 🏛️.

#### Datasets

##### Enron

- **The Enron Corpus: A New Dataset for Email Classification Research (2004)**, Klimt and Yang, [@Springer](https://link.springer.com/chapter/10.1007/978-3-540-30115-8_22) 🔒 / [@author](https://bklimt.com/papers/2004_klimt_ecml.pdf) 🔑.
- See also: **Introducing the Enron Corpus (2004)**, Klimt and Yang, [@author](https://bklimt.com/papers/2004_klimt_ceas.pdf).

##### ImageNet

- **ImageNet: A large-scale hierarchical image database (2009)**, Deng et al., [@IEEE](https://ieeexplore.ieee.org/document/5206848) 🔒 / [@author](http://www.image-net.org/papers/imagenet_cvpr09.pdf) 🔑.
- See also: **ImageNet Large Scale Visual Recognition Challenge (2015)**, [@Springer](https://link.springer.com/article/10.1007/s11263-015-0816-y) 🔒 / [@arXiv](https://arxiv.org/abs/1409.0575) 🔑 + [@author](http://www.image-net.org/challenges/LSVRC/) 🌐.

#### Decision Trees

- **Induction of Decision Trees (1986)**, Quinlan, [@Springer](https://link.springer.com/article/10.1007/BF00116251).

#### Deep Learning

##### AlexNet (image classification CNN)

- **ImageNet Classification with Deep Convolutional Neural Networks (2012)**, [@NIPS](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

##### Convolutional Neural Network

- **Gradient-based learning applied to document recognition (1998)**, LeCun, Bottou, Bengio, and Haffner, [@IEEE](https://ieeexplore.ieee.org/document/726791/) 🔒 / [@author](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) 🔑.
- See also: **Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position (1980)**, Fukushima, [@Springer](https://link.springer.com/article/10.1007/BF00344251) 🏛️.
- See also: **Phoneme recognition using time-delay neural networks (1989)**, Waibel, Hanazawa, Hinton, Shikano, and Lang, [@IEEE](https://ieeexplore.ieee.org/document/21701) 🏛️.
- See also: **Fully Convolutional Networks for Semantic Segmentation (2014)**, Long, Shelhamer, and Darrell, [@arXiv](https://arxiv.org/abs/1411.4038).

##### DeepFace (facial recognition)

- **DeepFace: Closing the Gap to Human-Level Performance in Face Verification (2014)**, Taigman, Yang, Ranzato, and Wolf, [Facebook Research](https://research.fb.com/publications/deepface-closing-the-gap-to-human-level-performance-in-face-verification/).

##### Generative Adversarial Network

- **Generative Adversarial Nets (2014)**, Goodfellow et al., [@NIPS](https://papers.nips.cc/paper/5423-generative-adversarial-nets) + [@Github](https://github.com/goodfeli/adversarial) 💽.

##### GPT

- **Improving Language Understanding by Generative Pre-Training (2018)** *aka* GPT, Radford, Narasimhan, Salimans, and Sutskever, [@OpenAI](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) + [@Github](https://github.com/openai/finetune-transformer-lm) 💽 + [@OpenAI](https://openai.com/blog/language-unsupervised/) 📔.
- See also: **Language Models are Unsupervised Multitask Learners (2019)** *aka* GPT-2, Radford, Wu, Child, Luan, Amodei, and Sutskever, [@OpenAI](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 🔬 + [@Github](https://github.com/openai/gpt-2) 💽 + [@OpenAI](https://openai.com/blog/better-language-models/) 📔.
- See also: **Language Models are Few-Shot Learners (2020)** *aka* GPT-3, Brown et al., [@arXiv](https://arxiv.org/abs/2005.14165) + [@OpenAI](https://openai.com/blog/openai-api/) 📔.

##### Inception (classification/detection CNN)

- **Going Deeper with Convolutions (2014)**, Szegedy et al., [@ai.google](https://ai.google/research/pubs/pub43022) + [@Github](https://github.com/google/inception) 💽.
- See also: **Rethinking the Inception Architecture for Computer Vision (2016)**, Szegedy, Vanhoucke, Ioffe, Shlens, and Wojna, [@ai.google](https://ai.google/research/pubs/pub44903) 🔬.
- See also: **Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning (2016)**, Szegedy, Ioffe, Vanhoucke, and Alemi, [@ai.google](https://ai.google/research/pubs/pub45169) 🔬.

##### Long Short-Term Memory (LSTM)

- **Long Short-term Memory (1997)**, Hochreiter and Schmidhuber, [@CiteSeerX](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.13.634).

##### Residual Neural Network (ResNet)

- **Deep Residual Learning for Image Recognition (2015)**, He, Zhang, Ren, and Sun, [@arXiv](https://arxiv.org/abs/1512.03385).

##### Transformer (sequence to sequence modeling)

- **Attention Is All You Need (2017)**, Vaswani et al., [@NIPS](http://papers.nips.cc/paper/7181-attention-is-all-you-need).

##### U-Net (image segmentation CNN)

- **U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)**, Ronneberger, Fischer, Brox, [@Springer](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) 🔒 / [@arXiv](https://arxiv.org/abs/1505.04597) 🔑.

##### VGG (image recognition CNN)

- **Very Deep Convolutional Networks for Large-Scale Image Recognition (2015)**, Simonyan and Zisserman, [@arXiv](https://arxiv.org/abs/1409.1556) + [@author](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) 🌐 + [@ICLR](https://iclr.cc/archive/www/lib/exe/fetch.php%3Fmedia=iclr2015:simonyan-iclr2015.pdf) 📊 + [@YouTube](https://www.youtube.com/watch?v=OQe-9P51Z0s) 🎥.

#### Ensemble Methods

##### AdaBoost

- **A Decision-Theoretic Generalization of on-Line Learning and an Application to Boosting (1997—published as abstract in 1995)**, Freund and Schapire, [@CiteSeerX](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.32.8918).

- See also: **Experiments with a New Boosting Algorithm (1996)**, Freund and Schapire, [@CiteSeerX](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.133.1040) 🔬.

##### Bagging

- **Bagging Predictors (1996)**, Breiman, [@Springer](https://link.springer.com/article/10.1023/A:1018054314350).

##### Gradient Boosting

- **Greedy function approximation: A gradient boosting machine (2001)**, Friedman, [@Project Euclid](https://projecteuclid.org/euclid.aos/1013203451).
- See also: **XGBoost: A Scalable Tree Boosting System (2016)**, Chen and Guestrin, [@arXiv](https://arxiv.org/abs/1603.02754) 🔬 + [@GitHub](https://github.com/dmlc/xgboost) 💽.

##### Random Forest

- **Random Forests (2001)**, Breiman and Schapire, [@CiteSeerX](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.125.5395).

#### Games

##### AlphaGo

- **Mastering the game of Go with deep neural networks and tree search (2016)**, Silver et al., [@Nature](https://www.nature.com/articles/nature16961).

##### Deep Blue

- **IBM's deep blue chess grandmaster chips (1999)**, Hsu, [@IEEE](https://ieeexplore.ieee.org/abstract/document/755469) 🔒.
- See also: **Deep Blue (2002)**, Campbell, Hoane, and Hsu, [@ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0004370201001291?via%3Dihub) 🔒.

#### Optimization

##### Adam

- **Adam: A Method for Stochastic Optimization (2015)**, Kingma and Ba, [@arXiv](https://arxiv.org/abs/1412.6980).

##### Expectation Maximization

- **Maximum likelihood from incomplete data via the EM algorithm (1977)**, Dempster, Laird, and Rubin, [@CiteSeerX](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.133.4884).

##### Stochastic Gradient Descent

- **Stochastic Estimation of the Maximum of a Regression Function (1952)**, Kiefer and Wolfowitz, [@ProjectEuclid](https://projecteuclid.org/euclid.aoms/1177729392).
- See also: **A Stochastic Approximation Method (1951)**, Robbins and Monro, [@ProjectEuclid](https://projecteuclid.org/euclid.aoms/1177729586) 🏛️.

#### Miscellaneous

##### Non-negative Matrix Factorization

- **Learning the parts of objects by non-negative matrix factorization (1999)**, Lee and Seung, [@Nature](https://www.nature.com/articles/44565) 🔒.

##### PageRank

- **The PageRank Citation Ranking: Bringing Order to the Web (1998)**, Page, Brin, Motwani, and Winograd, [@CiteSeerX](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.38.5427).

##### DeepQA (Watson)

- **Building Watson: An Overview of the DeepQA Project (2010)**, Ferrucci et al., [@AAAI](https://www.aaai.org/ojs/index.php/aimagazine/article/view/2303).

##### Quantization (Qualcomm)

- **A White Paper on Neural Network Quantization (2021)**, Nagel et al., [@arXiv](https://arxiv.org/abs/2106.08295) 🔬.

#### Natural Language Processing

##### Latent Dirichlet Allocation

- **Latent Dirichlet Allocation (2003)**, Blei, Ng, and Jordan, [@JMLR](http://jmlr.csail.mit.edu/papers/v3/blei03a.html)

##### Latent Semantic Analysis

- **Indexing by latent semantic analysis (1990)**, Deerwater, Dumais, Furnas, Landauer, and Harshman, [@CiteSeerX](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.108.8490).

##### Word2Vec

- **Efficient Estimation of Word Representations in Vector Space (2013)**, Mikolov, Chen, Corrado, and Dean, [@arXiv](https://arxiv.org/abs/1301.3781) + [@Google Code](https://code.google.com/archive/p/word2vec/) 💽.

#### Neural Network Components

##### Autograd

- **Autograd: Effortless Gratients in Numpy (2015)**, [@ICML](https://indico.ijclab.in2p3.fr/event/2914/contributions/6483/subcontributions/180/attachments/6060/7185/automl-short.pdf) +  [@ICML](https://indico.ijclab.in2p3.fr/event/2914/contributions/6483/subcontributions/180/attachments/6059/7184/talk.pdf) 📊 + [@Github](https://github.com/HIPS/autograd) 💽.

##### Back-propagation

- **Learning representations by back-propagating errors (1986)**, Rumelhart, Hinton, and Williams, [@Nature](https://www.nature.com/articles/323533a0) 🔒.
- See also: **Backpropagation Applied to Handwritten Zip Code Recognition (1989)**, LeCun et al., [@IEEE](https://ieeexplore.ieee.org/document/6795724) 🔒🔬 / [@author](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf) 🔑.

##### Batch Normalization

- **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (2015)**, Ioffe and Szegedy [@ICML via PMLR](http://proceedings.mlr.press/v37/ioffe15.html).

##### Dropout

- **Dropout: A Simple Way to Prevent Neural Networks from Overfitting (2014)**, Srivastava, Hinton, Krizhevsky, Sutskever, and Salakhutdinov, [@JMLR](http://jmlr.org/papers/v15/srivastava14a.html).

##### Gated Recurrent Unit

- **Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation (2014)**, Cho et al, [@arXiv](https://arxiv.org/abs/1406.1078).

##### Perceptron

- **The Perceptron: A Probabilistic Model for Information Storage and Organization in The Brain (1958)**, Rosenblatt, [@CiteSeerX](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.588.3775).

#### Recommender Systems

##### Collaborative Filtering

- **Using collaborative filtering to weave an information tapestry (1992)**, Goldberg, Nichols, Oki, and Terry, [@CiteSeerX](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.104.3739).

##### Matrix Factorization

- **Application of Dimensionality Reduction in Recommender System - A Case Study (2000)**, Sarwar, Karypis, Konstan, and Riedl, [@CiteSeerX](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.29.8381).
- See also: **Learning Collaborative Information Filters (1998)**, Billsus and Pazzani, [@CiteSeerX](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.487.3789) 🏛️.
- See also: **Netflix Update: Try This at Home (2006)**, Funk, [@author](https://sifter.org/~simon/journal/20061211.html) 📔 🔬.

##### Implicit Matrix Factorization

- **Collaborative Filtering for Implicit Feedback Datasets (2008)**, Hu, Koren, and Volinsky, [@IEEE](https://ieeexplore.ieee.org/document/4781121) 🔒 / [@author](http://yifanhu.net/PUB/cf.pdf) 🔑.

#### Regression

##### Elastic Net

- **Regularization and variable selection via the Elastic Net (2005)**, Zou and Hastie, [@CiteSeer](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.124.4696).

##### Lasso

- **Regression Shrinkage and Selection Via the Lasso (1994)**, Tibshirani, [@CiteSeerX](http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.35.7574).
- See also: **Linear Inversion of Band-Limited Reflection Seismograms (1986)**, Santosa and Symes, [@SIAM](https://epubs.siam.org/doi/10.1137/0907087) 🏛️.

#### Software

##### MapReduce

- **MapReduce: Simplified Data Processing on Large Clusters (2004)**, Dean and Ghemawat, [@ai.google](https://ai.google/research/pubs/pub62).

##### TensorFlow

- **TensorFlow: A system for large-scale machine learning (2016)**, Abadi et al., [@ai.google](https://ai.google/research/pubs/pub45381) + [@author](https://www.tensorflow.org/) 🌐.

##### Torch

- **Torch: A Modular Machine Learning Software Library (2002)**, Collobert, Bengio and Mariéthoz, [@Idiap](http://publications.idiap.ch/index.php/publications/show/712) + [@author](http://torch.ch/) 🌐.
- See also: **Automatic differentiation in PyTorch (2017)**, Paszke et al., [@OpenReview](https://openreview.net/forum?id=BJJsrmfCZ) 🔬+ [@Github](https://github.com/pytorch/pytorch) 💽.

#### Supervised Learning

##### k-Nearest Neighbors

- **Nearest neighbor pattern classification (1967)**, Cover and Hart, [@IEEE](https://ieeexplore.ieee.org/abstract/document/1053964) 🔒.
- See also: **E. Fix and J.L. Hodges (1951): An Important Contribution to Nonparametric Discriminant Analysis and Density Estimation (1989)**, Silverman and Jones, [@JSTOR](https://www.jstor.org/stable/1403796?seq=1) 🔒.

##### Support Vector Machine

- **Support Vector Networks (1995)**, Cortes and Vapnik, [@Springer](https://link.springer.com/article/10.1023/A:1022627411411).

#### Statistics

##### The Bootstrap

- **Bootstrap Methods: Another Look at the Jackknife (1979)**, Efron, [@Project Euclid](https://projecteuclid.org/euclid.aos/1176344552).
- See also: **Problems in Plane Sampling (1949)**, Quenouille, [@Project Euclid](https://projecteuclid.org/euclid.aoms/1177729989) 🏛️.
- See also: **Notes on Bias Estimation (1958)**, Quenouille, [@JSTOR](https://www.jstor.org/stable/2332914?seq=1) 🏛️.
- See also: **Bias and Confidence in Not-quite Large Samples (1958)**, Tukey, [@Project Euclid](https://projecteuclid.org/euclid.aoms/1177706647) 🔬.

### Credits

A special thanks to Alexandre Passos for his comment on [this Reddit thread](https://www.reddit.com/r/MachineLearning/comments/hj4cx/classic_papers_in_machine_learning/c1vt6ny/), as well as the responders to [this Quora post](https://www.quora.com/What-are-some-of-the-best-research-papers-or-books-for-Machine-learning). They provided many great papers to get this list off to a great start.
