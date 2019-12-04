### Landmark Papers in Machine Learning

*This document attempts to collect the papers which developed important techniques in machine learning. Research is a collaborative process, discoveries are made independently, and the difference between the original version and a precursor can be subtle, but I’ve done my best to select the papers that I think are novel or significant.*

*My opinions are by no means the final word on these topics. Please create an issue or pull request if you have a suggestion.*

#### Key

| Icon |                                                              |
| ---- | ------------------------------------------------------------ |
| 🔒    | Paper behind paywall. In some cases, I provide an alternative link to the paper *if* it comes directly from one of the authors. |
| 🔑    | Freely available version of paywalled paper, directly from the author. |
| 💽    | Code associated with the paper.                              |
| 🏛️    | Precursor or historically relevant paper. This may be a fundamental breakthrough that paved the way for the concept in question to be developed. |
| 🔬    | Iteration, advancement, elaboration, or major popularization of a technique. |
| 🌐    | Not an academic/technical paper/article.                     |

Papers proceeded by “See also” indicate either additional historical context or else major developments, breakthroughs, or applications.

* [Association Rule Learning](#association-rule-learning)
* [Clustering](#clustering)
  + [k-Nearest Neighbors](#k-nearest-neighbors)
* [Datasets](#datasets)
  + [ImageNet](#imagenet)
* [Decision Trees](#decision-trees)
* [Ensemble Methods](#ensemble-methods)
  + [AdaBoost](#adaboost)
  + [Bagging](#bagging)
  + [Generative Adversarial Network](#generative-adversarial-network)
  + [Gradient Boosting](#gradient-boosting)
  + [Random Forest](#random-forest)
* [Games](#games)
  + [AlphaGo](#alphago)
  + [Deep Blue](#deep-blue)
* [Optimization](#optimization)
  + [Expectation Maximization](#expectation-maximization)
  + [Stochastic Gradient Descent](#stochastic-gradient-descent)
* [Miscellaneous](#miscellaneous)
  + [Non-negative Matrix Factorization](#non-negative-matrix-factorization)
  + [PageRank](#pagerank)
  + [DeepQA (Watson)](#deepqa--watson-)
* [Natural Language Processing](#natural-language-processing)
  + [Latent Semantic Analysis](#latent-semantic-analysis)
  + [Word2Vec](#word2vec)
* [Neural Networks](#neural-networks)
  + [Back-propagation](#back-propagation)
  + [Convolutional Neural Network](#convolutional-neural-network)
  + [Dropout](#dropout)
  + [Inception (classification/detection CNN)](#inception--classification-detection-cnn-)
  + [Long Short-Term Memory (LSTM)](#long-short-term-memory--lstm-)
  + [Perceptron](#perceptron)
* [Recommender Systems](#recommender-systems)
  + [Collaborative Filtering](#collaborative-filtering)
  + [Matrix Factorization](#matrix-factorization)
  + [Implicit Matrix Factorization](#implicit-matrix-factorization)
* [Regression](#regression)
  + [Lasso](#lasso)
* [Support Vector Machine](#support-vector-machine)

- [Credits](#credits)

#### Association Rule Learning

- **Mining Association Rules between Sets of Items in Large Databases (1993)**, Agrawal, Imielinski, and Swami, [@CiteSeerX](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.40.6984). 

- See also: **The GUHA method of automatic hypotheses determination (1966)**, Hájek, Havel, and Chytil, [@Springer](https://link.springer.com/article/10.1007/BF02345483) 🔒 🏛️.

#### Clustering

##### k-Nearest Neighbors

- **Nearest neighbor pattern classification (1967)**, Cover and Hart, [@IEEE](https://ieeexplore.ieee.org/abstract/document/1053964) 🔒. 

- See also: **E. Fix and J.L. Hodges (1951): An Important Contribution to Nonparametric Discriminant Analysis and Density Estimation (1989)**, Silverman and Jones, [@JSTOR](https://www.jstor.org/stable/1403796?seq=1) 🔒.

#### Datasets

##### ImageNet

- **ImageNet: A large-scale hierarchical image database (2009)**, Deng et al., [@IEEE](https://ieeexplore.ieee.org/document/5206848) 🔒 / [@author](http://www.image-net.org/papers/imagenet_cvpr09.pdf) 🔑.
- See also: **ImageNet Large Scale Visual Recognition Challenge (2015)**, [@Springer](https://link.springer.com/article/10.1007/s11263-015-0816-y) 🔒 / [@arXiv](https://arxiv.org/abs/1409.0575) 🔑 + [@author](http://www.image-net.org/challenges/LSVRC/) 🌐.

#### Decision Trees

- **Induction of Decision Trees (1986)**, Quinlan, [@Springer](https://link.springer.com/article/10.1007/BF00116251).

#### Ensemble Methods

##### AdaBoost

- **A Decision-Theoretic Generalization of on-Line Learning and an Application to Boosting (1997—published as abstract in 1995)**, Freund and Schapire, [@CiteSeerX](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.32.8918). 

- See also: **Experiments with a New Boosting Algorithm (1996)**, Freund and Schapire, [@CiteSeerX](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.133.1040) 🔬.

##### Bagging

- **Bagging Predictors (1996)**, Breiman, [@Springer](https://link.springer.com/article/10.1023/A:1018054314350).

##### Generative Adversarial Network

- **General Adversarial Nets (2014)**, Goodfellow et al., [@NIPS](https://papers.nips.cc/paper/5423-generative-adversarial-nets) + [@Github](https://github.com/goodfeli/adversarial) 💽.

##### Gradient Boosting

- **Greedy function approximation: A gradient boosting machine (2001)**, Friedman, [@Project Euclid](https://projecteuclid.org/euclid.aos/1013203451).

##### Random Forest

- **Random Forests (2001)**, Breiman and Schapire, [@CiteSeerX](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.125.5395).

#### Games

##### AlphaGo

- **Mastering the game of Go with deep neural networks and tree search (2016)**, Silver et al., [@Nature](https://www.nature.com/articles/nature16961).

##### Deep Blue

- **IBM's deep blue chess grandmaster chips (1999)**, Hsu, [@IEEE](https://ieeexplore.ieee.org/abstract/document/755469) 🔒.
- See also: **Deep Blue (2002)**, Campbell, Hoane, and Hsu, [@ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0004370201001291?via%3Dihub) 🔒.

#### Optimization

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

#### Natural Language Processing

##### Latent Semantic Analysis

- **Indexing by latent semantic analysis (1990)**, Deerwater, Dumais, Furnas, Landauer, and Harshman, [@CiteSeerX](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.108.8490).

##### Word2Vec

- **Efficient Estimation of Word Representations in Vector Space (2013)**, Mikolov, Chen, Corrado, and Dean, [@arXiv](https://arxiv.org/abs/1301.3781) + [@Google Code](https://code.google.com/archive/p/word2vec/) 💽.

#### Neural Networks

##### Back-propagation

- **Learning representations by back-propagating errors (1986)**, Rumelhart, Hinton, and Williams, [@Nature](https://www.nature.com/articles/323533a0) 🔒.
- See also: **Backpropagation Applied to Handwritten Zip Code Recognition (1989)**, LeCun et al., [@IEEE](https://ieeexplore.ieee.org/document/6795724) 🔒🔬 / [@author](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf) 🔑.

##### Convolutional Neural Network

- **Gradient-based learning applied to document recognition (1998)**, LeCun, Bottou, Bengio, and Haffner, [@IEEE](https://ieeexplore.ieee.org/document/726791/) 🔒 / [@author](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) 🔑.
- See also: **Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position (1980)**, Fukushima, [@Springer](https://link.springer.com/article/10.1007/BF00344251) 🏛️.
- See also: **Phoneme recognition using time-delay neural networks (1989)**, Waibel, Hanazawa, Hinton, Shikano, and Lang, [@IEEE](https://ieeexplore.ieee.org/document/21701) 🏛️.

##### Dropout

- **Dropout: A Simple Way to Prevent Neural Networks from Overfitting (2014)**, Srivastava, Hinton, Krizhevsky, Sutskever, and Salakhutdinov, [@JMLR](http://jmlr.org/papers/v15/srivastava14a.html).

##### Inception (classification/detection CNN)

- **Going Deeper with Convolutions (2014)**, Szegedy et al., [@ai.google](https://ai.google/research/pubs/pub43022) + [@Github](https://github.com/google/inception) 💽.
- See also: **Rethinking the Inception Architecture for Computer Vision (2016)**, Szegedy, Vanhoucke, Ioffe, Shlens, and Wojna, [@ai.google](https://ai.google/research/pubs/pub44903) 🔬.
- See also: **Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning (2016)**, Szegedy, Ioffe, Vanhoucke, and Alemi, [@ai.google](https://ai.google/research/pubs/pub45169) 🔬.

##### Long Short-Term Memory (LSTM)

- **Long Short-term Memory (1995)**, Hochreiter and Schmidhuber, [@CiteSeerX](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.13.634).

##### Perceptron

- **The Perceptron: A Probabilistic Model for Information Storage and Organization in The Brain (1958)**, Rosenblatt, [@CiteSeerX](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.588.3775).

#### Recommender Systems

##### Collaborative Filtering

- **Using collaborative filtering to weave an information tapestry (1992)**, Goldberg, Nichols, Oki, and Terry, [@CiteSeerX](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.104.3739).

##### Matrix Factorization

- **Application of Dimensionality Reduction in Recommender System - A Case Study (2000)**, Sarwar, Karypis, Konstan, and Riedl, [@CiteSeerX](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.29.8381).
- See also **Learning Collaborative Information Filters (1998)**, Billsus, Pazzani, [@CiteSeerX](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.487.3789) 🏛️.
- See also **Netflix Update: Try This at Home (2006)**, Funk, [@author](https://sifter.org/~simon/journal/20061211.html) 🌐 🔬.

##### Implicit Matrix Factorization

- **Collaborative Filtering for Implicit Feedback Datasets (2008)**, Hu, Koren, and Volinsky, [@IEEE](https://ieeexplore.ieee.org/document/4781121) 🔒 / [@author](http://yifanhu.net/PUB/cf.pdf) 🔑.

#### Regression

##### Lasso

- **Regression Shrinkage and Selection Via the Lasso (1994)**, Tibshirani. [@CiteSeerX](http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.35.7574). 

- See also: **Linear Inversion of Band-Limited Reflection Seismograms (1986)**, Santosa and Symes, [@SIAM](https://epubs.siam.org/doi/10.1137/0907087) 🏛️.

#### Support Vector Machine

- **Support Vector Networks (1995)**, Cortes and Vapnik, [@Springer](https://link.springer.com/article/10.1023/A:1022627411411).

### Credits

A special thanks to Alexandre Passos for his comment on [this Reddit thread](https://www.reddit.com/r/MachineLearning/comments/hj4cx/classic_papers_in_machine_learning/c1vt6ny/), as well as the responders to [this Quora post](https://www.quora.com/What-are-some-of-the-best-research-papers-or-books-for-Machine-learning). They provided many great papers to get this list off to a great start.
