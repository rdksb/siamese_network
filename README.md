# Siamese Neural Network 

An efficient implementation of Siamese network, a class of deep learning networks designed to recognize similarity between two inputs. Here the network is used for one-shot learning of handwritten characters from the Omniglot dataset.

## Dataset

The Omniglot dataset consists of 50 alphabets. The number of characters in each alphabet varies considerably
from about 15 up to 40 characters; each character comes with 20 samples (drawn by 20 different people). 

The dataset comes with a standard split into two sets:
* A background set of 30 alphabets with 964 characters (classes).
* An evaluation set of 20 alphabets with 659 classes.
* Each character is a 105x105 binary-valued image. 

The background set is split into training (90%) and validation (10%) data. The split is based on alphabets (27 vs. 3 alphabets). The evaluation set is used for generating test data.


## References

Koch et al., 2015. [Siamese Neural Networks for One-shot Image Recognition](http://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
