## Deep Learning for Anomaly Detection

As data becomes high dimensional, it is increasingly challenging to effectively
learn a model of normal behaviour across variables within each model. In this
chapter, we will review a set of relevant deep learning model architectures and
how they can be applied to the task of anomaly detection. As discussed in
[Background](#background), anomaly detection using each of these models is
explored as a function of how they can be applied first in modeling normal
behaviour within data, and then generating an anomaly score.

The deep learning approaches discussed below typically consist of two important
components - an encoder that learns to generate an internal representation of
the input data, and a decoder which attempts to reconstruct the original input
based on this internal representation. While the exact techniques for encoding
and decoding vary across models, the overall benefit they offer is the ability
to learn the distribution of normal input data and construct a measure of
anomaly respectively.  

### Autoencoders

Autoencoders are neural networks designed to learn a low dimensional
representation, given some input data. They consist of two components - an
encoder  which learns to map input data to a low dimensional representation
(termed the bottleneck), and a decoder which learns to map this low dimensional
representation back to the original input data. By structuring the learning
problem in this manner, the encoder network learns an efficient “compression”
function which maps input data to a salient lower dimension representation, such
that the decoder network is able to successfully reconstruct the original input
data. The model is trained by minimizing the reconstruction error: the
difference (mean squared error) between the original input and the reconstructed
output produced by the decoder. In practice, autoencoders have been applied as a
dimensionality reduction technique, as well as in other use cases -  such as
noise removal from images, image colorization, unsupervised feature extraction,
data compression, etc. 

It is important to note that the mapping function learned by an autoencoder is
specific to the training data distribution, i.e., an autoencoder will typically
not succeed at reconstructing data which is significantly different from data it
has seen during training. As we will see later in this section, this property of
learning a distribution specific mapping (as opposed to a generic linear
mapping) is particularly useful for the task of anomaly detection.

![An illustration of the components of autoencoder](figures/ill-1.png)

#### Modeling Normal Behavior and Anomaly Scoring

Applying an autoencoder for anomaly detection follows the general principle of
first modeling normal behaviour and subsequently generating an anomaly score for
a new data sample. To model normal behaviour, we follow a semi-supervised
approach where we train the autoencoder on a normal data sample. This way, the
model learns a mapping function that successfully reconstructs normal data
samples with a very small reconstruction error (the difference between actual
sample and the version reconstructed by the model). This behaviour is replicated
at test time, where the reconstruction error is small for normal data samples,
and large for abnormal data samples. To identify anomalies, we use the
reconstruction error score as an anomaly score and flag samples with
reconstruction errors above a given threshold.

![An illustration of how autoencoders can be applied for anomaly detection. As
the autoencoder attempts to reconstruct abnormal data, it does so in a manner
that is weighted towards normal samples (square shapes). The difference between
what it reconstructs and the input is the reconstruction error. We can specify a
threshold and flag anomalies as samples with reconstruction error above the
given threshold.](figures/ill-2.png)

### Variational Autoencoders

A variational autoencoder (VAE) is an extension of the autoencoder. Similar to
an autoencoder, it consists of both an encoder and a decoder network component,
but also includes important changes in the structure of the learning problem to
accommodate variational inference. As opposed to learning a mapping from input
data to a fixed bottleneck vector (a point estimate), a VAE learns a mapping
from input to a distribution, and learns to reconstruct the original data by
sampling from this distribution using a latent code. In Bayesian terms, our
prior is the distribution of the latent code, our likelihood is the distribution
of the input given the latent code, and our posterior is the distribution of the
latent code, given our input. The components of a VAE serve to derive good
estimates for these terms. 

The encoder network learns the parameters (mean and variance) of a distribution
that outputs a latent code vector, given our input data (posterior). In other
words, we can draw samples of the bottleneck vector that “correspond” to samples
from our input data. The nature of this distribution can vary depending on the
nature of the input data (e.g., while gaussian distributions are commonly used,
bernoulli distributions can be used if the input data is known to be binary).  
On the other hand, the decoder learns a distribution that outputs the original
input data point (or something really close to it), given a latent bottleneck
sample (likelihood). Typically, an isotropic gaussian distribution is used to
model this reconstruction space.  

The VAE model is trained by minimizing the difference between the estimated
distribution produced by the model and the real distribution of the data. This
difference is estimated using the Kullback-Leibler divergence, which quantifies
the distance between two distributions by measuring how much information is lost
when one distribution is used to represent the other. Similar to AEs, VAEs have
been applied in use cases such as unsupervised feature extraction,
dimensionality reduction, image colorization, image denoising, etc. In addition,
given that they use model distributions, they can be leveraged for controlled
sample generation.

The probabilistic Bayesian components introduced in VAEs lead to a few useful
benefits. First, VAEs enable Bayesian inference; essentially, we can now sample
from the learned encoder distribution and decode samples that do not explicitly
exist in the original dataset, but belong to the same data distribution. Second,
VAEs learn a disentangled representation of a data distribution - i.e., a single
unit in the latent code is only sensitive to a single generative factor. This
allows some interpretability of the output of VAEs, as we can vary units in the
latent code for controlled generation of samples. Third, a VAE provides true
probability measures which offer a principled approach to quantifying
uncertainty when applied in practice: e.g., the probability that a new data
point belongs to the distribution of normal data is 80%.  

![An illustration of a variational autoencoder](figures/ill-3.png)

#### Modeling Normal Behavior and Anomaly Scoring

Similar to an autoencoder, we begin by training the VAE on normal data samples.
At test time, we can compute an anomaly score in two ways. First, we can draw
samples of the latent code z from the encoder given our input data, sample
reconstructed values from the decoder using z, and compute a mean reconstruction
error. Anomalies are flagged based on some threshold on the reconstruction
error.

In addition, we can also output a mean and variance parameter from the decoder,
and compute the probability that the new datapoint belongs to the distribution
of normal data on which the model was trained. If the datapoint lies in a low
density region (below some threshold) we then flag that as an anomaly. (We can
do this now because we model a distribution as opposed to a point estimate.)

![An illustration of two approaches to anomaly scoring with a VAE. We can output mean
reconstruction probability i.e., the probability that a sample belongs to the
normal data distribution.](figures/ill-4.png)

### Generative Adversarial Networks

GANs are neural networks designed to learn a generative model of an input data
distribution. In their classic formulation, GANs are composed of a pair of
typically feed-forward neural networks termed a generator G and discriminator D.
Both networks are trained jointly and play a competitive skill game with the end
goal of learning the distribution of input data X.

The generator network G learns a mapping from random noise of a fixed dimension
(Z) to samples X_ that closely resemble members of the input data distribution.
The discriminator D learns to correctly tell apart real samples that originated
in the source data (X) from fake images (X_) that are generated by G. At each
epoch during training, the parameters of G are updated to to maximise its
ability to generate samples that are indistinguishable by D, while the
parameters of  D are updated to maximize its ability to to correctly tell apart
true samples X from generated samples X_. As training progresses, G becomes
proficient at producing samples that are similar to X, and D also upskills on
the task of distinguishing true X from X_. 

In this classic formulation of GANs, while G learns to model the source
distribution X well (it learns to map random noise from Z to the source
distribution), there is no straightforward approach that allows us to harness
this knowledge for controlled inference - i.e,. generate an image that is
similar to a given known image. While we can conduct a broad search over the
latent space with the goal of recovering the most representative latent noise
vector for an arbitrary image, this process is compute intensive and very slow
in practice. 

To address these issues, recent research studies have explored new formulations
of GANs that enable just this sort of controlled, adversarial inference by
introducing an encoder (E) network.^[[BiGAN, Donahue,
2016](https://arxiv.org/abs/1605.09782)] ^[[GANomaly AkCay et al 2018](
https://arxiv.org/abs/1805.06725)] In simple terms, the encoder learns the
reverse mapping of the generator G; it learns to generate a fixed vector Z_,
given an image. Given this change, the input to the discriminator is also
modified - the discriminator now takes in pairs of input that include the latent
representation  (Z, and Z_), in addition to the data samples (X and X_).  The
encoder E is then jointly trained with the generator G; G learns an induced
distribution that outputs samples of X given a latent code z, while E learns an
induced distribution that outputs Z, given a sample X.

Again, the mappings learned by components in the GAN network are specific to the
data used in training. For example, the generator component of a GAN trained on
images of cars will always output an image that looks like a car given any
latent code. At test time, we can leverage this property to infer how different
a given input sample is from the data distribution on which the model was
trained.

![An illustration of a traditional GAN](figures/ill-5.png)

![An illustration of a BiGAN - a
traditional GAN extended to include an
encoder.](figures/ill-6.png)

#### Modeling Normal Behavior and Anomaly Scoring

To model normal behaviour, we train a BiGAN on normal data samples. At the end
of the training process, we have an encoder E that has learned a mapping from
data sample (X) to latent code space (Z_), a discriminator D that has learned to
distinguish real from generated data, and a Generator G that has learned a
mapping from latent code space  to sample space. Note that these mappings are
specific to the distribution of normal data that has been seen during training.
At test time, we perform the following steps to generate an anomaly score for a
given sample X. First, we obtain a latent space value z from the encoder given
X, which is fed to the generator and yields a sample X_. Next, we can compute an
anomaly score based on the reconstruction loss (difference between X and X_) and
the discriminator loss (cross entropy loss or feature differences in the last
dense layer of the discriminator given both X and X_)

![An illustration of a biGAN applied to the task of AD](figures/ill-7.png)

### Sequence to Sequence Models

Sequence to sequence models are a class of neural networks mainly designed to
learn mappings between data that are best represented as sequences. Data
containing sequences can be challenging as each token in a sequence may have
some form of temporal dependence on other tokens, a relationship that has to be
modeled to achieve good results. For example, consider the task of language
translation where a sequence of words in one language needs to be mapped to a
sequence of words in a different language. To excel at these tasks, a model must
take into consideration the (contextual) location of each word/token within the
broader sentence in order to generate an appropriate translation (See our
previous report on Natural Language Processing to learn more about this area.) 

On a high level, sequence to sequence models typically consist of an encoder E
which generates a hidden representation of the input tokens, and a decoder D,
which takes in the encoder representation and sequentially generates a set of
output tokens. Traditionally, the encoder and decoder are composed of LSTM
blocks which are particularly suitable for modelling temporal relationships
within input data tokens. 

While sequence to sequence models excel at modeling data with temporal
dependence, they can be slow during inference (each individual token in the
model output is sequentially generated at each time step, where the total number
of steps is the length of the output token).

We can use this encoder-decoder structure for anomaly detection by revising the
sequence to sequence model task to function like an autoencoder - train the
model to output the same tokens as the input (shifted by 1). This way, the
encoder learns to generate a hidden representation that allows the decoder to
reconstruct input data that is similar to examples seen in the training dataset.   

![An illustration of sequence to sequence models](figures/ill-8.png)

#### Modeling Normal Behavior and Anomaly Scoring

To identify anomalies, we take a semi supervised approach where we train the
sequence to sequence model on normal data. At test time, we can compare the
difference (mean square error) between output sequence generated by the model to
its input. Similar to other approaches discussed above, we can use this value as
an anomaly score.

In this section, we discuss One-Class Support Vector Machines (OCSVM), a
non-deep-learning technique which we will later use (see [Prototype](#prototype))
as a baseline approach. 

Traditionally, the goal of classification approaches is to help distinguish
between a number of classes using some training data. However, consider a
scenario where we have data for only one class, and the goal is to determine
whether test data samples are similar to the training samples or not. One-class
SVMs were introduced exactly for this sort of task - novelty detection - or the
detection of novel samples. SVMs have been very popular for classification, and
introduced the use of kernel functions to create non-linear decision boundaries
(hyperplanes) by projecting data into a higher dimension. Similarly, OCSVMs
learn a decision function which specifies regions in the input data space where
the probability density of the data is high. The model is trained with various
hyperparameters:

- nu, the outliers_fraction, specifies the proportion of outliers (data samples
that do not belong to our class of interest) which we expect in our data  
- kernel specifies the kernel type to be used in the algorithm. Examples of kernel
functions include RBF, Poly and Linear, kernels. This enables SVMs to use a
non-linear function to project the input data to a higher dimension.
- gamma is a parameter of the RBF kernel type and controls the influence of
individual training samples - this affects the "smoothness" of the model.

![An OCSVM classifier learns a decision boundary around data seen during
training.](figures/ill-9.png)

#### Modeling Normal Behavior and Anomaly Scoring

To apply OCSVM for anomaly detection, we train an OCSVM model using normal data,
or data containing some abnormal samples. Within most implementations of OCSVM,
the model returns an estimate of how similar a data point is to the data samples
seen during training. This estimate may be the distance from the decision
boundary (the separating hyperplane) or discrete class values (+1 for data that
is similar and -1 for data that is not). Both of these scores can be used as an
anomaly score.

![At test time, An OCSVM model classifies data points outside the learned
decision boundary as anomalies (assigned class of -1).](figures/ill-10.png)

### Additional Considerations

#### Anomalies as Rare Events

For the training approaches discussed above, we operate on the assumption of the
availability of “normal” labeled data, which is then used to learn a model of
normal behaviour. In practice, it is often the case that labels do not exist or
can be expensive to obtain. However, it is also a frequent observation that
anomalies (by definition) are relatively infrequent events and therefore
constitute a small percentage of the entire event dataset (e.g., the occurrence
of fraud, machine failure, cyber attacks, etc.). Based on our experiments (see
[Prototype](#prototype) for more discussion), the neural network approaches
discussed above remain robust in the presence of small amounts of anomaly (less
than 10%). This is mainly because introducing a small percentage of anomalies
does not significantly affect the network’s model of normal behaviour.  For
scenarios where anomalies are known to occur sparingly, our experiment results
relax the requirement of assembling a dataset of labeled normal samples for
training.

#### Discretizing Data and Handling Stationarity

To apply deep learning approaches for anomaly detection (as with any other
task), we need to construct a dataset of training samples. For problem spaces
where data is already discrete, we can use the data as is (e.g., a dataset of
images of wall panels, where the task is to find images containing abnormal
walls). When data exists as a time series, we can construct our dataset by
discretizing the series into training samples. Typically this involves slicing
the data into chunks with comparable statistical properties. For example, given
a series of recordings generated by a datacenter temperature sensor, we can
discretize the data into daily or weekly time slices and construct a dataset
based on these chunks. This becomes our basis of anomaly comparison (e.g., the
temperature pattern for today is anomalous compared to patterns for the last 20
days). The choice of the discretization approach (e.g., daily, weekly,
averages.) will often require some domain expertise; however, the one
requirement is that each discrete sample is comparable. For example, given that
temperatures may spike during work hours compared to non-work hours, it may be
challenging to discretize this data by hour as these hours exhibit different
statistical properties. 

![The figure above illustrates temperature readings for a datacenter over
several days and how they can be discretized (sliced) into daily 24hr readings
and labelled (0 for a normal day temperature, 1 for abnormal temperature) to
construct a dataset.](figures/ill-11.png)

This notion of constructing a dataset of comparable samples is related to the
idea of stationarity. A stationary series is one in which properties of the data
(mean, variance) do not vary with time. Data containing trends (e.g., rising
global temperatures) or with seasonality (e.g., the hourly temperature within
each day) represents examples of non-stationary data. These need to be handled
during discretization. We can remove trends by applying a differencing function
to the entire dataset. To handle seasonality, we can explicitly include
information on seasonality as a feature of each discrete sample (e.g., to
discretize by hour, we can attach a categorical variable representing hour of
day). A common misconception regarding the application of neural networks (e.g.,
LSTMs) is that they automatically learn/model properties of the data useful for
predictions (including trends and seasonality). However, the extent to which
this is possible is dependent on how much of this behaviour is represented in
each training sample (e.g., to automatically account for trends or patterns
across the day, we can discretize data by hour with an additional categorical
feature for hour_of_day, or discretize by day (24 features for each hour).   

Note: For most ML algorithms, it is a requirement that samples be independent
and identically distributed. Ensuring we construct comparable samples (handle
trends and seasonality) from time series data allows us to satisfy the identical
requirement but not the independence requirement. This can impact model
performance. In addition, by constructing our dataset, there is a chance that
the learned model may perform poorly in correctly predicting output values that
lie outside the range of values (distribution) seen during training i.e a
distribution shift. This greatly amplifies the need to - and frequency of -
retraining the model as new data arrives; and complicates the model deployment
process. In general, discretization should be applied with care. 

### Selecting a Model

There are several factors that can influence the primary approach taken when it
comes to detecting anomalies. These include the data properties (time series vs
non-time series, stationary vs non-stationary, univariate versus multivariate,
low dimensional vs high dimensional), and business or application requirements (latency, uncertainty
reporting, and accuracy). More importantly, deep learning methods
are not always the best approach! To provide a framework for navigating this
space, we offer the following recommendations (footnote: linear models are
mentioned below and refer to approaches such as AR, MA, ARMA, ARIMA, SARIMA, VAR
models).

#### Data Properties 

##### Time series data
As discussed in the previous sections, it is important to correctly discretize
data as well as handle stationarity before training a model.  

##### Univariate vs Multivariate
Deep learning methods are recommended for high dimensional data and work well in
modeling the interactions between multiple variables. These include data that
has a wide range of features or high dimensional data such as images. For most
univariate datasets, linear models are both fast and accurate and thus
recommended.

#### Business Requirements
##### Latency
Deep learning models are slower compared to linear models. For scenarios which
include high data volume, and low latency requirements, linear models are
recommended (e.g., detecting anomalies in the authentication requests for
200,000 work sites, with each machine generating 500 requests per second). 

##### Accuracy
Deep learning approaches tend to be robust, providing better accuracy, precision
and recall.

##### Uncertainty
For scenarios where it is a requirement to provide a principled estimate of
uncertainty for each anomaly classification, deep learning methods such as VAEs
and BiGANs are recommended.

### General Considerations in Selecting a Deep Learning Approach 
When the (discretized) data contains sequences with temporal dependencies, a
sequence to sequence model can model these relationships, yielding better
results. For scenarios requiring principled estimates of uncertainty, a VAE and
GAN based approaches are suitable. For scenarios where the data is images, AE’s
VAEs and GANs designed with convolution blocks are suitable.  

![A flow chart illustrating steps for selecting an approach to anomaly
detection.](figures/ill-12.png)


| Model                     | Pros                                                                                                                                                                                                                                                                       | Cons                                                                                                                                                                                                                                                                                         |
| ----                      | ----                                                                                                                                                                                                                                                                       | ----                                                                                                                                                                                                                                                                                         |
| AutoEncoder               | <ul><li>Flexible approach to modeling complex non-linear patterns in data</li>                                                                                                                                                                                             | <ul><li>Does not support variational inference (estimates of uncertainty)</li><li>Requires a large dataset for training</li></ul>                                                                                                                                                            |
| Variational AutoEncoder   | <ul><li>Supports variational inference (probabilistic measure of uncertainty)</li></ul>                                                                                                                                                                                    | <ul><li>Requires a large amount of training data, training can take a while</li>                                                                                                                                                                                                             |
| GAN (BiGAN)               | <ul><li>Supports variational inference (probabilistic measure of uncertainty) </li><li>Use of discriminator signal allows better learning of data manifold (useful for high dimensional image data).</li><li>Performs well for  high dimensional data (images)</li></ul> | <ul><li>Requires a large amount of training data, training can take a while </li><li>Training can be unstable (GAN mode collapse)</li></ul>                                                                                                                                                  |
| Sequence to Sequence Mode | <ul><li>Well suited for data with temporal components (e.g., discretized time series data)</li></ul>                                                                                                                                                                       | <ul><li>Slow inference (compute scales with sequence length which needs to be fixed)</li><li>Training can be slow</li><li>Limited accuracy when data contains features with no temporal dependence</li><li>Supports variational inference (probabilistic measure of uncertainty)</li></ul> |
| One Class SVM             | <ul><li>Does not require a large amount of data</li><li>Fast to train</li><li>Fast inference time</li></ul>                                                                                                                                                                | <ul><li>Limited capacity in capturing complex relationships within data</li><li>Requires kernel selection and other parameters (nu, gamma) that need to be carefully tuned.</li><li>Does not model a probability distribution, harder to compute estimates of confidence.</li></ul>         |
