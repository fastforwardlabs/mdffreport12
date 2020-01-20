## Background

In this chapter, we provide an overview of approaches to anomaly detection based
on the type of data available, how to evaluate an anomaly detection model and
how each approach constructs a model of normal behaviour and why deep learning
models are valuable. It concludes with a discussion of pitfalls that may occur
while deploying these models.

### Anomaly Detection Approaches

Anomaly detection approaches can be categorized in terms of the type of data
needed to train an anomaly detection model. Within most use cases, it is
expected that anomalous samples represent a very small percentage of the entire
dataset. Thus, even when available data is labeled, normal data samples are more
readily available compared to abnormal samples. This assumption is critical for
most  applications today.  In the following sections, we touch on how the
availability of labeled data impacts the choice of approach.

### Supervised learning 
When learning with supervision, machines rely on examples that illustrate the
relationship between the input features and output. The goal of supervised
anomaly detection algorithms is to incorporate application-specific knowledge
into the anomaly detection process. With sufficient normal and anomalous
examples, the anomaly detection task can be reframed as a classification task
where the machines can learn to accurately predict whether a given example is an
anomaly or not.  That said, for many anomaly detection use cases the proportion
of normal versus anomalous examples is highly imbalanced. And while there may be
multiple anomalous classes, each of them could be quite under-represented. 

![An illustration of supervised learning](figures/supervised_learning.png)

This approach assumes we have labeled examples for all types of anomalies that
could occur and can correctly classify them. In practice, this is usually not
the case, as anomalies can take many different forms, with novel anomalies
emerging at test time. Thus, we need approaches that generalize well and
effectively identify anomalies that have previously been unseen.  

### Unsupervised learning 

When it comes to unsupervised approaches, one does not possess examples that
illustrate the relationship between input features and output. Instead, in this
case, machines learn by finding structure within the input features. Owing to
the frequent lack of labeled anomalous data, unsupervised approaches are more
popular than supervised ones in the anomaly detection field. That said, the
nature of the anomalies is often highly specific to particular kinds of abnormal
activity in the underlying application. In such cases, many of the anomalies
found in a completely unsupervised manner could correspond to noise, and may not
be of any interest to the business.

![An illustration of unsupervised learning](figures/unsupervised_learning.png) 

### Semi-supervised learning

Semi-supervised learning falls between supervised and unsupervised learning
approaches. It includes a set of methods that take advantage of large amounts of
unlabeled data as well as small amounts of labeled data.  Many real world
anomaly detection use cases nicely fit this criteria, in the sense that there
are a huge number of normal examples available but the more unusual or abnormal
classes of interest are insufficient to be effectively learned from. Following
the assumption that most data points within an unlabeled dataset are normal, we
can train a robust model on an unlabeled dataset and evaluate its skill (as well
as  tune the model’s parameters) using a small amount of labeled data ([^1]:
[Deep Semi-Supervised Anomaly Detection](https://arxiv.org/abs/1906.02694)). For
instance, in a network intrusion detection application, one may have examples of
the normal class and some examples of the intrusion classes, but new kinds of
intrusions may often arise with time. 

[^1]: [Deep Semi-Supervised Anomaly Detection](https://arxiv.org/abs/1906.02694)

![An illustration of semi-supervised learning](figures/semisupervised_learning.png)


To give another example, in the case of border security or X-ray screening for
aviation, anomalous items posing a security threat are not commonly encountered.
Exemplary data of anomalies can be difficult to obtain in any quantity, since no
such events may have occurred in the first place. In addition, the nature of any
anomaly posing a potential threat may evolve due to a range of external factors. 

![Exemplary data in certain applications can be difficult to obtain](figures/xray_screening.png)

Such situations may require the determination of both abnormal classes as well
as novel classes, for which little to no labeled data is available. One way to
address this is to use some variant of a supervised or semi-supervised
classification approach. 

## Evaluating Models: Accuracy is not Enough

As mentioned earlier in [Anomaly Detection Approaches](#anomaly-detection-approaches), 
it is expected that the distribution between the normal and abnormal class(es) 
can be very skewed. This is commonly referred to as **class imbalance**. 

A model that learns from such data may not be robust: it may be accurate when
classifying examples within the normal class, but perform poorly when
classifying anomalous examples. 

For example, consider a dataset of 1000 images of luggage that go through a
security checkpoint. 950 images are of normal luggage and 50 are abnormal.
Assuming our model always classifies an image as normal, it can achieve high
overall accuracy for this dataset (95% - i.e,. 95% for normal data and 0% for
abnormal data).

Such a model may also misclassify normal examples as anomalous (**false positives,
FP**), or misclassify anomalous examples as normal ones (**false negatives, FN**).  
As we consider both of these types of errors, it becomes obvious that the
traditional accuracy metric (total number of correct classifications divided by
total classifications) is insufficient in evaluating the skill of an anomaly
detection model.

Two important metrics have been introduced that provide a better measure of
model skill:  precision and recall. Precision is defined as the number of true
positives (TP) divided by the number of true positives (TP) plus the number of
false positives (FP), while recall is the number of true positives (TP) divided
by the number of true positives (TP) plus the number of false negatives (FN).
Depending on the use case or application, it may be desirable to optimize for
either precision or recall.  

Optimizing for precision may be useful when the cost of failure is low, or to
reduce human workload. Optimising for high recall may be more appropriate when
the cost of a false negative is very high (e.g., airport security, where it is
better to flag many items for human inspection in an image (low cost) in order
to avoid the cost of incorrectly admitting a dangerous item on a flight). While
there are several ways to optimize for precision or recall, the manner in which
a threshold is set can be used to reflect the precision and recall preferences
for each specific use case. 

In this section, we have reviewed reasons why an unsupervised or semi-supervised
approach to anomaly detection is desirable, and explored robust metrics for
evaluating these models. In the next section, we focus on these semi-supervised
approaches and discuss how they work.

## Anomaly Detection as Learning Normal Behavior  

The underlying strategy for most approaches to anomaly detection is to first
model normal behavior, and then exploit this knowledge in identifying deviations
(anomalies).  This approach typically falls under the semi-supervised category
and is accomplished across two steps in the anomaly detection loop. The first
step, which we can refer to as the training step, involves building a model of
normal behavior using available data. Depending on the specific anomaly
detection method, this training data may contain both normal and abnormal data
points or only normal data points (see <<Chapter 3: Technical>> for additional
details on AD methods).  Based on this model, an anomaly score is then assigned
to each data point that represents a measure of deviation from normal behavior. 

![Illustration shows the training phase in the anomaly detection loop. Based on
data (which may or may not contain abnormal samples), the AD model learns a
model of normal data and assigns an anomaly score based on
this.](figures/learning_normal_behavior.png)

![Illustration of the test step in the anomaly detection
loop.](figures/anomaly_detection_loop)

The second step in the anomaly detection loop, the test step, introduces the
concept of threshold-based anomaly tagging. Given the range of scores assigned
by the model, we can select a threshold rule that drives the anomaly tagging
process - e.g., scores above a given threshold are tagged as anomalies, while
those below it are tagged as normal. The idea of a threshold is valuable, as it
provides the analyst some easy lever to tune the “sensitivity” of the anomaly
tagging process. Interestingly, while most methods for anomaly detection follow
this general approach, they differ in how they model normal behaviour and
generate anomaly scores. 

![Anomaly scoring](figures/anomaly_scoring)

To further illustrate this process, consider the scenario where the task is to
detect abnormal temperatures (e.g., spikes), given data from the temperature
sensor attached to servers in a data center. We can use a statistical approach
(see table in section [Approaches to Modeling Normal
Behavior](#approaches-to-modeling-normal-behavior) for an overview of common methods). 
In step 1, we assume the samples follow a normal distribution,
and we can use sample data to learn the parameters of this distribution (mean
and variance). We can assign an anomaly score based on a sample’s deviation from
the mean and set a threshold (e.g., any value with more than 3 standard
deviations from the mean is an anomaly).  In step 2, we then tag all new
temperature readings and generate a report.

## Approaches to Modeling Normal Behavior

Given the importance of the anomaly detection task, multiple approaches have
been proposed and rigorously studied over the last few decades. To provide a
high level summary, we categorize the more popular techniques into four main
areas: clustering, nearest neighbour, classification, and statistical [^2] 
[Anomaly Detection, A Survey by Chandola et al 2009](https://dl.acm.org/doi/10.1145/1541880.1541882). 
The Table below provides a summary of examples, assumptions, and anomaly scoring 
strategies taken by approaches within each category.

| AD Method | Assumptions | Anomaly Scoring | Notable Examples |
| --------- | ----------- | --------------- | ---------------  |
| Clustering | Normal data points belong to a cluster (or lie close to its centroid) in the data while anomalies do not belong to any clusters | Distance from nearest cluster centroid | Self Organising Maps (SOM), K-Means Clustering, Expectation Maximization (EM) |
| Nearest Neighbour | Normal data instances occur in dense neighborhoods while anomalous data are far from their nearest neighbors | Distance from Kth nearest neighbour| KNN |
| Classification | <br> 1. A classifier can be learned which distinguishes between normal and anomalous with the given feature space <br> 2. Labeled data exists for normal and abnormal data | A measure of classifier estimate (likelihood) that a data point belongs to the normal class|One Class SVM, Autoencoders, Sequence to Sequence Models|
|Statistical | Given an assumed stochastic model, normal data falls in high probability regions of the model while abnormal data lie in low probability regions | Probability that datapoint lies a high  probability region in the assumed distribution|Regression Models (ARMA, ARIMA), Gaussian Models, GANs, VAEs |

![Approaches to model normal behavior]

The majority of these approaches have been applied to univariate time series
data -  a single datapoint generated by the same process at various time steps
(e.g., readings from a temperature sensor over time) and assume linear
relationships within the data. Examples include KNNs, K-Means Clustering, ARMA,
ARIMA, etc. However, data is increasingly high dimensional (e.g., multivariate
datasets, images, videos) and the detection of anomalies may require the joint
modeling of interactions between each variable. For these sorts of problems,
deep learning approaches (the focus of this report) such as Autoencoders,
Variational Autoencoders, Sequence to Sequence Models, and Generative
Adversarial Networks present some benefits.

## Why Deep Learning for Anomaly Detection

Deep learning approaches, when applied to anomaly detection, offer several
advantages.

### Multivariate, high dimensional data

Deep learning approaches are designed to work with multivariate data of
different data types across each variable. This makes it easy to integrate
information from multiple sources;  it also eliminates challenges associated
with individually modeling anomaly for each variable and aggregating the
results. 

### Modeling interaction between variables

Deep learning approaches work well in jointly modeling the interactions between
multiple variables with respect to a given task. In addition, beyond the
specification of generic hyperparameters (number of layers, units per layer,
etc.), deep learning models require minimal tuning to achieve good results.

### Performance

Deep learning methods offer the opportunity to model complex, non-linear
relationships within data, and leverage this for the anomaly detection task. The
performance of deep learning models can also potentially scale with the
availability of appropriate training data, making them suitable for data-rich
problems. 

### Interpretability

While deep learning methods can be complex (leading to their reputation as black
box models), interpretability techniques such as LIME <<See our previous report:
Interpretability>> and Deep SHAP^[[A Unified Approach to Interpreting Model
Predictions](https://arxiv.org/abs/1705.07874)] provide opportunities to inspect 
their behaviour and make them more interpretable by analysts. 

## What can go wrong?

There is a proliferation of algorithmic approaches that can help tackle an
anomaly detection task and allow us to build solid models, at times even with
just normal samples. But then what is the catch? Do they really work? What could
possibly go wrong?

### Contaminated normal examples

In large scale applications that have huge volumes of data, it is quite possible
that the large unlabeled data is considered as the normal class wherein a small
percentage of examples may actually be anomalous or simply be poor training
examples. And while some models like one-class SVM or isolation forest can
account for this there are others that may not be robust to detecting anomalies 

### Computational complexity

Anomaly detection scenarios can sometimes have low-latency requirements i.e the
ability to  speedily retrain existing models as new data becomes available and
perform inference. This can be computationally expensive at scale, even for
linear models on univariate data. In addition, deep learning models, incur
additional compute costs to estimate their large number of parameters. To
address these compute issues, it is recommended to explore tradeoffs which
balance the frequency of retraining and overall accuracy.    

### Human supervision

One major challenge with unsupervised and semi-supervised approaches is that
they can be noisy and may generate a large amount of false positives. In turn,
false positives incur labour costs associated with human review. Given these
costs, an important goal for anomaly detection systems is to incorporate the
results of human review (as labels) in improving model quality.   

### Definition of anomaly

The boundary between normal and anomalous behavior is often not precisely
defined in several data domains and is continually evolving. Unlike other task
domains where dataset shift occurs sparingly, the anomaly detection systems
should anticipate (frequent) and account for changes in the distribution of the
data. In many cases, this can be achieved by frequent retraining of models.  

### Threshold selection

The process of selecting a good threshold value can be challenging. In a
semi-supervised setting, we have access to a pool of labeled data. Using these
labels (and some domain expertise), we can determine a suitable threshold.
Specifically, we can explore the range of anomaly scores for each data point in
the validation set and select a threshold as the point that yields the best
performance metric (accuracy, precision, recall). In the absence of labeled
data, and if we assume that most data points are normal, we can use statistics
such as standard deviation and percentiles to infer a good threshold. 
