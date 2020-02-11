## Background

In this chapter, we provide an overview of approaches to anomaly detection based
on the type of data available, how to evaluate an anomaly detection model, how 
each approach constructs a model of normal behavior, and why deep learning
models are valuable. We conclude with a discussion of pitfalls that may be 
encountered while deploying these models.

### Anomaly Detection Approaches

Anomaly detection approaches can be categorized in terms of the type of data
needed to train the model. In most use cases, it is
expected that anomalous samples represent a very small percentage of the entire
dataset. Thus, even when labeled data is available, normal data samples are more
readily available than abnormal samples. This assumption is critical for
most applications today. In the following sections, we touch on how the
availability of labeled data impacts the choice of approach.

### Supervised Learning 

When learning with supervision, machines learn a function that maps input features 
to outputs based on example input-output pairs. The goal of supervised
anomaly detection algorithms is to incorporate application-specific knowledge
into the anomaly detection process. With sufficient normal and anomalous
examples, the anomaly detection task can be reframed as a classification task
where the machines can learn to accurately predict whether a given example is an
anomaly or not. That said, for many anomaly detection use cases the proportion
of normal versus anomalous examples is highly imbalanced; while there may be
multiple anomalous classes, each of them could be quite underrepresented. 

![An illustration of supervised learning.](figures/ill-16.png)

This approach assumes that one has labeled examples for all types of anomalies that
could occur and can correctly classify them. In practice, this is usually not
the case, as anomalies can take many different forms, with novel anomalies
emerging at test time. Thus, approaches that generalize well and are more effective at identifying previously unseen anomalies are preferable. 

### Unsupervised learning 

With unsupervised learning, machines do not possess example input-output pairs 
that allow it to learn a function that maps the input features to outputs. Instead, 
they learn by finding structure within the input features. Because, as 
mentioned previously, labeled anomalous data is relatively rare, unsupervised approaches are more
popular than supervised ones in the anomaly detection field. That said, the 
nature of the anomalies one hopes to detect is often highly specific. Thus, many of the anomalies 
found in a completely unsupervised manner could correspond to noise, and may not
be of interest for the task at hand.

![An illustration of unsupervised learning.](figures/ill-17.png) 

### Semi-supervised learning

Semi-supervised learning approaches represent a sort of middle ground, employing 
a set of methods that take advantage of large amounts of
unlabeled data as well as small amounts of labeled data. Many real-world
anomaly detection use cases are well suited to semi-supervised learning, in that 
there are a huge number of normal examples available from which to learn, but 
relatively few examples of the more unusual or abnormal classes of interest. Following
the assumption that most data points within an unlabeled dataset are normal, one
can train a robust model on an unlabeled dataset and evaluate its performance (and 
tune the model’s parameters) using a small amount of labeled data.^[See e.g. Lukas Ruff et al.,
"Deep Semi-Supervised Anomaly Detection" (2019), [arXiv:1906.02694](https://arxiv.org/abs/1906.02694).]  

![An illustration of semi-supervised learning](figures/ill-20.png)

This hybrid approach is well suited to applications like network intrusion detection, 
where one may have multiple examples of the normal class and some examples of intrusion 
classes, but new kinds of intrusions may arise over time.

To give another example, consider X-ray screening for
aviation or border security. Anomalous items posing a security threat are not commonly 
encountered and can take many forms. In addition, the nature of any
anomaly posing a potential threat may evolve due to a range of external factors. 
Exemplary data of anomalies can therefore be difficult to obtain in any useful quantity.

![Exemplary data in certain applications can be difficult to obtain](figures/ill-19.png)

Such situations may require the determination of abnormal classes as well
as novel classes, for which little or no labeled data is available. 
In cases like these, a semi-supervised classification approach that enables detection of both known and previously unseen anomalies is an ideal solution.

### Evaluating Models: Accuracy Is Not Enough

As mentioned in the previous section, in anomaly detection applications 
it is expected that the distribution between the normal and abnormal class(es) 
may be highly skewed. This is commonly referred to as the _class imbalance problem_. 
A model that learns from such skewed data may not be robust; it may be accurate when
classifying examples within the normal class, but perform poorly when
classifying anomalous examples. 

For example, consider a dataset consisting of 1,000 images of luggage passing through a
security checkpoint. 950 images are of normal pieces of luggage, and 50 are abnormal.
A classification model that always classifies an image as normal can achieve high
overall accuracy for this dataset (95%), even though its accuracy rate for classifying
abnormal data is 0%.

Such a model may also misclassify normal examples as anomalous (_false positives_,
FP), or misclassify anomalous examples as normal ones (_false negatives_, FN). 
As we consider both of these types of errors, it becomes obvious that the
traditional accuracy metric (total number of correct classifications divided by
total classifications) is insufficient in evaluating the skill of an anomaly
detection model.

Two important metrics have been introduced that provide a better measure of
model skill: _precision_ and _recall_. Precision is defined as the number of _true
positives_ (TP) divided by the number of true positives plus the number of
_false positives_ (FP), while recall is the number of true positives divided
by the number of true positives plus the number of false negatives (FN).
Depending on the use case or application, it may be desirable to optimize for
either precision or recall.  

Optimizing for precision may be useful when the cost of failure is low, or to
reduce human workload. Optimizing for high recall may be more appropriate when
the cost of a false negative is very high; for example, in airport security, where it is
better to flag many items for human inspection (low cost) in order
to avoid the much higher cost of incorrectly admitting a dangerous item onto a flight. While
there are several ways to optimize for precision or recall, the manner in which
a threshold is set can be used to reflect the precision and recall preferences
for each specific use case. 

::: info
You now have an idea of why an unsupervised or semi-supervised approach 
to anomaly detection is desirable, and what metrics are best to use for
evaluating these models. In the next section, we focus on semi-supervised
approaches and discuss how they work.
:::

### Anomaly Detection as Learning Normal Behavior  

The underlying strategy for most approaches to anomaly detection is to first
model normal behavior, and then exploit this knowledge to identify deviations
(anomalies). This approach typically falls under the semi-supervised learning
category and is accomplished through two steps in the anomaly detection loop. The first
step, referred to as the training step, involves building a model of
normal behavior using available data. Depending on the specific anomaly
detection method, this training data may contain both normal and abnormal data
points, or only normal data points (see [Chapter 3. Deep Learning for Anomaly Detection](#deep-learning-for-anomaly-detection) for additional
details on anomaly detection methods). Based on this model, an anomaly score is then assigned
to each data point that represents a measure of deviation from normal behavior.

![The training step in the anomaly detection loop: based on
data (which may or may not contain abnormal samples), the anomaly detection model learns a
model of normal behavior which it uses to assign anomaly scores.](figures/ill-13.png)

The second step in the anomaly detection loop, the test step, introduces the
concept of threshold-based anomaly tagging. Based on the range of scores assigned
by the model, one can select a threshold rule that drives the anomaly tagging
process; e.g., scores above a given threshold are tagged as anomalies, while
those below it are tagged as normal. The idea of a threshold is valuable, as it
provides the analyst an easy lever with which to tune the “sensitivity” of the anomaly
tagging process. Interestingly, while most methods for anomaly detection follow
this general approach, they differ in how they model normal behavior and
generate anomaly scores. 

![The test step in the anomaly detection loop.](figures/ill-14.png)

To further illustrate this process, consider the scenario where the task is to
detect abnormal temperatures (e.g., spikes), given data from the temperature
sensors attached to servers in a data center. We can use a statistical approach
to solve this problem (see the table in the following section for an overview of
common methods). In step 1, we assume the samples follow a normal distribution,
and we can use sample data to learn the parameters of this distribution (mean
and variance). We assign an anomaly score based on a sample’s deviation from
the mean and set a threshold (e.g., any value more than 3 standard
deviations from the mean is an anomaly). In step 2, we then tag all new 
temperature readings and generate a report.

![Anomaly scoring](figures/ill-15.png)

### Approaches to Modeling Normal Behavior

Given the importance of the anomaly detection task, multiple approaches have
been proposed and rigorously studied over the last few decades. To provide a
high-level summary, we categorize the more popular techniques into four main
areas: clustering, nearest neighbor, classification, and statistical.^[For a survey of existing techniques, see Varun Chandola et al., "Anomaly Detection, A Survey" (2009), _ACM Computing Surveys_ 41(3), https://dl.acm.org/doi/10.1145/1541880.1541882.] 
The following table provides a summary of the assumptions and anomaly scoring 
strategies employed by approaches within each category, and some examples of each.

| Anomaly Detection Method         | Assumptions                                                                                                                                                                          | Anomaly Scoring                                                                             | Notable Examples                                                              |
| ---------         | -----------                                                                                                                                                                          | ---------------                                                                             | ---------------                                                               |
| Clustering        | Normal data points belong to a cluster (or lie close to its centroid) in the data while anomalies do not belong to any clusters.                                                       | Distance from nearest cluster centroid                                                      | Self-organizing maps (SOMs), _k_-means clustering, expectation maximization (EM) |
| Nearest Neighbour | Normal data instances occur in dense neighborhoods while anomalous data are far from their nearest neighbors                                                                         | Distance from _k_th nearest neighbour                                                         | _k_-nearest neighbors (KNN)                                                                         |
| Classification    | <ul><li>A classifier can be learned which distinguishes between normal and anomalous with the given feature space</li><li>Labeled data exists for normal and abnormal classes</li></ul> | A measure of classifier estimate (likelihood) that a data point belongs to the normal class | One-class support vector machines (OCSVMs)                      |
| Statistical       | Given an assumed stochastic model, normal data instances fall in high-probability regions of the model while abnormal data points lie in low-probability regions.| Probability that a data point lies in a high-probability region in the assumed distribution | Regression models (ARMA, ARIMA)   |
| Deep learning       | Given an assumed stochastic model, normal data instances fall in high-probability regions of the model while abnormal data points lie in low-probability regions.| Probability that a data point lies in a high-probability region in the assumed distribution | autoencoders, sequence-to-sequence models, generative adversarial networks (GANs), variational autoencoders (VAEs)                |

The majority of these approaches have been applied to univariate time series
data; a single data point generated by the same process at various time steps
(e.g., readings from a temperature sensor over time); and assume linear
relationships within the data. Examples include _k_-means clustering, ARMA,
ARIMA, etc. However, data is increasingly high-dimensional (e.g., multivariate
datasets, images, videos), and the detection of anomalies may require the joint
modeling of interactions between each variable. For these sorts of problems,
deep learning approaches (the focus of this report) such as autoencoders,
VAEs, sequence-to-sequence models, and GANs present some benefits.

### Why Use Deep Learning for Anomaly Detection?

Deep learning approaches, when applied to anomaly detection, offer several
advantages. First, these approaches are designed to work with 
multivariate and high dimensional data. 
This makes it easy to integrate information from multiple sources, 
and eliminates challenges associated with individually modeling anomalies for each variable and aggregating the
results. Deep learning approaches are also well-adapted to jointly modeling the interactions between
multiple variables with respect to a given task and - beyond the
specification of generic hyperparameters (number of layers, units per layer,
etc.) - deep learning models require minimal tuning to achieve good results.

Performance is another advantage. Deep learning methods offer the opportunity to model complex, nonlinear
relationships within data, and leverage this for the anomaly detection task. The
performance of deep learning models can also potentially scale with the
availability of appropriate training data, making them suitable for data-rich
problems. 

### What Can Go Wrong?

There are a proliferation of algorithmic approaches that can help one tackle an
anomaly detection task and build solid models, at times even with
just normal samples. But do they really work? What could
possibly go wrong? Here are some of the issues that need to be considered:

_Contaminated normal examples_   
In large-scale applications that have huge volumes of data, it's possible
that within the large unlabeled dataset that's considered the normal class, 
a small percentage of the examples may actually be anomalous, or simply be poor training
examples. And while some models (like a one-class SVM or isolation forest) can
account for this, there are others that may not be robust to detecting
anomalies. 
   
_Computational complexity_   
Anomaly detection scenarios can sometimes have low latency requirements; i.e., it may be necessary to be able to speedily retrain existing models as new data becomes available, and
perform inference. This can be computationally expensive at scale, even for
linear models for univariate data. Deep learning models also incur
additional compute costs to estimate their large number of parameters. To
address these issues, it is recommended to explore trade-offs that
balance the frequency of retraining and overall accuracy.    
    
_Human supervision_   
One major challenge with unsupervised and semi-supervised approaches is that
they can be noisy and may generate a large amount of false positives. In turn,
false positives incur labor costs associated with human review. Given these
costs, an important goal for anomaly detection systems is to incorporate the
results of human review (as labels) to improve model quality.

_Definition of anomaly_   
In many data domains, the boundary between normal and anomalous behavior is not precisely
defined and is continually evolving. Unlike in other task
domains where dataset shift occurs sparingly, anomaly detection systems
should anticipate and account for (frequent) changes in the distribution of the
data. In many cases, this can be achieved by frequent retraining of the models.  

_Threshold selection_  
The process of selecting a good threshold value can be challenging. In a
semi-supervised setting (the approaches covered above), one has access to a pool 
of labeled data. Using these labels, and some domain expertise, it is possible to 
determine a suitable threshold. Specifically, one can explore the range of 
anomaly scores for each data point in the validation set and select as a threshold 
the point that yields the best performance metric (accuracy, precision, 
recall). In the absence of labeled data, and assuming that most data points 
are normal, one can use statistics such as standard deviation and percentiles to 
infer a good threshold. 

_Interpretability_  
Deep learning methods for anomaly detection can be complex, leading to their reputation as black
box models. However, interpretability techniques such as LIME (see our previous report,
["Interpretability"](https://blog.fastforwardlabs.com/2017/08/02/interpretability.html)) and [Deep SHAP](https://arxiv.org/abs/1705.07874) provide opportunities for analysts to inspect 
their behavior and make them more interpretable. 
