## Prototype

In this section, we provide an overview of the data and experiments used to evaluate each of the approaches mentioned in the <<technical>> chapter. We also introduce two prototypes we built to demonstrate results from the experiments and how we designed each prototype.  


### Datasets

#### KDD

The  [KDD network intrusion dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) is a dataset of TCP connections that have been labelled as normal or representative of network attacks. 

> A connection is a sequence of TCP packets starting and ending at some well defined times, between which data flows to and from a source IP address to a target IP address under some well defined protocol.”

These attacks fall into four main categories - denial of service, unauthorized access from a remote machine, unauthorized access to local superuser privileges, and surveillance e.g. port scanning.  Each TCP connection is represented as a set of attributes or features (derived based on domain knowledge) pertaining to each connection such as the number  of failed logins, connection duration, data bytes from source to destination etc. To make the data more realistic, the test portion of the dataset contains attack types that are not in the train portion. 


#### ECG5000
The  [ECG5000](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000) contains examples of ECG signals from a patient. Each data sample, which corresponds to an extracted heartbeat containing 140 points, has been labelled as normal or being indicative of heart conditions related to congestive heart failure. Given an ECG signal sample, the task is to predict if it is normal or abnormal. ECG5000 is well suited to a prototype for a few reasons — it is visual (signals can be visualized easily) and it is based on real data associated with a concrete use case (heart disease detection). While the task itself is not extremely complex, the data is multidimensional (140 values per sample which allows us demonstrate the value of a deep model), but small enough to rapidly train and run inference. 
 

### Benchmarking Experiment Setup 
We sought to compare each of the models discussed earlier using the KDD dataset. We preprocessed the data to keep only 18 continuous features (for easy reproducibility). Feature scaling (0-1 minmax scaling) is also applied to the data; scaling parameters are learned from training data and then applied to test data. We then trained each model using normal samples (97,278 samples) and evaluated it with a randomly selected subset of the test data (5,000 normal samples, and 5,000 abnormal samples). 

We implemented each model using comparable parameters (see table below <<>>) that allow us to benchmark them in terms of training (mean time per training epoch, mean training time to best accuracy), inference (mean inference time) and storage (size of weights, number of parameters) metrics. The deep learning models (AE, VAE, Seq2seq, BiGAn) were implemented in Tensorflow (keras api); each model trained till best accuracy on the same validation dataset, using the [Adam optimizer](https://keras.io/optimizers/) and a learning rate of  0.01.  OCSVM was implemented using the [Sklearn OCSVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html) library.  Additional details on the parameters for each model are summarized in the table below for reproducibility.

Table ...


In terms of training, we found that the ocsvm and the autoencoder was the fastest to train to peak accuracy. GANs have known stability issues (foot note: Improved Techniques for Training GANs  https://arxiv.org/abs/1606.03498) and can be challenging to train. Overall, the BiGAN approach, required more training epochs to arrive at stable results compared to the other deep methods. OC SVM had the fastest inference time. In terms of storage, the bigan model has the largest number of parameters and overall size of weights.


###  Web Application Prototypes
We built two prototypes that demonstrate results and insights from our experiments. The first prototype -- is built on on the KDD dataset used in the experiments above and is a visualization of the performance of 4 approaches to anomaly detection. The second prototype is an interactive explainer that focuses on the autoencoder model and results from applying it to detecting anomalies in ECG data.

#### Prototype I - Grant to Add Prototype Name
This prototype is built on the KDD network intrusion dataset 



#### Prototype II - Anomagram

This section describes Anomagram - an interactive web based experience where the user can build, train and evaluate an autoencoder to detect anomalous ECG signals. It utilizes the ECG5000 dataset mentioned above <<>>.


##### UX Goals for Anomagram

Anomagram is designed as part of a growing area interactive visualizations (see Neural Network Playground [3], ConvNet Playground, GANLab, GAN dissection, etc) that help communicate technical insights on how deep learning models work. It is entirely browser based and  implemented in Tensorflow.js. This way, users can explore live experiments with no installations required. Importantly, Anomagram moves beyond the user of toy/synthetic data and situates learning within the context of a concrete task (anomaly detection for ECG data). The overall user experience goals for Anomagram are summarized as follows. 

Goal 1: Provide an introduction to Autoencoders and how they can be applied to the task of anomaly detection. This is achieved via the introduction module. This entails providing definitions of concepts (reconstruction error, thresholds etc) paired with interactive visualizations that demonstrate concepts (e.g. an interactive visualization for inference on test data, a visualization of the structure of an autoencoder, a visualization of error histograms as training progresses, etc). 


Goal 2: Provide an interactive, accessible experience that supports technical learning by doing. This is mostly accomplished within the train a model module and is designed for users interested in additional technical depth. It entails providing a direct manipulation interface that allows the user to specify a model (add/remove layers and units within layers), modify model parameters (training steps, batchsize, learning rate, regularizer, optimizer, etc), modify training/test data parameters (data size, data composition), train the model, and evaluate model performance (visualization of accuracy, precision, recall, false positive, false negative, ROC etc metrics) as each parameter is changed. Who should use Anomagram? Anyone interested in an accessible way to learn about autoencoders and anomaly detection . Useful for educators (tool to support guided discussion of the topic), entry level data scientists, and non-ML experts (citizen data scientists, software developers, designers etc).

#####  Interface Affordances and Insights 
This section discusses some explorations the user can perform with Anomagram, and some corresponding insights. 

Craft (Adversarial) Input: Anomalies by definition can take many different and previously unseen forms. This makes the assessment of anomaly detection models more challenging. Ideally, we want the user to conduct their own evaluations of a trained model e.g. by allowing them to upload their own ECG data. In practice, this requires the collection of digitized ECG data with similar preprocessing (heartbeat extraction) and range as the ECG5000 dataset used in training the model. This is challenging. The next best way to allow testing on examples contributed by the user is to provide a simulator — hence the draw your ECG data feature. This provides a (html) canvas on which the user can draw signals and observe the model’s behaviour. Drawing strokes are converted to an array, with interpolation for incomplete drawings (total array size=140) and fed to the model. While this approach has limited realism (users may not have sufficient domain expertise to draw meaningful signals), it provides an opportunity to craft various types of (adversarial) samples and observe the model’s performance. 
Insights: The model tends to expect reconstructions that are close to the mean of normal data samples. Using the Draw your ecg data feature, the user can draw (adversarial) examples of input data and observe model predictions/performance.
 


Visually Compose a Model: Users can intuitively specify an autoencoder architecture using a direct manipulation model composer. They can add layers and add units to layers using clicks. This architecture is then used to specify the model’s parameters each time the model is compiled. This follows a similar approach used in “A Neural Network Playground”[3]. The model composer connector lines are implemented using the leaderline library. Relevant lines are redrawn or added as layers are added or removed from the model. Insights: There is no marked difference between a smaller model (1 layer) and a larger model (e.g. 8 layers) for the current task. This is likely because the task is not especially complex (a visualization of PCA points for the ECG dataset suggests it is linearly separable). Users can visually compose the autoencoder model — add remove layers in the encoder and decoder. To keep the encoder and decoder symmetrical, add/remove operations on either is mirrored. 


Effect of Learning Rate, Batchsize, Optimizer, Regularization: The user can select from 6 optimizers (Adam, Adamax, Adadelta, Rmsprop, Momentum, Sgd), various learning rates, and regularizers (l1, l2, l1l2). 
Insights: Adam reaches peak accuracy with less steps compared to other optimizers. Training time increases with no benefit to accuracy as batchsize is reduced (when using Adam). A two layer model will quickly overfit on the data; adding regularization helps address this to some extent. Try them out! 

Effect of Threshold Choices on Precision/Recall Earlier in this report (see background section on <<Is Accuracy Enough>>) we highlight the importance of metrics such as precision and recall and why accuracy is not enough. To support this discussion, the user can visualize how threshold choices impact each of these metrics. 
Insights: As threshold changes, accuracy can stay the same but, precision and recall can vary. This further illustrates how the threshold can be used by an analyst as a lever to reflect their precision/recall preferences. 



Effect of Data Composition: We may not always have labelled normal data to train a model. However, given the rarity of anomalies (and domain expertise), we can assume that unlabelled data is mostly comprised of normal samples. However, this assumption raises an important question - does model performance degrade with changes in the percentage of abnormal samples in the dataset? In the train a model section, you can specify the percentage of abnormal samples to include when training the autoencoder model. 
Insights: We see that with 0% abnormal data, the model AUC is ~96%. Great! At 30% abnormal sample composition, AUC drops to ~93%. At 50% abnormal data points, there is just not enough information in the data that allows the model to learn a pattern of normal behaviour. It essentially learns to reconstruct normal and abnormal data well and mse is no longer a good measure of anomaly. At this point, model performance is only slightly above random chance (AUC of 56%).