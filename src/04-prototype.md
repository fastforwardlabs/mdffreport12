## Prototype

In this section, we provide an overview of the data and experiments used to evaluate each of the approaches mentioned in the <<technical>> chapter. We also introduce two prototypes we built to demonstrate results from the experiments and how we designed each prototype.  


### Datasets

#### KDD

The  [KDD network intrusion dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) is a dataset of TCP connections that have been labeled as normal or representative of network attacks. 

> A connection is a sequence of TCP packets starting and ending at some well-defined times, between which data flows to and from a source IP address to a target IP address under some well-defined protocol.”

These attacks fall into four main categories - denial of service, unauthorized access from a remote machine, unauthorized access to local superuser privileges, and surveillance, e.g., port scanning. Each TCP connection is represented as a set of attributes or features (derived based on domain knowledge) pertaining to each connection such as the number of failed logins, connection duration, data bytes from source to destination, etc. The dataset is comprised of a training set (97278 normal traffic samples, 396743 attack traffic samples) and a test set (63458 normal packet samples, 185366 attack traffic samples). To make the data more realistic, the test portion of the dataset contains 14 additional attack types that are not in the train portion; thus, a good model should generalize well and detect attacks unseen at during training.


#### ECG5000
The  [ECG5000](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000) dataset contains examples of ECG signals from a patient. Each data sample, which corresponds to an extracted heartbeat containing 140 points, has been labeled as normal or being indicative of heart conditions related to congestive heart failure. Given an ECG signal sample, the task is to predict if it is normal or abnormal. ECG5000 is well suited to a prototype for a few reasons — it is visual (signals can be visualized easily) and it is based on real data associated with a concrete use case (heart disease detection). While the task itself is not extremely complex, the data is multidimensional (140 values per sample, which allows us to demonstrate the value of a deep model), but small enough to rapidly train and run inference. 
 

### Benchmarking Models
We sought to compare each of the models discussed earlier using the KDD dataset. We preprocessed the data to keep only 18 continuous features (Note: this slightly simplifies the problem and results on differences from similar benchmarks on the same dataset). Feature scaling (0-1 minmax scaling) is also applied to the data; scaling parameters are learned from training data and then applied to test data. We then trained each model using normal samples (97,278 samples) and evaluated it on a random subset of the test data (8000 normal samples and 2000 normal samples). 

<!-- a randomly selected subset of the test data (8,000 normal samples, and 2,000 abnormal samples).  -->
| Method | Encoder | Decoder | Other Parameters |
|----------------------------|------------------------------------------------------------------------|---------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| PCA | NA | NA | 2 Component PCA |
| OCSVM | NA | NA | Kernel: Rbf, Outlier fraction: 0.01; gamma: 0.5. <br/> Anomaly score as distance from decision boundary. |
| Autoencoder | 2 hidden layers [15, 7] | 2 hidden layers [15, 7] | Latent dimension: 2Batch size: 256Loss: Mean squared error |
| Variational Autoencoder | 2 hidden layers [15, 7] | 2 hidden layers [15, 7] | Latent dimension: 2 <br/>Batch size: 256 <br/> Loss: Mean squared error + KL divergence |
| Sequence to Sequence Model | 1 hidden layer, [10] | 1 hidden layer [20] | Bidirectional LSTMs <br/> Batch size: 256 <br/> Loss: Mean squared error |
| Bidirectional GAN | Encoder: 2 hidden layers [15, 7], <br/> Generator: 2 hidden layers [15, 7] | Generator: 2 hidden layers [15, 7] Discriminator: 2 hidden layers [15, 7] | Latent dimension: 32 <br/>Loss: Binary Cross Entropy <br/>Learning rate: 0.1 |

<br/>

We implemented each model using comparable parameters (see table above) that allow us to benchmark them in terms of training and inference (total training time to best accuracy, inference time) , storage (size of weights, number of parameters), and performance (accuracy, precision, recall). The deep learning models (AE, VAE, Seq2seq, BiGAN) were implemented in Tensorflow (keras api); each model was trained till best accuracy measured on the same validation dataset, using the [Adam optimizer](https://keras.io/optimizers/), batch size of 256 and a learning rate of  0.01.  OCSVM was implemented using the [Sklearn OCSVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html) library using the non-linear _rbf_ kernel and parameters (_nu_=0.01 and _gamma_=0.5). Results from PCA (using the the sum of the projected distance of a sample on all eigenvectors as the anomaly score) are also included.  Additional details on the parameters for each model are summarized in the table below for reproducibility. These experiments were run on an Intel(R) Xeon(R) CPU @ 2.30GHz and an NVIDIA T4 GPU (applicable to the deep models).


#### Training, Inference, Storage
| Method 	| Model Size (KB) 	| Inference Time (Seconds) 	| # of Parameters 	| Total Training Time (Seconds) 	|
|---------	|-----------------	|--------------------------	|-----------------	|-------------------------------	|
| bigan 	| 47.945 	| 1.26 	| 714 	| 111.726 	|
| ae 	| 22.008 	| 0.279 	| 842 	| 32.751 	|
| ocsvm 	| 10.77 	| 0.029 	| NA 	| 0.417 	|
| vae 	| 23.797 	| 0.391 	| 858 	| 27.922 	|
| seq2seq 	| 33.109 	| 400.131 	| 2741 	| 645.448 	|
| pca 	| 1.233 	| 0.003 	| NA	| 0.213 	|

<br/>

![Comparison of anomaly detection models in terms of model size, inference time, paramters and training time.](figures/storagemetrics.png)


Each model is compared in terms of inference time on the entire test dataset, total training time to peak accuracy, number of parameters (deep models) and model size.
As expected, a linear model like PCA is both fast to train and fast for inference. This is followed by OCSVM,autoencoders, variational autoencoders, BiGAN, and sequence-to-sequence models in order of increasing model complexity. The GAN based model required the most training epochs to achieve stable results; this is in part due to a known [stability issue](https://arxiv.org/abs/1606.03498) associated with GANs.  The sequence-to-sequence model is particularly slow for inference given the sequential nature of the decoder.

For each of the deep models, we store the network weights to disk and compute the size of weights. For OCSVM and PCA, we serialize the model to disk using [pickle](https://docs.python.org/3/library/pickle.html) and compute the size of each  model file. These values are are helpful in estimating memory and storage costs when deploying these models in production.

#### Performance

| Method 	| ROC AUC 	| Accuracy 	| Precision 	| Recall 	| f1 	| f2 	|
|---------	|---------	|----------	|-----------	|--------	|-------	|-------	|
| BiGAN 	| 0.972 	| 0.962 	| 0.857 	| 0.973 	| 0.911 	| 0.947 	|
| Autoencoder 	| 0.963 	| 0.964 	| 0.867 	| 0.968 	| 0.914 	| 0.945 	|
| OCSVM 	| 0.957 	| 0.949 	| 0.906 	| 0.83 	| 0.866 	| 0.844 	|
| VAE 	| 0.946 	| 0.93 	| 0.819 	| 0.836 	| 0.827 	| 0.832 	|
| Seq2Seq 	| 0.919 	| 0.829 	| 0.68 	| 0.271 	| 0.388 	| 0.308 	|
| PCA 	| 0.771 	| 0.936 	| 0.977 	| 0.699 	| 0.815 	| 0.741 	|

<br/>

![Histogram for the distribution of anomaly scores assigned to the test data for each model. The red vertical line represents a threshold value that yields the best accuracy.](figures/anomalyscores.png)

For each model, we use labeled test data to first select a threshold that yields the best accuracy and then report on metrics such as f1, f2, precision and recall at that threshold. We also report on ROC (area under the curve) to evaluate the overall skill of each model. Given that the dataset we use is not extremely complex (18 features), we see that most models perform relatively well.  Deep models (BiGAN, AE) are more robust (precision, recall, ROC AUC), compared to PCA and OCSVM. The  Sequence-to-sequence model is not particularly competitive, given the data is not temporal. On a more complex dataset such as images, we expect to see (similar to [existing research](https://arxiv.org/abs/1605.07717)), more pronounced advantages in using a deep learning model.


###  Web Application Prototypes
We built two prototypes that demonstrate results and insights from our experiments. The first prototype -- is built on on the KDD dataset used in the experiments above and is a visualization of the performance of four approaches to anomaly detection. The second prototype is an interactive explainer that focuses on the autoencoder model and results from applying it to detecting anomalies in ECG data.

#### Blip

![The Blip prototype.](figures/blip-1.png)

[Blip](https://blip.fastforwardlabs.com) plays back and visualizes the performance of four different algorithms on a subset of the KDD network intrusion dataset. Blip dramatizes the analogy detection process and builds user intution about the trade-offs involved.

![The terminal section shows the data streaming in.](figures/blip-2.png)

The concept of an anomaly is easy to visualize: something that doesn't look the same. The conceptual simplicity of it actually makes the prototype's job tricker. If we show you a dataset where the anomalies are easy to spot, it's not clear what you need an algorithm for. Instead, we want to place you in a situation where the data is complicated enough, and streaming in fast enough, that the benefits of an algorithm are clear. Often in a data visualization, you want to remove complexity; in Blip, we wanted to preserve it, but place it in context. We did this by including at the top a terminal-like view of the connection data coming in. The speed and number of features involved make the usefulness of an algorithm, which can operate at a higher speed and scale than a human, clear.

![The strategy section shows the performance of the algorithms across different metrics.](figures/blip-3.png)

Directly below the terminal-like streaming data, we show performance metrics for each of the algorithms. These metrics include accuracy, recall, and precision. The three different measurements hint at the trade-offs involved in choosing an anomaly detection algorithm. You will want to prioritize different metrics depending on the situation. Accuracy is a measure of how often the algorithm is right. Prioritizing precision will minimize false positives, while focusing on recall will minimize false negatives. By showing the formulas for each of these metrics, updated in real-time with the streaming data, we build intuition about how the different measures interact.

![The last section visualizes the performance of each algorithm.](figures/blip-4.png)

In the visualizations, we want to give the user a feel for how each algorithm performs across the various metrics. If a connection is classified by the algorithm as an anomaly, it is stacked on the left; if it is classified as normal, it is placed on the right. The ground truth is indicated by the color: red for anomaly, black for normal. In a perfectly performing algorithm, the left side would be completely red and the right completely black. An algorithm that has lots of false negatives (low recall) will have a higher density of red mixed in with the black on the right side. A low precision performance will show up as lots of black mixed into the left side. The fact that each connection gets its own spot makes the scale of the dataset clear (versus the streaming-terminal view where old connections quickly leave the screen). Our ability to quickly assess visual density makes it easier to get a feel for what differences in performance metrics across algorithms really mean.

One difficulty in designing the prototype was figuring out when to reveal the ground truth (visualized as the red or black color). In a real-world situation, you would not know the truth as it came in (you woudn't need an algorithm then). Early versions of the prototype experimented with only revealing the color after classification. Ultimately, we decided that because there is already a lot happening in the prototype, a delayed reveal pushed the complexity a step too far. Part of this is because the prototype shows the performance of four different algorithms. If we were showing only one, we'd have more room to animate the truth reveal. We decided to reveal the color truth at the start to strengthen the visual connection between the connection data as shown in the terminal and in each of the algorithm visualizations.

#### Prototype II - Anomagram

This section describes Anomagram - an interactive web based experience where the user can build, train and evaluate an autoencoder to detect anomalous ECG signals. It utilizes the ECG5000 dataset mentioned above.



##### UX Goals for Anomagram

Anomagram is designed as part of a growing area interactive visualizations (see [Neural Network Playground](https://playground.tensorflow.org/), [ConvNet Playground](http://convnetplayground.fastforwardlabs.com/), [GANLab](https://poloclub.github.io/ganlab/), [GAN dissection](https://gandissect.csail.mit.edu/), etc) that help communicate technical insights on how deep learning models work. It is entirely browser based and  implemented in Tensorflow.js. This way, users can explore live experiments with no installations required. Importantly, Anomagram moves beyond the user of toy/synthetic data and situates learning within the context of a concrete task (anomaly detection for ECG data). The overall user experience goals for Anomagram are summarized as follows. 

**Goal 1**: Provide an introduction to Autoencoders and how they can be applied to the task of anomaly detection. This is achieved via the _Introduction_ module (see screenshot below). This entails providing definitions of concepts (reconstruction error, thresholds etc) paired with interactive visualizations that demonstrate concepts (e.g. an interactive visualization for inference on test data, a visualization of the structure of an autoencoder, a visualization of error histograms as training progresses, etc).  

![_Introduction_ module view.](figures/anomagram-1.png)


**Goal 2**: Provide an interactive, accessible experience that supports technical learning by doing. This is mostly accomplished within the _Train a Model_ module (see screenshot below)  and is designed for users interested in additional technical depth. It entails providing a direct manipulation interface that allows the user to specify a model (add/remove layers and units within layers), modify model parameters (training steps, batchsize, learning rate, regularizer, optimizer, etc), modify training/test data parameters (data size, data composition), train the model, and evaluate model performance (visualization of accuracy, precision, recall, false positive, false negative, ROC etc metrics) as each parameter is changed. Who should use Anomagram? Anyone interested in an accessible way to learn about autoencoders and anomaly detection . Useful for educators (tool to support guided discussion of the topic), entry level data scientists, and non-ML experts (citizen data scientists, software developers, designers etc).

![_Train a Model_ module view.](figures/anomagram-2.png)

#####  Interface Affordances and Insights 
This section discusses some explorations the user can perform with Anomagram, and some corresponding insights. 

**Craft (Adversarial) Input**: Anomalies by definition can take many different and previously unseen forms. This makes the assessment of anomaly detection models more challenging. Ideally, we want the user to conduct their own evaluations of a trained model e.g. by allowing them to upload their own ECG data. In practice, this requires the collection of digitized ECG data with similar preprocessing (heartbeat extraction) and range as the ECG5000 dataset used in training the model. This is challenging. The next best way to allow testing on examples contributed by the user is to provide a simulator — hence the draw your ECG data feature. This provides a (html) canvas on which the user can draw signals and observe the model’s behaviour. Drawing strokes are converted to an array, with interpolation for incomplete drawings (total array size=140) and fed to the model. While this approach has limited realism (users may not have sufficient domain expertise to draw meaningful signals), it provides an opportunity to craft various types of (adversarial) samples and observe the model’s performance. 

_Insights_: The model tends to expect reconstructions that are close to the mean of normal data samples. Using the Draw your ecg data feature, the user can draw (adversarial) examples of input data and observe model predictions/performance.

<!-- ![Using the Draw your ecg data feature, the user can draw (adversarial) examples of input data and observe model predictions/performance.](figures/anomagram-3.png) -->
 


**Visually Compose a Model**: Users can intuitively specify an autoencoder architecture using a direct manipulation model composer. They can add layers and add units to layers using clicks. This architecture is then used to specify the model’s parameters each time the model is compiled. This follows a similar approach used in “A Neural Network Playground”[3]. The model composer connector lines are implemented using the leaderline library. Relevant lines are redrawn or added as layers are added or removed from the model. 

_Insights_: There is no marked difference between a smaller model (1 layer) and a larger model (e.g. 8 layers) for the current task. This is likely because the task is not especially complex (a visualization of PCA points for the ECG dataset suggests it is linearly separable). Users can visually compose the autoencoder model — add remove layers in the encoder and decoder. To keep the encoder and decoder symmetrical, add/remove operations on either is mirrored. 


**Effect of Learning Rate, Batchsize, Optimizer, Regularization**: The user can select from 6 optimizers (Adam, Adamax, Adadelta, Rmsprop, Momentum, Sgd), various learning rates, and regularizers (l1, l2, l1l2). 

_Insights_: Adam reaches peak accuracy with less steps compared to other optimizers. Training time increases with no benefit to accuracy as batchsize is reduced (when using Adam). A two layer model will quickly overfit on the data; adding regularization helps address this to some extent. Try them out! 

**Effect of Threshold Choices on Precision/Recall**: Earlier in this report (see [Chapter 2. Evaluating Models: Accuracy Is Not Enough ](#evaluating-models%3A-accuracy-is-not-enough)) we highlight the importance of metrics such as precision and recall and why accuracy is not enough. To support this discussion, the user can visualize how threshold choices impact each of these metrics. 

_Insights_: As threshold changes, accuracy can stay the same but, precision and recall can vary. This further illustrates how the threshold can be used by an analyst as a lever to reflect their precision/recall preferences. 



**Effect of Data Composition**: We may not always have labelled normal data to train a model. However, given the rarity of anomalies (and domain expertise), we can assume that unlabelled data is mostly comprised of normal samples. However, this assumption raises an important question - does model performance degrade with changes in the percentage of abnormal samples in the dataset? In the train a model section, you can specify the percentage of abnormal samples to include when training the autoencoder model. 

_Insights_: We see that with 0% abnormal data, the model AUC is ~96%. Great! At 30% abnormal sample composition, AUC drops to ~93%. At 50% abnormal data points, there is just not enough information in the data that allows the model to learn a pattern of normal behaviour. It essentially learns to reconstruct normal and abnormal data well and mse is no longer a good measure of anomaly. At this point, model performance is only slightly above random chance (AUC of 56%).
