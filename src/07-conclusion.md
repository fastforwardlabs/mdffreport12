## Conclusion

Anomaly detection is a classic problem, common to many business domains. In this
report, we have explored how a set of deep learning approaches can be applied to this task in
a semi-supervised setting. We think this focus is useful for the following reasons: 
- While deep learning has demonstrated superior performance for many business
tasks, there is fairly limited information on how anomaly detection can be cast
as a deep learning problem and how deep models perform.
- A semi-supervised approach is desirable in handling previously unseen, unknown anomalies
and does not incur vast data labeling costs for businesses.
- Traditional machine learning approaches are suboptimal for handling 
high-dimensional, complex data and for modeling interactions between each variable.  
   
In our prototype, described in [Chapter 4. Prototype](#prototype), we show how deep learning models can achieve competitive
performance on a multivariate tabular dataset (for the task of network intrusion detection).
Our findings are in concurrence with results from existing research that show the superior
performance of deep learning models for high-dimensional data, such as images. 

But while deep learning approaches are useful, there are a few challenges
that may limit their deployment in production settings. Problems to consider include: 

_Latency_   
Compared to linear models (AR, ARMA, etc.) or shallow machine learning
models such as OCSVMs, deep learning models can have significant
latency associated with inference. This makes it expensive to apply them in
streaming data use-cases at scale (high volume, high velocity). For example, our
experiments show that inference with an OCSVM is 12x faster than with an
autoencoder. 

_Data requirements_   
Deep learning models typically require a large dataset (tens of thousands of
samples) for effective training. The models are also prone to overfitting
and need to be carefully evaluated to address this risk. Anomaly detection use-cases
frequently have relatively few data points; for example, daily sales data for two years will
generate 712 samples, which may be insufficient to train a model. In such
scenarios, linear models designed to work with smaller datasets are a better
option.  

_Managing distribution shift_   
In many scenarios (including, but certainly not limited to, anomaly detection),
the underlying process generating data may legitimately
change such that a data point that was previously anomalous becomes normal. The
changes could be gradual, cyclical, or even abrupt in nature. This phenomenon
is known as _concept drift_. With
respect to anomaly detection, one way to handle it is to frequently retrain
the model as new data arrives, or to trigger retraining when concept drift is
detected. It can be challenging to maintain this continuous learning approach
for deep models, as training time can be significant.  

Going forward, we expect the general approaches discussed in this report to
continue evolving and maturing. As examples, see recent extensions to the
encoder-decoder model approach that are based on Gaussian mixture models,^[Bo Zong et al., "Deep Autoencoding
Gaussian Mixture Model for Unsupervised Anomaly Detection" (2018), https://openreview.net/forum?id=BJJLHbb0-.] LSTMs,^[Run-Qing Chen et al., "Sequential
VAE-LSTM for Anomaly Detection on Time Series" (2019), [arXiv:1910.03818](https://arxiv.org/abs/1910.03818).], 
convolutional neural networks,^[Hansheng Ren et al., "Time-Series Anomaly Detection Service at
Microsoft" (2019), [arXiv:1906.03821](https://arxiv.org/abs/1906.03821).] and GANs.^[Dan Li et al., "MAD-GAN: Multivariate Anomaly Detection for Time Series Data with Generative Adversarial
Networks" (2019) [arXiv:1901.04997](https://arxiv.org/abs/1901.04997).]
As with everything else in machine learning, there is no “one size fits all”; no 
one model works best for every problem. The right approach always depends on the 
use-case and the data. 
