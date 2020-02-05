## Conclusion

Anomaly detection is a classic problem, common to many business domains. In this
report, we have explored how a set of deep learning approaches can be applied in
a semi-supervised setting for addressing the anomaly detection task. We think
this focus is useful for the following reasons:
- While deep learning has demonstrated superior performance for many business
tasks, there is fairly limited information on how anomaly detection can be cast
as a deep learning problem and how deep models perform.
- A semi-supervised approach is desirable in handling unseen, unknown anomalies
and does not incur vast data labeling costs for businesses.
- Traditional machine learning approaches are suboptimal for handling high
dimensional, complex data and modeling interactions between each variable.

In our <<prototype>>, we show how deep learning models can achieve competitive
performance on a multivariate tabular dataset (network intrusion detection).
This is in concurrence with results from existing research that show superior
performance of deep learning models for high dimensional data such as images
(cite).

Overall, while deep learning approaches are useful, there are a few challenges
that may limit their deployment in production settings.

- **Latency:** Compared to linear models (such as AR, ARMA, etc.) or shallow machine learning
models (such as One Class SVM), deep learning models can have significant
latency associated with inference. This makes it expensive to apply them within
streaming data (high volume, high velocity) use cases at scale. For example, our
experiments show that inference with OCSVM is 12x faster than with an
autoencoder. 

- **Data Requirements:** Deep learning models typically require a large dataset (tens of thousands of
samples) for effective training. Further, deep models are prone to overfitting
and need to be carefully evaluated to address this. Many anomaly use cases can
frequently have few data points (e.g., daily sales data for two years will
generate 712 samples, which may be insufficient to train a model). In such
scenarios, linear models designed to work with smaller datasets are a better
option.

- **Managing Distribution Shift:** In many scenarios, the underlying process generating data may legitimately
change such that a datapoint that was previously anomalous becomes normal. The
changes could be gradual, cyclical, or even abrupt in nature. This phenomenon,
although not unique to anomaly detection, is known as concept drift. With
respect to anomaly detection, one way to handle this is to frequently retrain
the model as new data arrives - or trigger retraining when concept drift is
detected. It can be challenging to maintain this continuous learning approach
for deep models, as training time can be significant.

Going forward, we expect the general approaches discussed in the report to
continually evolve and mature. As examples, see recent extensions to the
encoder-decoder model approach that are based on GMMs^[[ Deep Autoencoding
Gaussian Mixture Model for Unsupervised Anomaly
Detection](https://openreview.net/forum?id=BJJLHbb0-)], LSTMs^[[Sequential
VAE-LSTM for Anomaly Detection on Time Series](https://arxiv.org/abs/1910.03818)], 
CNNs^[[Time-Series Anomaly Detection Service at
Microsoft](https://arxiv.org/abs/1906.03821)], and GANs^[[MAD-GAN: Multivariate
Anomaly Detection for Time Series Data with Generative Adversarial
Networks](https://arxiv.org/abs/1901.04997)].
Like everything else in machine learning, there is no “one size fits all”; no
one model works best for every problem. The right approach always depends on the
use case and data. 
