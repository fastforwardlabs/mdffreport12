## Ethics

Machine learning models that learn to solve tasks independently from data are
susceptible to biases and other issues that may raise ethical concerns. Anomaly
detection models are associated with specific risks and mitigation tactics. In anomaly
detection, when the definition of “normal” is independently learned and applied
without controls, it may inadvertently reflect societal biases that can lead to
harm. This presents risks to human welfare in scenarios where anomaly detection
applications intersect with human behavior. It should go without saying that in
relation to humans, one must resist the assumption that different is bad.

![Different is _not_ necessarily bad.](figures/ill-21.png)

### Diversity Matters

Although anomalies are often scrutinized, their absence in the data may be even
more problematic. Consider a model that is trained on X-ray images of normal luggage
contents, but where the training data includes only images of luggage packed by citizens of North America.
This could lead to unusually high stop and search rates for users from other
parts of the world, where the items packed might differ greatly.

To limit the potential harm of a machine learning model’s tendency to assume
that “different is bad,” one can use a larger (or more varied) dataset and
have a human review the model's output (both measures reduce the likelihood of
errors). In the luggage example, this translates to using more varied X-ray image
data to expand the machine’s view of what is normal, and using human review to
ensure that positive predictions aren’t false positives.

### Explainability

In many anomaly detection applications, the system presents anomalous instances
to the end user, who provides a verdict (label) that can then
be fed back to the model to refine it further. Unfortunately, some applications
may not provide enough explanation about why an instance was considered
anomalous, leaving the end user with no particular guidance on where to begin
investigation. Blindly relying on such applications may cause or, in certain
cases, exacerbate bias.

### Additional Use Cases

Ethical considerations are important in every use of machine learning,
particularly when the use case affects humans. In this section, we consider 
a few use cases that highlight ethical concerns with regard to anomaly detection.

#### Data Privacy

Protecting individuals’ data has been a growing concern in the last few
years, culminating in recent data privacy laws like the EU's General Data Protection
Regulation (GDPR) and the California Consumer Privacy Act (CCPA). These
laws don’t just penalize data breaches, but also guide and limit
how an individual’s personal data can be used and processed. Thus, when anomaly
detection methods are used on protected data, privacy is a top concern.

To give an example, in [Chapter 3. Deep Learning for Anomaly Detection](#deep-learning-for-anomaly-detection) we
discussed the autoencoder, a type of neural network that has been widely used for
anomaly detection. As we saw, autoencoders have two parts: an encoder network that reduces the dimensions
of the input data, and a decoder network that aims to reconstruct the input. Their
learning goal is to minimize the reconstruction error, which is
consequently the loss function. Because the dimensionality reduction brings
information loss, and the learning goal encourages preservation of the information
that is common to most training samples, anomalies that contain rare information 
can be identified by measuring model loss.

In certain use cases, these identified anomalies could correspond to
individuals. Improper disclosure of such data can have adverse consequences for
a data subject’s privacy, or even lead to civil liability or bodily
harm. One way to minimize these effects is to use a technique called 
_differential privacy_^[Cynthia Dwork  (2006), "Differential Privacy", _Proceedings of the 33rd International Conference on Automata, Languages and Programming_ Part II: 1-12, [https://doi.org/10.1007/11787006_1](https://doi.org/10.1007/11787006_1).] on the data before it is fed into an anomaly detection system. This technique essentially adds a small amount of noise to the data, in
order to mask individual identities while maintaining the accuracy of aggregated
statistics. When coupled with an anomaly detection system, differential privacy
has been shown to reduce the rate of false positives,^[Min Du et al.  (2019), "Robust Anomaly Detection and
Backdoor Attack Detection Via Differential Privacy", [arXiv:1911.07116](https://arxiv.org/abs/1911.07116).] thus protecting the privacy of
more individuals who might otherwise have been singled out and scrutinized. 

#### Health Care Diagnostics

Anomaly detection can be applied in health care scenarios to provide quick and
early warnings for medical conditions. For example, a model can surface chest
X-rays that substantially differ from normal chest X-rays, or highlight images
of tissue samples that contain abnormalities. These analyses can have
tremendous consequences for the patient: in addition to the positive outcomes, 
a false negative might mean an undetected disease, and a false positive could 
lead to unnecessary (and potentially painful or even harmful) treatment. 

For other machine learning tasks, such as churn prediction or resumé review, analysts
strive to remove racial or socioeconomic factors from the equation. But in health
care diagnostics, it may be both appropriate and advantageous to consider them.  

To the extent that an anomaly to be detected is connected with a certain group
of people, models can be tailored to that group. For example, sickle-cell
disease is more prevalent in parts of Africa and Asia than in Europe and the
Americas. A diagnostic system for detecting this disease should include enough
samples of Asian and African patients and acknowledgment of their ethnicity to
ensure the disease is identified.  

In any event, it is important to ensure that these systems remain curated (i.e.,
that a medical professional verifies the results) and that the models are correctly
evaluated (precision, recall) before being put into production.

#### Security

Similar to the luggage example mentioned at the beginning of the chapter, home or business
security systems trained to identify anomalies present a problem of the
“different is bad” variety. These systems need to have enough data; in terms of
quantity and variability; to prevent bias that would make them more likely to
identify people of different races, body types, or socioeconomic status as anomalies based
on, for example, their skin color, size, or clothing.  

#### Content Moderation

Blind reliance on anomaly detection systems for content moderation can lead to
false positives that limit or silence system users based on the language or
types of devices they use. Content moderation software should be monitored for
patterns of inappropriate blocking or reporting, and should have a user
appeal/review process. 

#### Financial Services

Determining what harmful anomalies are in financial services applications is complex. On 
the one hand, fraudsters are actively trying to steal and launder money, and move it
around the world. On the other hand, the vast majority of transactions are legitimate
services for valuable customers. Fraudsters emulate normal 
business, making anomalies especially difficult to identify. As a result,
financial services organizations should consider first whether anomaly detection
is desirable in their use cases, and then consider the potential ethical and
practical risks.

For example, banks use customer data to offer mortgages and student loans. A lack
of diverse data could lead to unfavorable results for certain demographic
groups. Such biased algorithms can result in costly mistakes, reduce customer 
satisfaction, and damage a brand’s reputation. To combat this, one should pose
questions (like the following) that can help check for bias in the data or
model:
- Fact check: do the detected anomalies mostly belong to underrepresented groups?
- Are there enough features that explain minority groups?
- Has model performance been evaluated for each subgroup within the data?

### Innovation

As we have seen, the term _anomaly_ can carry negative connotations. 
Anomaly detection by its very nature involves identifying samples that are different from the bulk of the
other samples - but, as discussed here, assuming that different is bad may not
be fair in many cases. Instead of associating faulty
interpretations with anomalies, it may be helpful to investigate them to reveal new truths. 

After all, progress in science is often triggered by anomalous activities that
lead to innovation!

