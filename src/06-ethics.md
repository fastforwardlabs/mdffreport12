## Ethics

Machine learning models that learn to solve tasks independently from data are
susceptible to biases and other issues that may raise ethical concerns. Anomaly
detection models have specific risks and mitigation tactics. In anomaly
detection, when the definition of “normal” is independently learned and applied
without controls, it may inadvertently reflect societal biases that can lead to
harm. This presents risks to human welfare in scenarios where anomaly detection
applications intersect with human behavior. It should go without saying that in
relation to humans, we must resist the assumption that different is bad. 

### Diversity Matters

Although anomalies are often scrutinized, their absence in the data may be even
worse! Consider a model that is trained on x-ray images of normal luggage
content that only uses images of luggage packed by citizens of North America.
This could lead to unusually high stop and search rates for users from other
parts of the world, where items packed might differ greatly.

To limit the potential harm of a machine learning model’s tendency to assume
that “different is bad,” we can use a larger (or more varied) dataset, and
provide human supervision of the model (both measures reduce the likelihood of
errors). In the above example, this translates to using more varied x-ray image
data to expand the machine’s view of what is normal, and using human review to
ensure that positive predictions aren’t false positives.

### Explainability

In many anomaly detection applications, the system presents anomalous instances
to the end user - a decision maker to provide a verdict (label) which can then
be fed back to the model to refine it further. Unfortunately, some applications
may not provide enough explanation about why an instance was considered
anomalous, leaving the end user with no particular guidance on where to begin
investigation. Blindly relying on such applications may cause or in certain
cases exacerbate bias.

### Additional Use Cases

Ethical considerations are important in every use of machine learning,
particularly when the use case affects humans. Below are a few use cases in
which this is particularly true of anomaly detection.

#### Data Privacy

Protecting an individual’s data has been a growing concern in the last few
years, culminating in recent data privacy laws like the GDPR and CCPA. These
laws don’t just consider and penalize data breaches, but also guide and limit
how an individual’s personal data can be used and processed. Thus, when anomaly
detection methods are used on protected data, privacy is paramount.

To give an example, in <<Chapter 3. Deep Learning for Anomaly Detection>> we
discussed autoencoders, a type of neural network that has been widely used for
anomaly detection. It contains an encoder network which reduces the dimension of
the input data, and a decoder network which aims to reconstruct the input. The
learning goal of autoencoders is to minimize the reconstruction error, which is
consequently the loss function. Because the dimensionality reduction brings
information loss, and the learning goal encourages to preserve the information
that is common to most training samples, anomalies that contain rare information
could be identified by measuring model loss. 

In certain use cases, these identified anomalies could correspond to
individuals. Improper disclosure of such data can have adverse consequences for
a data subject’s private information, or even lead to civil liability or bodily
harm. One way to minimize these effects is to use a technique called
differential privacy^[Differential Privacy, Dwork 2006] on the data before it is fed into an anomaly detection
system. This technique essentially adds a small amount of noise to the data, in
order to mask individual identities while maintaining the accuracy of aggregated
statistics. When coupled with an anomaly detection system, differential privacy
has been shown to reduce the false positives^[[Robust Anomaly Detection and
Backdoor Attack Detection Via Differential
Privacy](https://arxiv.org/abs/1911.07116)], thus protecting the privacy of
more individuals who would otherwise have been scrutinized by being singled out. 

#### Health Care Diagnostics

Anomaly detection can be applied in health care scenarios to provide quick and
early warnings for medical conditions. For example, a model can surface chest
x-rays that substantially differ from normal chest x-rays, or highlight images
of tissue samples as containing abnormalities. These analyses can have
tremendous consequences for the patient. A false negative means an undetected
disease, and a false positive means unnecessary - and potentially painful or
even harmful - treatment. 

For other machine learning tasks (e.g., churn prediction, resume review), we
strive to remove racial or socioeconomic factors from the equation. In health
care diagnostics, it may be both appropriate and advantageous to focus on them.

To the extent that an anomaly to be detected is connected with a certain group
of people, models can be tailored to that group. For example, sickle-cell
disease is more prevalent in parts of Africa and Asia than in Europe and the
Americas. A diagnostic system for detecting this disease should include enough
samples of Asian and African patients and acknowledgement of their ethnicity to
make sure the disease is identified.

In any event, it is important to ensure that these systems remain curated (i.e.,
a medical professional verifies results) and that the models are correctly
evaluated (precision, recall) before being put into production.

#### Security

Similar to the luggage example we’ve already referenced, home or business
security systems trained to identify anomalies also present a problem of the
“different is bad” variety. These systems need to have enough data - in terms of
quantity and variability -  to prevent bias that would make them more likely to
identify people of different races or socioeconomic status as anomalies based
on, for example, their color or clothing.

#### Content Moderation

Blind reliance on anomaly detection systems for content moderation can lead to
false positives that limit or silence system users based on the language or
types of devices they use. Content moderation software should be monitored for
patterns of inappropriate blocking or reporting, and should have a user
appeal/review process. 

#### Financial Services

Determining what harmful anomalies are in financial services is complex. On one
hand, fraudsters are actively trying to steal and launder money, and move it
around the world. On the other hand, nearly all transactions are legitimate
services for valuable customers. Fraudsters affirmatively emulate normal
business, making anomalies especially difficult to identify. As a result,
financial services organizations should consider first whether anomaly detection
is desirable in their use cases, and then consider the potential ethical and
practical risks.

For example, banks use customer data to offer mortgages or student loans. A lack
of diverse data could lead to unfavorable results for certain demographic
groups. These biased algorithms can result in costly mistakes, reduce customer
satisfaction, and damage a brand’s reputation. To combat this, one should pose
questions (like the following) that can help check for bias in the data or
model:
- Fact check: do the detected anomalies mostly belong to underrepresented groups?
- Are there enough features that explain minority groups?
- Has model performance been evaluated for each subgroup within your data?

### Innovation

As we have seen, the term anomaly can carry negative connotations. The nature of
anomaly detection is identifying samples that are different from the bulk of the
other samples - but as discussed, assuming that “different is bad” is not
necessarily fair in many cases. Perhaps instead of associating faulty
interpretations with anomalies, pursuing them to reveal new truths could be
helpful. 

After all, progress in science is often triggered by anomalous activities that
lead to innovation!

