## Landscape

In this section, we review the landscape of open source tools, service vendor
offerings, their trade-offs, and when to use each. 

### Open Source Tools and Frameworks

Popular open source machine learning libraries or packages in Python and R
include implementations of algorithmic techniques that can be applied to anomaly
detection tasks.  Algorithms (e.g., clustering, OC-SVM, isolation forests) exist
as part of a general-purpose framework like scikit-learn and do not cater
specifically to anomaly detection. In addition, generic packages for univariate
time series forecasting (e.g., Facebook’s
[Prophet](https://facebook.github.io/prophet/)) have been applied widely to
anomaly detection tasks where anomalies are identified based on the difference
between the true value and a forecast.

In this section of the report, our focus is on comprehensive toolboxes that
specifically address the task of anomaly detection.

#### Python Outlier Detection (PyOD)
[PyOD](https://github.com/yzhao062/pyod) is an open-source Python toolbox for performing scalable outlier detection
on multivariate data. Uniquely, it provides access to a wide range of outlier
detection algorithms, including established outlier ensembles and more recent
neural network-based approaches, under a single, well-documented API. 

Distinct advantages:
- It contains more than 20 algorithms which cover both classical techniques such
as local outlier factor and recent neural network architectures such as
autoencoders or adversarial models.
- PyOD implements combination methods for merging the results of multiple
detectors and outlier ensembles which are an emerging set of models. 
- It includes a unified API, detailed documentation, and interactive examples
across all algorithms for clarity and ease of use.
- All models are covered by unit testing with cross-platform continuous
integration, code coverage, and code maintainability checks.
- Optimization instruments are employed when possible: just-in-time (JIT)
compilation and parallelization are enabled in select models for scalable
outlier detection. Lastly, PyOD is compatible with both Python 2 and 3 across
major operating systems.

#### Seldon’s Anomaly Detection Package

[Seldon.io](https://github.com/SeldonIO) is known for its open source machine learning deployment solution for
Kubernetes, which can in principle be used to serve arbitrary models. In
addition, the Seldon team has recently released
[alibi-detect](https://github.com/SeldonIO/alibi-detect) - a Python package
focused on outlier, adversarial, and concept drift detection. The package aims
to cover both online and offline detectors for tabular data, text, images and
time series. The outlier detection methods should allow the user to identify
global, contextual and collective outliers.

They have identified anomaly detection as a sufficiently important capability to
warrant a dedicated attention in the framework, and have implemented several
models to use “out-the-box.” The existing models include seq2seq LSTMs,
variational auto-encoders, spectral residual for time series, gaussian mixture
models, isolation forests, Mahalanobis distance, and others. Note that they also
provide examples and documentation on how to use along with their platform.

In the Seldon Core architecture, anomaly detection methods may be implemented as
either a model or an input transformer. In the latter case, they can be composed
with other data transformations to process inputs to another model. This nicely
illustrates one role anomaly detection can play in machine learning systems:
flagging bad inputs before they pass through the rest of a pipeline.

#### R packages

The following section reviews R packages that have been created for anomaly
detection. Interestingly, most of them deal with time series data.

##### Twitter’s AnomalyDetection package

Twitter’s [AnomalyDetection](https://github.com/twitter/AnomalyDetection) is an open-source R package to automatically detect
anomalies. It is applicable across a wide variety of contexts (for example,
detecting anomalies in system metrics after a new software release, user
engagement after an A/B test, or problems in econometrics, financial
engineering, political and social sciences). It can help detect global/local
anomalies as well as positive/negative (i.e., a point-in-time increase/decrease
in values) anomalies.

The primary algorithm, Seasonal Hybrid ESD (S-H-ESD), builds upon the
Generalized ESD test for detecting anomalies - which can be global as well as
local. This is achieved by employing time series decomposition and using robust
statistical metrics, i.e., median together with ESD. In addition, for a long
time series (say, 6 months of minutely data), the algorithm employs piecewise
approximation. 

Besides time series, the package can also be used to detect anomalies in a
vector of numerical values when the corresponding timestamps are not available.
The package provides rich visualization support. The user can specify the
direction of anomalies, the window of interest (such as last day, last hour), as
well as enable/disable piecewise approximation, and the x- and y-axis are
annotated to assist visual data analysis.

##### Anomalize package

The [anomalize](https://github.com/business-science/anomalize) package, open sourced by Business Science, does time series
anomaly detection that goes inline with other [Tidyverse
packages](https://www.tidyverse.org/) (or packages
supporting tidy data). 

Anomalize has three main functions:
- Decompose: separates out the time series into seasonal, trend, and remainder
components
- Anomalize: applies anomaly detection methods to the remainder component
- Recompose: calculates upper and lower limits that separate the “normal” data
from the anomalies

##### Tsoutliers package

This [package](https://www.rdocumentation.org/packages/tsoutliers/versions/0.6-8) 
implements a procedure based on the approach described in [Chen and
Liu
(1993)](https://www.researchgate.net/publication/243768707_Joint_Estimation_of_Model_Parameters_and_Outlier_Effects_in_Time_Series) 
for automatic detection of outliers in time series. Time series data
often undergoes non-systematic changes that alter the dynamics of the data
transitory or permanently. The approach considers innovational outliers,
additive outliers, level shifts, temporary changes, and seasonal level shifts -
while fitting a time series model.

#### Numenta’s HTM (Hierarchical Temporal Memory)

Research organization [Numenta](https://numenta.com/) has introduced hierarchical temporal memory (HTM)
- a machine learning model for anomaly detection. At the core of HTM are
  time-based learning algorithms that store and recall temporal patterns. Unlike
most other machine learning methods, HTM algorithms learn time-based patterns in
unlabeled data on a continuous basis. They are robust to noise, and high
capacity - meaning they can learn multiple patterns simultaneously. The HTM
algorithms are documented and available through its open source project, NuPIC
(Numenta Platform for Intelligent Computing). The HTM technology is suited to
address a number of problems, particularly those with the following
characteristics: streaming data, underlying patterns in data change over time,
subtle patterns, and time-based patterns. 

One of the first commercial applications developed using NuPIC is
[Grok](https://grokstream.com/), which
performs IT analytics, giving insight into IT systems to identify unusual
behavior and reduce business downtime. Another is
[Cortical.io](https://www.cortical.io/), which enables
applications in natural language processing.

The NuPIC platform also offers several tools such as the HTM Studio and the
Numenta Anomaly Benchmark (NAB). HTM Studio is a desktop tool that finds
anomalies in time series without the need to program, code, or set parameters.
NAB is a novel benchmark for evaluating and comparing algorithms for anomaly
detection in streaming, real-time applications. It is composed of over 50
labeled real-world and artificial time series data files, plus a novel scoring
mechanism designed for real-time applications.

Besides this, there are example applications available on NuPIC that include
sample code and white papers for: tracking anomalies in the stock market, ogue
behavior detection - finding anomalies in human behavior, geospatial tracking -
finding anomalies in objects moving through space and time 

Numenta is a technology provider and does not create go-to-market solutions for
specific use cases. The company licenses their technology and application code
to developers, organizations, and companies who wish to build upon their
technology. Numenta has several different types of licenses, including open
source, trial, and commercial licenses. Developers can use Numenta technology
within NuPIC using the AGPL v3 open source license. Their HTM Studio is a free,
desktop tool allows you to test whether our Hierarchical Temporal Memory (HTM)
algorithms will find anomalies in your data without having to program, code or
set parameters. 

### Anomaly Detection as a Service

In this section, we survey a sample of anomaly detection service providers. Most
of these providers can access data from public cloud databases, provide some
kind of dashboard/report format to view/analyze data, have an alert mechanism
when an anomaly occurs, and view underlying causes. An anomaly detection
solution should try to reduce the time to detect and speed up the time to
resolution by identifying KPIs and attributes that are causing the alert.

#### Anodot

*Capabilities*
[Anodot](https://www.anodot.com/) is a real-time analytics and automated anomaly detection system that
detects and 
turns outliers in time series data into business insights. They explore anomaly
detection from the perspective of forecasting, where anomalies are identified
based on deviations from expected forecasts. As part of their product, they do
two things:
- Business monitoring: the system does SAAS monitoring, finds anomalies and helps
with root cause analysis
- Business forecasting: forecasting, what if, and optimization. 

*Data requirements*
Anodot supports multiple input data sources including direct uploads, or
integrations with Amazon’s S3 or Google Cloud storage. They are data agnostic
and can track a variety of  metrics, e.g., revenue, number of sales, the number
of page visits, number of daily active users, etc. 

*Modeling approach/technique(s)* 
Anodot analyzes all the business metrics in real-time and at scale by running
its ML algorithms on the live data stream itself, without reading or writing
into a database. Every data point that flows into Anodot from all data sources
is correlated with the relevant metric’s existing normal model, and either
flagged as an anomaly or serves to update the normal model. They believe that no
single model can be used to cover all metrics. To allocate the optimal model for
each metric, they first create a library of model types for different signal
types (metrics that are stationary, non-stationary, multimodal, discrete,
irregularly sampled, sparse, stepwise, etc.). Each new metric goes through a
classification phase, and is matched with the optimal model. The model then
learns “normal behavior” for each metric, which is a prerequisite to identifying
anomalous behavior. To accommodate this kind of learning in real-time at scale,
they use sequential adaptive learning algorithms which initialize a model of
what is normal on the fly, and then compute the relation of each new data point
going forward.

*Point anomalies or intervals*
Instead of flagging individual data points as anomalies, Anodot highlights
intervals. Points within the entire duration of the interval are considered
anomalous, preventing redundant alerts.

*Thresholding*
While users can specify static thresholds which trigger alerts, Anodot also
provides automatic defaults where no thresholding input from the user is
required.

*Root cause investigation*
Anodot helps investigate why the alert was triggered. It tries to understand how
different active anomalies correlate (to expedite root cause investigation) and
shortens the time to resolution. Anodot bands groups together different
anomalies (or “incidents”) which tell the story of the phenomena. These can be
(a) multiple anomalous values in the same dimension (e.g., a drop in traffic
from sources A and B, but not C, D or E), or (b) correlation between different
anomalous KPIs, such as visits, conversions, orders revenue, error rate. Each
incident has an Anomap, a graphic distribution of the dimensions most impacted.
This is essentially a heat map that makes it easier to understand the whole
picture.

*User Interface*
The Anodot interface enables the user to visualize and explore alerts. With the
receipt of every alert, users are prompted to give the alert a binary score
(good catch / bad catch). This input is fed back into the learning model to
further tune it by providing real-life indications about the validity of its
performance. By training the algorithms with direct feedback on anomalies, users
can influence the system’s functionality and results.

*Delivery*
Notifications can be forwarded to every user through his/her choice of
channel(s). Anodot notification integrations include—but are not limited
to—Slack, API, email, pagerduty, Jira, Microsoft Teams OpsGenie, etc.

#### Amazon’s QuickSight

*Capabilities*
Amazon [QuickSight](https://docs.aws.amazon.com/quicksight/latest/user/anomaly-detection-function.html) 
is a cloud-native BI service that allows its users to create
dashboards and visualizations to communicate business insights. In winter 2019,
Amazon’s machine learning capability was integrated with QuickSight to provide
anomaly detection, forecasting, and auto-narrative capabilities as part of the
BI tool. It is a licensed software and pricing is usage-based; you only pay for
the active usage, regardless of the number of users. That said, the
pay-per-session could end up being expensive since anomaly detection tasks are
compute intensive

*Data requirements*
QuickSight requires you to connect or import structured data  directly query a
SQL-compatible source, or ingest the data into SPICE. There is a requirement on
the number of  historical data points varying based on the task (analyzing
anomalies or forecasting). There are also restrictions on the number of category
dimensions that could be included (for example, product category, region,
segment).

*Modeling approach/technique(s)* 
QuickSight provides a point and click solution to learn anomalous behavior and
generate forecasts. It utilizes a built-in version of the Random Cut Forest
(RCF) online algorithm, which can not only  be noisy, but lead to large amounts
of false positive alerts. On the plus side,, it provides a customizable
narrative feature that explains key takeaways from the insights generated. For
instance, it could provide a summary of how the revenue compares to a previous
period or a 30-day average, and/or highlight the event (in case of an anomaly). 

*Point anomalies or intervals*
Anomalous events are presented discretely, on a point-by-point basis. So if an
anomaly lasts more than a single time unit, the system will flag several events
which could be noisy and redundant.

*Thresholding*
Anomaly detection with QuickSight employs a thresholding approach to trigger
anomalous events. The user provides a threshold value (low, medium, high, very
high) that determines how sensitive the detector is to detected anomalies.
Expect to see more anomalies when the setting is low, and fewer anomalies when
the setting is set to high. This sensitivity is determined based on standard
deviations of the anomaly score generated by the RCF algorithm. This approach
could be very tedious - especially when there are multiple time series being
analyzed across various data hierarchy combinations - and introduces manual
intervention.

*Root cause investigation*
One can interactively explore the anomalies on the Quicksight dashboard/report
to help understand the root cause. QuickSight performs a contribution analysis
which highlights  the factors that significantly contribute to an anomaly. If
you have dimensions in your data that are not being used in the anomaly
detection, you can add up to four of them for contribution analysis. In
addition, it can support interactive "what-if" queries. In these, some of the
forecasts can be altered and treated as hypotheticals to provide conditional
forecasts.

*User interface*
Quicksight anomaly detection provides a basic reporting interface. From a UI
perspective, it is fairly unexceptional. For instance, it lacks a way to
understand the overall picture with anomalous points. (Do the anomalies have
some common contributing factors?) Furthermore, the forecasted values do not
have confidence intervals associated with them, which would help the end-user
visually understand the magnitude of the anomaly. As it stands, there is no
basis for comparison. 

*Delivery*
The QuickSight dashboards can be embedded within applications, shared among
users, and/or be sent via email - as long as the recipients have a QuickSight
subscription.

#### Outlier.ai

*Capabilities*
[Outlier.ai](https://outlier.ai/) is a licensed software that uses artificial intelligence to automate
the process of business analytics. It can connect to databases provided by cloud
services and automatically provides insights to your inbox - without the need to
create reports or write queries. Outlier works with customers across industry
segments, applying machine learning to automatically serve up business critical
insights.

*Data requirements*
Outlier.ai can connect to databases provided by cloud services.

*Modeling approach/technique(s)* 
Unknown 

*Point anomalies or intervals*
Anomalous events are presented discretely, on a point-by-point basis. 

*Thresholding*
N/A

*Root cause investigation*
Outlier.ai allows customers to not only surface key insights about business
changes automatically, but also identify the likely root causes of those
changes; this feature guides teams in making quick, informed business decisions.
Teams can easily share stories through PDF, PowerPoint optimized images, or an
auto-generated email, annotated with their comments.

*User interface*
The UI is very similar to most standard BI tools, making it fairly
user-friendly.

*Delivery*
The dashboards can be embedded within applications, shared among users, and/or
be sent via email. 

#### Vectra.ai

*Capabilities*
In simple terms, Vectra’s flagship platform, [Cognito](https://www.vectra.ai/) can be described as an
intrusion detection system. This cloud-based, network detection-and-response
system performs a number of cybersecurity related tasks including network
traffic monitoring and anomaly detection. For this latter task, metadata
collected from captured packets (rather than via deep-packet inspection) is
analyzed using a range of behavioral detection algorithms. This provides
insights about outlier characteristics that can be applied in a wide range of
cybersecurity detection-and-response use cases. Cognito works with both
encrypted and unencrypted traffic. 

*Data requirements* 
It can connect to databases provided by cloud services. Cognitio also uses
metadata drawn from Active Directory and DHCP logs. 

*Modeling approach/ technique(s)* 
According to a
[whitepaper](https://content.vectra.ai/rs/748-MCE-447/images/WhitePaper_2019_The_data_science_behind_Cognito_AI_threat_detection_models_English.pdf?mkt_tok=eyJpIjoiWkRGaVpHVmtaVGxrTkdFeiIsInQiOiJ2RVhkK3M0cHU3dXNQRDZ2YnA3QW16K0ZKVFVEK1lDeFRwcTZPMGxXZlB0clhOYmhPaVBXenkzRmY1Ylwvakp5d2FcL1dSakVKbDZhcHZtNEdZU1A3aHFMYkpxVlZHWXllXC9xUGRPOXNtZ0NyTFRjTitxUlVkaXBzNFdiQlBaUUxwVSJ9) 
published by Vectra, a mix of ML approaches are used
to deliver the cybersecurity features the platform supports, in both global and
local (network) contexts. For example, supervised learning techniques (such as
Random Forest) can help to address cyberthreats associated with suspicious HTTP
traffic patterns. Drawing on large-scale analysis of many types of malicious
traffic and content as well as domain expertise, RF can be used to identify
patterns of command-and-control behavior that don’t exist in benign HTTP
traffic. 
Vectra also uses unsupervised ML techniques such as k-means clustering to
identify valid credentials that have potentially been stolen from a compromised
host. This type of theft is the basis of cyberattacks such as pass-the-hash and
golden ticket. Elsewhere, deep learning has proven effective in detecting
suspicious domain activity; specifically, the detection of
algorithmically-generated domains that are set up by cyberattackers as the
front-end of their command-and-control infrastructure. 

*Point anomalies or intervals*
Anomalous events are presented discretely, on a point-by-point basis.

*Thresholding*
The scoring of compromised hosts by the Vectra Threat Certainty Index allows
security teams to define threshold levels based on combined scoring. 

*Root cause investigation*
All detection events are correlated to specific hosts that show signs of threat
behaviors. In turn, all context is assimilated into an up-to-the-moment score of
the overall risk to the organization.

*User interface*
Vectra.ai’s Cognito platform delivers detection information via a simple
dashboard which displays information such as: a prioritized (in terms of risk)
list of compromised hosts, changes in a host’s threat and certainty scores, and
‘key assets’ that show signs of attack.

*Delivery*
The platform supports information-sharing by security teams on demand, or on a
set schedule managed by its customizable reporting engine. Real-time
notifications about network hosts - with attack indicators that have been
identified (with the highest degree of certainty) as posing the biggest risk -
is also a supported feature.

#### Yahoo’s Anomaly Detector - Sherlock 

*Capabilities*
[Sherlock](https://github.com/yahoo/sherlock) is an open source anomaly detection 
service built on top of [Druid](http://druid.io/) (an
open-source, distributed data store). It leverages the [Java library, Extensible
Generic Anomaly Detection System (EGADS)](https://github.com/yahoo/egads) to detect anomalies in time-series
data. It’s fast and scalable. It allows users to schedule jobs on an hourly,
daily, weekly, or monthly basis (although it also supports ad hoc real-time
anomaly detection requests). Anomaly reports can be viewed from Sherlock's
interface, or received via email.

*Data requirements*
Sherlock accesses time series data via Druid JSON queries and uses a Redis
backend to store job metadata, the anomaly reports (and other information) it
generates, as well as a persistent job queue. These anomaly reports can be
accessed via direct requests using the Sherlock client API or delivered via
scheduled email alerts. 

*Modeling approach/ technique(s)* 
Sherlock takes a time-series modeling based approach to anomaly detection using
three important modules from the EGADS library - time series modelling, anomaly
detection, and alerting. The times-series modeling module supports the use of
historical data to learn trends and seasonality in the data using model classes
such as ARIMA. The resulting values are applied to the anomaly detection models
that comprise the Anomaly Detection module.  These models support a number of
detection scenarios that are relevant in a cybersecurity context (e.g., outlier
detection and change point detection). The Alerting module uses the error metric
produced from anomaly detection models and outputs candidate anomalies based on
dynamically learnt thresholds, learning to filter out irrelevant anomalies over
time. 

*Point anomalies or intervals*
Anomalous events are presented both discretely, on a point-by-point basis, and
as intervals. 

*Threshold*
Thresholds are learnt dynamically. No thresholding input from the user is
required/supported.

*Root cause analysis*
N/A

*User interface*
Sherlock’s user interface is built with [Spark Java](http://sparkjava.com/) - 
a UI framework for building
web applications. The UI enables users to submit instant anomaly analyses,
create and launch detection jobs, and view anomalies on both a heatmap and a
graph.

*Delivery*
Scheduled anomaly requests are delivered via email or directly via API-based
queries. 

