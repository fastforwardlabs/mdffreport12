## Landscape

This chapter provides an overview of the landscape of currently available open source tools and service vendor
offerings available for anomaly detection, and considers the trade-offs as well as when to use each.

### Open Source Tools and Frameworks

Several popular open source machine learning libraries and packages in Python and R
include implementations of algorithmic techniques that can be applied to anomaly
detection tasks. Useful algorithms (e.g., clustering, OCSVMs, isolation forests) also exist
as part of general-purpose frameworks like `scikit-learn` that do not cater
specifically to anomaly detection. In addition, generic packages for univariate
time series forecasting (e.g., Facebook’s
[Prophet](https://facebook.github.io/prophet/)) have been applied widely to
anomaly detection tasks where anomalies are identified based on the difference
between the true value and a forecast.

In this section, our focus is on comprehensive toolboxes that
specifically address the task of anomaly detection.

#### Python Outlier Detection (PyOD)
[PyOD](https://github.com/yzhao062/pyod) is an open source Python toolbox for performing scalable outlier detection
on multivariate data. It provides access to a wide range of outlier 
detection algorithms, including established outlier ensembles and more recent
neural network-based approaches, under a single, well-documented API. 

PyOD offers several distinct advantages:
- It gives access to over 20 algorithms, ranging from classical techniques such
as local outlier factor (LOF) to recent neural network architectures such as
autoencoders and adversarial models.
- It implements combination methods for merging the results of multiple
detectors and outlier ensembles, which are an emerging set of models. 
- It includes a unified API, detailed documentation, and interactive examples
across all algorithms for clarity and ease of use.
- All models are covered by unit testing with cross-platform continuous
integration, code coverage, and code maintainability checks.
- Optimization instruments are employed when possible: just-in-time (JIT)
compilation and parallelization are enabled in select models for scalable
outlier detection. 
- It's compatible with both Python 2 and 3 across major operating systems.

#### Seldon’s Anomaly Detection Package

[Seldon.io](https://github.com/SeldonIO) is known for its open source ML deployment solution for
Kubernetes, which can in principle be used to serve arbitrary models. In
addition, the Seldon team has recently released
[`alibi-detect`](https://github.com/SeldonIO/alibi-detect), a Python package
focused on outlier, adversarial, and concept drift detection. The package aims
to provide both online and offline detectors for tabular data, text, images, and
time series data. The outlier detection methods should allow the user to identify
global, contextual, and collective outliers.

Seldon has identified anomaly detection as a sufficiently important capability to
warrant dedicated attention in the framework, and has implemented several
models to use out of the box. The existing models include sequence-to-sequence LSTMs,
variational autoencoders, spectral residual models for time series, Gaussian mixture
models, isolation forests, Mahalanobis distance, and others. Examples and 
documentation are provided.

In the Seldon Core architecture, anomaly detection methods may be implemented as
either models or input transformers. In the latter case, they can be composed
with other data transformations to process inputs to another model. This nicely
illustrates one role anomaly detection can play in ML systems:
flagging bad inputs before they pass through the rest of a pipeline.

#### R Packages

The following section reviews R packages that have been created for anomaly
detection. Interestingly, most of them deal with time series data.

##### Twitter’s AnomalyDetection package

Twitter’s [AnomalyDetection](https://github.com/twitter/AnomalyDetection) is an 
open source R package that can be used to automatically detect 
anomalies. It is applicable across a wide variety of contexts (for example,
anomaly detection in system metrics after a new software release, user
engagement after an A/B test, or problems in econometrics, financial
engineering, or the political and social sciences). It can help detect both global and local
anomalies as well as positive/negative anomalies (i.e., a point-in-time increase/decrease
in values).

The primary algorithm, Seasonal Hybrid Extreme Studentized Deviate (S-H-ESD), builds upon the
Generalized ESD test for detecting anomalies, which can be either global or 
local. This is achieved by employing time series decomposition and using robust
statistical metrics; i.e., median together with ESD. In addition, for long
time series (say, 6 months of minutely data), the algorithm employs piecewise
approximation. 

The package can also be used to detect anomalies in a
vector of numerical values when the corresponding timestamps are not available,
and it provides rich visualization support. The user can specify the
direction of anomalies and the window of interest (such as the last day or last hour) 
and enable or disable piecewise approximation, and the x- and y-axes are 
annotated to assist visual data analysis.

##### anomalize package

The [`anomalize` package](https://github.com/business-science/anomalize), open sourced by Business Science, performs time series anomaly detection that goes inline with other [Tidyverse
packages](https://www.tidyverse.org/) (or packages
supporting tidy data). 

Anomalize has three main functions:
- _Decompose_ separates out the time series into seasonal, trend, and remainder
components.
- _Anomalize_ applies anomaly detection methods to the remainder component.
- _Recompose_ calculates upper and lower limits that separate the “normal” data
from the anomalies.

##### tsoutliers package

This [package](https://www.rdocumentation.org/packages/tsoutliers/versions/0.6-8) 
implements a procedure based on the approach described in [Chen and
Liu
(1993)](https://www.researchgate.net/publication/243768707_Joint_Estimation_of_Model_Parameters_and_Outlier_Effects_in_Time_Series) 
for automatic detection of outliers in time series. Time series data
often undergoes nonsystematic changes that alter the dynamics of the data, either
transitorily or permanently. The approach considers innovational outliers,
additive outliers, level shifts, temporary changes, and seasonal level shifts 
while fitting a time series model.

#### Numenta’s HTM (Hierarchical Temporal Memory)

Research organization [Numenta](https://numenta.com/) has introduced 
a machine intelligence framework for anomaly detection called _Hierarchical Temporal Memory_ (HTM). 
At the core of HTM are time-based learning algorithms that store and recall temporal patterns. Unlike
most other ML methods, HTM algorithms learn time-based patterns in
unlabeled data on a continuous basis. They are robust to noise and high-capacity, 
meaning they can learn multiple patterns simultaneously. The HTM
algorithms are documented and available through the open source 
Numenta Platform for Intelligent Computing (NuPIC). They're particularly suited to
problems involving streaming data and to identifying underlying patterns in data change over time,
subtle patterns, and time-based patterns. 

One of the first commercial applications to be developed using NuPIC is
[Grok](https://grokstream.com/), a tool that 
performs IT analytics, giving insight into IT systems to identify unusual
behavior and reduce business downtime. Another is
[Cortical.io](https://www.cortical.io/), which enables
applications in natural language processing.

The NuPIC platform also offers several tools, such as HTM Studio and
Numenta Anomaly Benchmark (NAB). HTM Studio is a free desktop tool that enables developers to find
anomalies in time series data without the need to program, code, or set parameters. 
NAB is a novel benchmark for evaluating and comparing algorithms for anomaly
detection in streaming, real-time applications. It is composed of over 50
labeled real-world and artificial time series data files, plus a novel scoring
mechanism designed for real-time applications.

Besides this, there are example applications available on NuPIC that include
sample code and whitepapers for tracking anomalies in the stock market, rogue
behavior detection (finding anomalies in human behavior), and geospatial tracking 
(finding anomalies in objects moving through space and time). 

Numenta is a technology provider and does not create go-to-market solutions for
specific use cases. The company licenses its technology and application code
to developers, organizations, and companies that wish to build upon it.
Open source, trial, and commercial licenses are available. Developers can use Numenta technology
within NuPIC via the AGPL v3 open source license. 

### Anomaly Detection as a Service

In this section, we survey a sample of the anomaly detection services available at the time of writing. Most
of these services can access data from public cloud databases, provide some
kind of dashboard/report format to view and analyze data, have an alert mechanism
to indicate when an anomaly occurs, and enable developers to view underlying causes. An anomaly detection 
solution should try to reduce the time to detect and speed up the time to
resolution by identifying key performance indicators (KPIs) and attributes that are causing the alert. We compare the offerings across a range of criteria, from capabilities to delivery.

#### Anodot

- *Capabilities*  
[Anodot](https://www.anodot.com/) is a real-time analytics and automated anomaly 
detection system that detects outliers in time series data and turns them into 
business insights. It explores anomaly
detection from the perspective of forecasting, where anomalies are identified
based on deviations from expected forecasts. 
The product has a dual focus: business monitoring (SaaS monitoring, anomaly detection, 
and root cause analysis) and business forecasting (trend prediction, "what ifs," and optimization).

- *Data requirements*  
Anodot supports multiple input data sources, including direct uploads and
integrations with Amazon’s S3 or Google Cloud storage. It's data-agnostic
and can track a variety of metrics: revenue, number of sales, number
of page visits, number of daily active users, and others.

- *Modeling approach/technique(s)*  
Anodot analyzes business metrics in real time and at scale by running
its ML algorithms on the live data stream itself, without reading or writing
into a database. Every data point that flows into Anodot from all data sources 
is correlated with the relevant metric’s existing normal model, and either is
flagged as an anomaly or serves to update the normal model. Anodot's philosophy is that
no single model can be used to cover all metrics. To allocate the optimal model for
each metric, they've created a library of model types for different signal
types (metrics that are stationary or non-stationary, multimodal or discrete,
irregularly sampled, sparse, stepwise, etc.). Each new metric goes through a 
classification phase and is matched with the optimal model. The model then
learns “normal behavior” for that metric, which is a prerequisite to identifying
anomalous behavior. To accommodate this kind of learning in real time at scale,
Anodot uses sequential adaptive learning algorithms which initialize a model of
what is normal on the fly, and then compute the relation of each new data point
going forward.

- *Point anomalies or intervals*  
Instead of flagging individual data points as anomalies, Anodot highlights
intervals. Points within the entire duration of the interval are considered
anomalous, preventing redundant alerts.

- *Thresholding*  
While users can specify static thresholds which trigger alerts, Anodot also
provides automatic defaults where no thresholding input from the user is
required.

- *Root cause investigation*  
Anodot helps users investigate why an alert was triggered. It tries to understand how
different active anomalies correlate in order to expedite root cause investigation and
shorten the time to resolution, grouping together different
anomalies (or “incidents”) that tell the story of the phenomena. These might be
multiple anomalous values in the same dimension (e.g., a drop in traffic
from sources A and B, but not C, D, or E), or correlations between different
anomalous KPIs, such as visits, conversions, orders, revenue, or error rate. Each
incident has an _Anomap_, a graphic distribution of the dimensions most impacted.
This is essentially a heat map that makes it easier to understand the whole
picture.

- *User interface*  
The Anodot interface enables users to visualize and explore alerts. With the
receipt of each alert, the user is prompted to assign it a binary score
(good catch/bad catch). This input is fed back into the learning model to
further tune it by providing real-life indications about the validity of its
performance. By training the algorithms with direct feedback on anomalies, users
can influence the system’s functionality and results.

- *Delivery*  
Notifications can be forwarded to each user through their choice of
channel(s). Anodot notification integrations include an API, Slack, 
email, PagerDuty, Jira, Microsoft Teams OpsGenie, and more.


#### Amazon QuickSight

- *Capabilities*  
[Amazon QuickSight](https://docs.aws.amazon.com/quicksight/latest/user/anomaly-detection-function.html) 
is a cloud-native business intelligence (BI) service that allows its users to create
dashboards and visualizations to communicate business insights. In early 2019,
Amazon’s machine learning capability was integrated with QuickSight to provide
anomaly detection, forecasting, and auto-narrative capabilities as part of the
BI tool. Its licensed software and pricing is usage-based; you only pay for
active usage, regardless of the number of users. That said, the
pricing model could end up being expensive, since anomaly detection tasks are
compute-intensive.

- *Data requirements*  
QuickSight requires you to connect or import structured data directly query a 
SQL-compatible source, or ingest the data into SPICE. There is a requirement on
the number of historical data points that must be provided, which varies based on the task (analyzing
anomalies or forecasting). There are also restrictions on the number of category
dimensions that can be included (for example: product category, region,
segment).

- *Modeling approach/technique(s)*  
QuickSight provides a point-and-click solution for learning about anomalous behavior and
generating forecasts. It utilizes a built-in version of the Random Cut Forest
(RCF) online algorithm, which not only can be noisy but also can lead to large amounts
of false positive alerts. On the plus side, it provides a customizable
narrative feature that explains key takeaways from the insights generated. For 
instance, it can provide a summary of how revenue compares to a previous
period or a 30-day average, and/or highlight the event in case of an anomaly. 

- *Point anomalies or intervals*  
Anomalous events are presented discretely, on a point-by-point basis. If an
anomaly lasts more than a single time unit the system will flag several events,
which could be noisy and redundant.

- *Thresholding*  
Anomaly detection with QuickSight employs a thresholding approach to trigger
anomalous events. The user provides a threshold value (low, medium, high, very 
high) that determines how sensitive the detector is to anomalies:
expect to see more anomalies when the setting is low, and fewer anomalies when
it's high. The sensitivity is determined based on standard
deviations of the anomaly score generated by the RCF algorithm. This approach 
can be tedious, especially when there are multiple time series being
analyzed across various data hierarchy combinations, and introduces the need for manual
intervention.

- *Root cause investigation*  
Users can interactively explore anomalies on the QuickSight dashboard or report
to help understand the root causes. The tool performs a contribution analysis
which highlights the factors that significantly contributed to an anomaly. If
there are dimensions in the data that are not being used in the anomaly detection, it's possible to add up to four of them for the contribution analysis task. In
addition, QuickSight supports interactive "what-if" queries. In these, some of the
forecasts can be altered and treated as hypotheticals to provide conditional
forecasts. 

- *User interface*  
QuickSight provides a basic reporting interface. From a UI
perspective, it is fairly unexceptional. For instance, it lacks a way to 
understand the overall picture with anomalous points (do the anomalies have
some common contributing factors?). Furthermore, the forecasted values do not
have confidence intervals associated with them, which would help the end user
visually understand the magnitude of an anomaly. As it stands, there is no
basis for comparison. 

- *Delivery*  
QuickSight dashboards and reports can be embedded within applications, shared among
users, and/or sent via email, as long as the recipients have a QuickSight
subscription. 

#### Outlier.ai

- *Capabilities*  
[Outlier.ai](https://outlier.ai/) is licensed software that uses artificial intelligence to automate
the process of business analytics. It can connect to databases provided by cloud
services and automatically provides insights to your inbox, without the need to
create reports or write queries. Outlier works with customers across industry 
segments, applying ML to automatically serve up business-critical
insights.

- *Data requirements*  
Outlier can connect to databases provided by cloud services.

- *Point anomalies or intervals*  
Anomalous events are presented discretely, on a point-by-point basis. 

- *Root cause investigation*  
Outlier allows customers to not only surface key insights about business
changes automatically, but also identify the likely root causes of those
changes; this feature guides teams in making quick, informed business decisions. 
Teams can easily share stories through PDFs, PowerPoint-optimized images, or
auto-generated emails, annotated with their comments.

- *User interface*  
The UI is similar to that provided by most standard BI tools, making it fairly
user-friendly.

- *Delivery*  
The generated dashboards can be embedded within applications, shared among users, and/or
sent via email.

#### Vectra Cognito

- *Capabilities*  
In simple terms, Vectra.ai’s flagship platform, [Cognito](https://www.vectra.ai/), can be described as an
intrusion detection system. It's a cloud-based network detection and response
system that performs a number of cybersecurity-related tasks, including network
traffic monitoring and anomaly detection. For this latter task, metadata 
collected from captured packets (rather than via deep packet inspection) is
analyzed using a range of behavioral detection algorithms. This provides
insights about outlier characteristics that can be applied in a wide range of 
cybersecurity detection and response use cases. Cognito works with both
encrypted and unencrypted traffic.

- *Data requirements*   
Cognito can connect to databases provided by cloud services. It also uses
metadata drawn from Active Directory and DHCP logs. 

- *Modeling approach/technique(s)*   
According to a
[whitepaper](https://content.vectra.ai/rs/748-MCE-447/images/WhitePaper_2019_The_data_science_behind_Cognito_AI_threat_detection_models_English.pdf?mkt_tok=eyJpIjoiWkRGaVpHVmtaVGxrTkdFeiIsInQiOiJ2RVhkK3M0cHU3dXNQRDZ2YnA3QW16K0ZKVFVEK1lDeFRwcTZPMGxXZlB0clhOYmhPaVBXenkzRmY1Ylwvakp5d2FcL1dSakVKbDZhcHZtNEdZU1A3aHFMYkpxVlZHWXllXC9xUGRPOXNtZ0NyTFRjTitxUlVkaXBzNFdiQlBaUUxwVSJ9) 
published by Vectra, a mix of ML approaches are used
to deliver the cybersecurity features the platform supports, in both global and
local (network) contexts. For example, supervised learning techniques such as
random forests can help to address cyberthreats associated with suspicious HTTP
traffic patterns. Drawing on large-scale analysis of many types of malicious 
traffic and content as well as domain expertise, the random forest technique can be used to identify
patterns of command-and-control behavior that don’t exist in benign HTTP
traffic. 
Cognito also uses unsupervised ML techniques such as _k_-means clustering to
identify valid credentials that have potentially been stolen from a compromised
host. This type of theft is the basis of cyberattacks such as pass-the-hash and
golden ticket. Elsewhere, deep learning has proven effective in detecting
suspicious domain activity; specifically, the detection of 
algorithmically generated domains that are set up by cyberattackers as the
frontend of their command-and-control infrastructure. 

- *Point anomalies or intervals*  
Anomalous events are presented discretely, on a point-by-point basis. 

- *Thresholding*   
The scoring of compromised hosts by the Vectra Threat Certainty Index allows 
security teams to define threshold levels based on a combination of factors. 

- *Root cause investigation*  
All detection events are correlated to specific hosts that show signs of threat
behaviors. In turn, all context is assimilated into an up-to-the-moment score of
the overall risk to the organization. 

- *User interface*  
Vectra’s Cognito platform delivers detection information via a simple
dashboard which displays information such as a prioritized (in terms of risk)
list of compromised hosts, changes in a host’s threat and certainty scores, and 
"key assets" that show signs of attack. 

- *Delivery*  
The platform supports information sharing by security teams on demand, or on a
set schedule managed by its customizable reporting engine. Real-time 
notifications about network hosts, with attack indicators that have been
identified (with the highest degree of certainty) as posing the biggest risk,
are also supported.

#### Yahoo’s Anomaly Detector: Sherlock 

- *Capabilities*  
[Sherlock](https://github.com/yahoo/sherlock) is an open source anomaly detection 
service built on top of [Druid](http://druid.io/) (an open source, distributed data store). It leverages the [Extensible
Generic Anomaly Detection System (EGADS)](https://github.com/yahoo/egads) 
Java library to detect anomalies in time series
data. Sherlock is fast and scalable. It allows users to schedule jobs on an hourly,
daily, weekly, or monthly basis (although it also supports ad hoc real-time
anomaly detection requests). Anomaly reports can be viewed from Sherlock's
interface, or received via email.

- *Data requirements*   
Sherlock accesses time series data via Druid JSON queries and uses a Redis
backend to store job metadata, the anomaly reports (and other information) it
generates, as well as a persistent job queue. The anomaly reports can be
accessed via direct requests using the Sherlock client API or delivered via
scheduled email alerts. 

- *Modeling approach/technique(s)*  
Sherlock takes a time series modeling-based approach to anomaly detection using
three important modules from the EGADS library: Time Series Modeling, Anomaly
Detection, and Alerting. The Time Series Modeling module supports the use of
historical data to learn trends and seasonality in the data using models 
such as ARIMA. The resulting values are applied to the models
that comprise the Anomaly Detection module. These models support a number of
detection scenarios that are relevant in a cybersecurity context (e.g., outlier
detection and change point detection). The Alerting module uses the error metric 
produced by the anomaly detection models and outputs candidate anomalies based on
dynamically learned thresholds, learning to filter out irrelevant anomalies over
time. 

- *Point anomalies or intervals*  
Anomalous events are presented both discretely, on a point-by-point basis, and
as intervals. 

- *Threshold*  
Thresholds are learned dynamically. No thresholding input from the user is
required/supported.

- *Root cause analysis*  
Out of the box root cause analysis is not supported.

- *User interface*  
Sherlock’s user interface is built with [Spark Java](http://sparkjava.com/), 
a UI framework for building
web applications. The UI enables users to submit instant anomaly analyses, 
create and launch detection jobs, and view anomalies on both a heat map and a
graph. 

- *Delivery*  
Scheduled anomaly requests are delivered via email or directly via API-based
queries. 
