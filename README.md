# Comparative Analysis of GNN Architectures in Classifying EEG-Based Signals for Inner Speech Recognition

Brain-Computer Interfaces (BCIs) are technologies primarily developed to improve the quality of life of people who have lost the fundamental abilities to interact with their immediate environment. The final goal of BCIs is to help patients have an effective interaction with their environment through external devices, such as computers, assistive appliances, or neural prostheses. BCIs are also used for cognitive rehabilitation.

Speech-related BCIs offer efficient vocal communication methods for operating devices via speech commands decoded from brain signals. Frameworks and methods utilised to create and execute these interactions between the brain and BCIs are called paradigms. The most common speech paradigms are Imagined Speech, Silent Speech, and Inner Speech.

“Inner Speech is defined as the internalized process in which the person thinks in pure meanings, generally associated with an auditory imagery of own inner voice”. This form of cognition doesn't require physical articulation, rendering it advantageous for individuals with restricted motor functions, such as those affected by Aphasia (stroke) or Amyotrophic lateral sclerosis (ALS).

Research into the field of BCIs based on inner speech has been ongoing for years. Despite significant efforts, most studies have yet to achieve state-of-the-art results, particularly when dealing with an increasing range of speech classes. For instance, in a recent study by Han Wei Ng and Guan, the average accuracy in recognising four classes of words – up, down, left, and right – was only 31.15%.

Graph Neural Networks (GNNs) offer an intelligent approach to understanding the complex structure of the brain, benefiting from the fact that the brain is readily represented as a network (a graph) of elements and their pairwise interconnections.

This study aims to develop a novel Brain-Computer Interface (BCI) framework using EEG data from Inner Speech and GNN models to push the current boundaries and potentially advancing the efficacy of BCIs to empower individuals with paralysis or those requiring such assistance.

# Background Study

Speech Imagery is a term that is often used interchangeably with other speech paradigms, including inner speech; however, researchers frequently employ the term ambiguously to describe various speech paradigms that may differ significantly. For instance, Hernandez-Galvan et al. defined speech imagery as "imagining speaking some phonemes, syllables, vowels, or words 'without producing any sound or doing facial movements'," while noting that speech imagery is also referred to as inner, silent, imagined, or covert speech. Although these speech paradigms are related, they differ in execution. Silent speech, for instance, involves articulation similar to normal speech, often accompanied by mouth or muscular gestures but without emitted sound. In contrast, inner speech is an internalized process and does not involve physical articulation of any part of the body.

Compared to brain signals involved in movement, speech processing appears to involve more complex neural networks across distinct cortical areas. This complexity underscores the challenges faced by previous research endeavours aiming to develop state-of-the-art BCIs based on inner speech. Berg et al. attained an average accuracy of 29.7% using Convolutional Neural Networks (CNNs) to classify four inner speech words. Merola et al. employed Support Vector Machines (SVM), Random Forest, and k-Nearest Neighbours (kNN), with the highest accuracy reaching 35.3% for the same problem. Gasparini et al. explored various methods and achieved the highest average accuracy of 36.1% with the Bidirectional Long Short-Term Memory (BiLSTM) architecture. More recently, Han Wei Ng and Guan introduced an implementation of a few/zero-shot subject-independent meta-learning framework for the multi-class inner speech, achieving an average accuracy of only 31.15%.

Recognition of graph models as inherently suitable for representing brain signals underscores the potential for achieving superior outcomes using this framework in this field. Choi et al. substantiated this notion by comparing GNN models with classical shallow models for brain network analysis, concluding that GNNs outperform these alternatives when analysing brain network data. In a more recent study, Chen et al. also evaluated GNN-based models against Multi-Layer Perceptron (MLP)-based models for Alzheimer's disease classification using brain data, concluding that GNNs demonstrate superior performance over MLP-based models.

# Dataset to Use

The dataset proposed for this project was collected and processed by Nieto et al. The dataset consists of electroencephalogram (EEG) recordings taken from ten naive BCI users. Among these participants, four are female and six are male, with an average age of 34 years and a standard deviation of 10 years.

During the recordings, participants performed mental tasks corresponding to the words 'up,' 'down,' 'left,' and 'right' under three distinct speech conditions: pronounced speech, visualised condition, and inner speech. The pronounced speech and visualised condition trials were proposed with the sole purpose of finding cross-related signals matching those activated during the inner speech.

Participants were asked to participate in 3 different sessions, during which they performed the aforementioned tasks across multiple trials. All participants completed 200 trials in each of sessions 1 and 2, but number of trials completed dropped in session 3, mainly due to fatigue and tiredness. In each session, 40 pronounced speech, 80 inner speech, and 80 visualised condition related tasks were performed by participants. In this study, only sessions 1 and 2, and the 80 inner speech trials in each session will be considered.

![Capture](https://github.com/walerazaq/bci-eeg-gnns/assets/165695047/c8f2963d-88a7-42a0-accd-a1c32c8df97c)

# Analysis to be Undertaken

The EEG signal data will be pre-processed according to the methodologies described by Nieto et al. who collected the data in their Thinking Out Loud publication (of the dataset), which include procedures such as event checking and correction, re-referencing, digital filtering, bandpass filtering, baseline correction, and demeaning.

Graphical brain connectivity networks will be derived from the data using the functional connectivity technique by aggregating nodes and edges into a connection matrix. These graphical representations typically consist of vertices (V) representing nodes, edges (E) representing timeseries data collected during the performance of tasks by participants, and weights (W) indicating the statistical relationships between nodes. These graph structures enable the training of Graph Neural Network (GNN) models.

Five distinct GNN architectures will be employed in this study to determine the optimal performing model. The models include Graph Convolutional Network (GCN), Graph Attention Network (GAN), Graph Isomorphism Network (GIN), as well as ChebNet. Additionally, the study will explore ChebNet augmented with EdgeConv layers, proposed by Yash Semlani et al. The goal of each model is to be able to recognise the four distinct words performed by the participants during inner speech.

Performance assessment will primarily rely on Accuracy, as well as model validation metrics such as loss and training accuracy. The analysis will be conducted using Python and MATLAB, with model training, validation, and optimisation executed on the PyTorch and PyTorch Geometric framework. To facilitate the rapid training of models, the entire process will be conducted using Kelvin 2, a scalable High-Performance Computing (HPC) and Research Data Storage environment.

# Legal / Social Ethical & Sustainability Issues

Collector of the data Nieto et al., noted that the experimental protocol was approved by the “Comité Asesor de Ética y Seguridad en el Trabajo Experimental” (CEySTE, CCT-CONICET, Santa Fe, Argentina, https://santafe.conicet.gov.ar/ceyste/) and that the participants gave their written informed consents. Also, the dataset is publicly available with an "open-access" and is downloadable from OpenNeuro.
