# Abstract

Inner Speech or Speech Imagery-based BCIs aim to help individuals affected by Aphasia or those in a locked-in state interact with their environments through internalised speech commands decoded from brain signals. Despite significant activity in this field, previous research endeavours have yet to achieve significant progress that would render BCIs suitable for widespread adoption by the public. This is primarily due to the inherent complexities of brain signals, particularly concerning speech imagery, and the variations in individual idiosyncrasies. In this regard, Graph Neural Networks, lauded for their ability to learn to recognise brain data, were assessed on an Inner Speech dataset acquired using EEG to determine if state-of-the-art results could be achieved. Five GNN models – Graph Convolutional Network (GCN), Graph Attention Network (GAT), Graph Isomorphism Network (GIN), ChebNet, and ChebNet Augmented with EdgeConv – were used, achieving highest average accuracy of 26.7%, with individual subject accuracies reaching up to 32.5% in the classification of four words. While these results fall within the scope of previous endeavours, they demonstrate the capability of GNNs to learn and recognise Inner Speech patterns, indicating room for further improvements.

# Dataset

The dataset proposed for this project was collected and processed by Nieto et al. The dataset consists of electroencephalogram (EEG) recordings taken from ten naive BCI users. Among these participants, four are female and six are male, with an average age of 34 years and a standard deviation of 10 years.

During the recordings, participants performed mental tasks corresponding to the words 'up,' 'down,' 'left,' and 'right' under three distinct speech conditions: pronounced speech, visualised condition, and inner speech. The pronounced speech and visualised condition trials were proposed with the sole purpose of finding cross-related signals matching those activated during the inner speech.

Participants were asked to participate in 3 different sessions, during which they performed the aforementioned tasks across multiple trials. All participants completed 200 trials in each of sessions 1 and 2, but number of trials completed dropped in session 3, mainly due to fatigue and tiredness. In each session, 40 pronounced speech, 80 inner speech, and 80 visualised condition related tasks were performed by participants. In this study, only sessions 1 and 2, and the 80 inner speech trials in each session were used.

In each trial, participants undergo a structured sequence of events. At the start, they engage in a 0.5 seconds concentration interval, focusing on a white dot displayed on the screen. At the 0.5 second mark, an arrow appears, indicating the direction for the mental task. The arrow remains visible for an additional 0.5 seconds, after which it disappears, and the white dot reappears. Participants then commence the mental task, which continues for 2.5 seconds. At the 3.5 second mark, the white dot transitions to blue, signalling the end of the mental task and the initiation of a 1 second resting period. Each trial spans a total duration of 4.5 seconds, with inter-trial intervals varying between 1.5 and 2 seconds, introducing variability between consecutive trials.

![Trial workflow](https://github.com/walerazaq/bci-eeg-gnns/assets/165695047/828c47c7-fb5f-4529-98c3-b2a0fb61ab6c)

Image credit: Nieto, Nicolás, et al. “Thinking out Loud, an Open-Access EEG-Based BCI Dataset for Inner Speech Recognition.” Scientific Data, vol. 9, no. 1, 14 Feb. 2022, https://doi.org/10.1038/s41597-022-01147-2. 

The dataset is publicly available with an "open-access", and is downloadable from OpenNeuro: https://openneuro.org/datasets/ds003626/versions/2.1.0

# Methodology

![Methodology](https://github.com/walerazaq/bci-eeg-gnns/assets/165695047/9f8edd57-8051-47c7-b156-7e657fdcdad5)


* **Signal Preprocessing:**
The signal preprocessing was caried out using the FieldTrip Toolbox within MATLAB. The preprocessing done include: Re-referencing, Demeaning, Baseline Correction, Bandpass Filtering, Discrete Fourier Transform (DFT), 

* **Graph Creation:**
Functional connectivity network was the foundation for constructing the graphs used in this study. The functional connectivity analysis was conducted in the frequency domain using the Coherence method.

  To identify the optimal frequency band for effectively resolving the signal data into functional connectivity for recognising inner speech signals, the analyses was performed over Alpha, Beta, Gamma, Delta, and Theta frequency bands. Additionally, a wider frequency band from 0.5 Hz to 45 Hz was also used. Each set of data was used to train a Graph Convolutional Network (GCN) model, with the Alpha frequency band dataset performing
  the best.

  Also, all functional connectivity analyses were conducted over the 2.5-second action interval within each trial. This approach ensured the exclusion of random signals, or noise, stemming from other segments of the trial, such as the concentration and resting intervals.

* **Node Features Extraction:**
For every graph in the data, 8 features were extracted for the nodes. These node features are as follows: Power Spectral Density (PSD), Efficiency, Betweenness, Clustering Coefficient, Strength, and the three Hjorth Parameters – Activity, Mobility, and Complexity. The PSD feature was extracted using the FieldTrip Toolbox in MATLAB, whereas the other features, excluding the Hjorth parameters, were extracted using the Brain Connectivity Toolbox.

* **Model Setup:**
Each model comprises convolutional layers with ReLU activation functions in the hidden layers. Afterwards, a graph pooling layer aggregates information from the node level to the graph level to facilitate graph-level classification. Following this, a fully connected MLP layer is employed for the classification of graphs. The training process employs the _Cross Entropy loss_, _Adam optimizer_, and the _CosineAnnealingLR learning rate scheduler_.

  This modelling and training process was conducted over the PyTorch and PyTorch Geometric frameworks.


  ![Modelling](https://github.com/walerazaq/bci-eeg-gnns/assets/165695047/ea55defb-0f2f-481d-a394-83f7f9c72537)



* **Hyperparameter Selection:**
To ensure optimal model training, a hyperparameter selection process was conducted using the Ray Tune library. This involved specifying a configuration of hyperparameters and their respective ranges for testing. The library then performed an exhaustive search to identify the most effective combination of hyperparameter values for each model. This iterative process was repeated for all five models, and the optimal parameters were subsequently employed for final training.

  The hyperparameters examined included: Number of hidden layers, Learning rate, Weight decay, Batch size, Number of attention heads for the GAT model, Chebyshev filter size for the ChebConv layers

* **Evaluation Methods:**
Two model evaluation methods were employed in this study: one served as the primary evaluation method for the final model training, while the other was employed during the hyperparameters selection stage.

  During the hyperparameters selection, a train-validation-test split was applied. The dataset was partitioned into these three subsets, which were then used for hyperparameters selection using Ray Tune. For the final model training, a Leave One Subject Out (LOSO) evaluation method was adopted. LOSO is similar to Cross Validation, however, it is performed on a subject-by-subject basis.

  The motivation for LOSO stems from the fact that BCIs are adopted for public use. For that reason, LOSO aims to provide insight into how each model would generalise to new, real-world data that it has not encountered previously. The train-test split method falls short in this regard because subjects' data are randomly allocated to both the training and test sets. During training, the model may inadvertently learn individual subject-specific patterns rather than the underlying patterns of performing the mental task. This can lead to overfitting to subjects' idiosyncrasies. Consequently, the presence of the same subjects' data in the test subset may falsely inflate the model's performance, as it is already familiar with specific subjects’ data.
LOSO addresses these limitations by systematically excluding each subject's data from the training at different stages and using it to test the model’s performance. This approach ensures that the model learns more generalised patterns and provides a more reliable evaluation of its performance on unseen data.

# Results

The first table below shows the results of the hyperparameter selection process for each model. These selected hyperparameters were employed in training the final models. 

The seconds table shows the GCN model slightly outperformed the others, achieving an average accuracy of 26.99%. The ChebNet model trailed closely behind with an accuracy of 26.69%. The GAT model followed with 26.04%, while ChebNet + EdgeConv and GIN performed the least, with accuracies of 25.56% and 25.01%, respectively.

Examining individual folds, the ChebNet model demonstrated the highest performance, achieving an accuracy of 32.5% for subject 7. Similarly, the GCN model achieved an accuracy of 28.75% for the same subject. In the fold corresponding to Subject 2, both the GCN and ChebNet models achieved seemingly higher accuracies, each reaching 30.63%.

This observation could mean that the models excelled at learning from training data that excluded subjects 7 and 2, or it could indicate the presence of particularly favourable data from these subjects that the models could effectively generalise to.

![Hyperparameter Results](https://github.com/walerazaq/bci-eeg-gnns/assets/165695047/9a5a0bed-da57-4b9a-9ba5-2aa7883b0e04)

![Train Results](https://github.com/walerazaq/bci-eeg-gnns/assets/165695047/f640a713-e863-44db-a5a9-cc9861bb82b6)

# Comparing Results to Related Works

To further assess the performances of the models in this study, the GCN model, being the highest performer, was benchmarked against previous related studies. Specifically, works that employed same dataset and used similar evaluation approaches were selected, ensuring a fair comparison.

![Comparison Table](https://github.com/walerazaq/bci-eeg-gnns/assets/165695047/758aa075-6884-409b-a46a-305326320f24)

[1] Berg, Bram van den, et al. “Inner Speech Classification Using EEG Signals: A Deep Learning Approach.” IEEE 2nd International Conference on Human-Machine Systems (ICHMS), 2021, https://doi.org/10.1109/ICHMS53169.2021.9582457.

[2] Gasparini, Francesca, et al. “Inner Speech Recognition through Electroencephalographic Signals.” ArXiv (Cornell University), 11 Oct. 2022, https://doi.org/10.48550/arxiv.2210.06472.

[3] Han Wei Ng, and Cuntai Guan. “Efficient Representation Learning for Inner Speech Domain Generalization.” Lecture Notes in Computer Science, 1 Jan. 2023, pp. 131–141, https://doi.org/10.1007/978-3-031-44237-7_13.

# Conclusion & Recommendation

Despite displaying lower performance levels compared to some others, the models in this study demonstrated an ability to learn and recognise Inner Speech to varying degrees. This indicates potential for further research and development. In fact, some of the better performing models in other studies achieved higher accuracies only after incorporating additional methods into their primary approaches.

As a recommendation, future studies could explore the integration of additional sub-processes to enhance the quality of inner speech signals before inputting them into a GNN model for classification. For example, the adoption of Variational Autoencoders (VAE) as an additional signal preprocessing step might facilitate better performance.

This method can be used to reform and resolve brain signals data into more generalised representations of each mental task. Notably, in Han Wei Ng and Guan’s work [3 above], this method led to an improvement of results by over 5%.

Additionally, but as a broader suggestion, further research endeavours could consider involving a greater number of BCI-literate subjects to facilitate the acquisition of higher-quality EEG signals. By training models on such data, there exists the possibility of developing more robust models. Subsequently, extending this framework by performing transfer learning on individual subjects or making slight recalibration could aid in adapting these models to suit individual users more effectively.
