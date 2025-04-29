# VR Activity Classification
Identifying user activities in a VR game using neural networks trained on network traffic data collected via Wireshark. The project involves analyzing features like packet count, inter-packet time, timestamps, and packet length.

## Methodology
- Data collection using Wireshark on Meta Quest Pro.
- Neural network techniques (e.g., RNNs) for activity classification.
- Focused on network traffic analysis in a VR environment.
- Evaluation of the model’s performance using metrics such as Accuracy, precision and recall.

## Project Goals
- Develop a dataset of VR network traffic.
- Train and evaluate different neural network architectures.
- Provide insights into activity classification from network data.

## Tools and Libraries
- Wireshark
- Python (TensorFlow/PyTorch)
- Scikit-learn, NumPy, Pandas

# Part-1

This project explores the possibility of identifying user activities in a VR game using neural networks. The VR game "GYM CLASS - BASKETBALL VR" was selected for this project due to its accessible training mode, which provides a controlled environment for performing activities without interference from other players in multiplayer settings. Network traffic generated during gameplay, captured using Wireshark, will serve as the input for analysis. Targetted in-game activities for identification are :

- **Walking**
- **Talking**
- **Ball Throwing**
- **Paused State**
- **No Activity(Idle State)**

 The primary objective is to develop a labeled dataset and train neural network models to classify activities based on extracted traffic features. By successfully identifying these activities, this work can demonstrate how VR activities are encoded in network traffic and the potential privacy implications of such data. The cornerstone of this project is the manual creation of a dataset tailored to the activities in GYM CLASS - BASKETBALL VR. Every second of gameplay is treated as a datapoint, with network traffic features extracted and manually labeled for the corresponding activity.  

 ### **Data Collection Process (Subject to change)**:
1. **Gameplay Sessions**  
   - Network traffic is being recorded via Wireshark while performing predefined activities in the VR game.  
   - Data is collected using packet capture tools and stored as raw traffic logs.
  
2. **Feature Extraction**  
   - The dataset will include traffic-based features of UDP (User Datagram Protocol) network packets extracted for each activity interval. UDP is often used for time-sensitive applications like gaming, video playback, and DNS lookups.
   - The packet characteristics will be the key features which are: average size of packets per second, number of packets passed per second, and timing intervals between corresponding packets.   

3. **Labeling**  
   - Activities will be manually labeled for each second of gameplay based on the performed action.  
   - Labels include the aforementioned *Walking, Talking, Ball Throwing, Paused,* and *No Activity*.  

4. **Traffic Segmentation**  
   - Traffic data will be segmented into one-second intervals, creating a structured dataset where each row corresponds to a single activity instance.  




# Part-2: Datasets

The dataset for the project was collected manually with the help of Wireshark on the Meta Quest Pro VR headset. Network traffic logs were recorded during performing various activities in the VR game GYM CLASS - BASKETBALL VR. Packet capture (PCAPNG) files and their processed counterparts in CSV format are included in the dataset. These processed files contain traffic-based features extracted for specific activity intervals, which I think will facilitate the training of neural network models.

### **Activities:**
As planned before in Part-1, I collected data and manually labeled samples of the following activities:

- **Walking**
- **Talking**
- **Ball Throwing**
- **Paused State**
- **No Activity(Idle State)**

### **Details of the features:**
Each second of network traffic that is captured is considered a datapoint. The following characteristics are present in the processed dataset:

- **Time (Seconds):** Recorded time for keeping track of starting and ending of an activity.
- **No. of Bits:** The total number of bits that were transferred throughout the time frame.
- **No. of Packets:** The total number of packets sent and received during the time frame.
- **Average Packet Length (Bytes):** The average length of the interval's packets.
- **Average Inter-Packet Arrival Time:** The average amount of time in the interval between consecutive packets.

Features are aggregated (mostly averaged) at 1-second intervals. Additionally, the activities were manually labeled and tracked while they were being performed and recorded. The data was gathered in a regulated virtual reality setting (training mode) to remove player interference.

### **Sample Data:** 
Below is a snapshot of the processed dataset:  

| Time (Seconds) | No. of Packets | No. of Bits | Activity      | Avg. Packet Length (Bytes) | Avg. Inter-packet Arrival Time |
|----------------|----------------|-------------|---------------|-----------------------------|--------------------------------|
| 97             | 70             | 113840      | Walking       | 203.29                     | 0.0133                         |
| 98             | 72             | 118064      | Walking       | 204.97                     | 0.0127                         |
| 99             | 69             | 119128      | Walking       | 215.81                     | 0.0134                         |
| 100            | 70             | 116816      | No Activity   | 208.60                     | 0.0131                         |
| 163            | 70             | 110608      | No Activity   | 194.19                     | 0.0110                         |
| 164            | 71             | 108120      | Talking       | 190.35                     | 0.0125                         |
| 165            | 69             | 126520      | Talking       | 229.20                     | 0.0126                         |
| 166            | 71             | 135008      | Talking       | 237.69                     | 0.0135                         |



### **Drive link of Dataset:** https://drive.google.com/drive/folders/1pI8XeaFCjSOEUK7IC5u5HBcOUWC2gaCw?usp=sharing

To facilitate efficient training, hyperparameter tuning, and final evaluation, the dataset is then divided into subsets for training (60%) validation (20%) and testing (20%). Using the existing features, a neural network model (such as an RNN) will first be trained to classify activities. Additional statistical features like variance, mean, and patterns from packet data will be added to the model to improve its predictive accuracy if its performance is subpar. For better outcomes, feature engineering and model tuning will be improved through this iterative process.

# Part-3: First update

Using network traffic data recorded before while performing different tasks in virtual reality (VR) environments, this project focuses on activity recognition. Features such as time, the number of packets, the number of bits, average packet length, average inter-packet arrival time, and associated activity labels were included in the dataset. The objective was to create a classification model that can use these features to predict activities with accuracy. 

The code uses a methodical approach to guarantee accuracy and dependability when classifying VR activities based on network traffic data. The dataset is first preprocessed to fix errors and get it ready for modeling. Activity labels are then encoded for machine learning model compatibility, and features like packet count, bit rate, and inter-packet arrival times are scaled using StandardScaler for numerical stability. Using TensorFlow/Keras, a deep learning model is constructed, and the best-performing configurations are found by optimizing the layers and parameters using the Keras Tuner's Hyperband. During the training phase, overfitting is reduced by using strategies like dropout regularization, and progress is evaluated during the validation phase using metrics like accuracy and loss. The code can be found here:

https://github.com/muhaiminsk/vr-activity-classification/blob/e882a1402d5623a92d0b26834d8a77792690ff84/VR_Activity_classf.ipynb

### **Challenges Encountered**
**1. Imbalance and Data Quality:**
Although there are several activity classes in the dataset, their distributions are not uniform. While "Talking" and "Ball Throwing" have fewer samples, "No Activity" and "Paused" are overrepresented. The model's capacity to generalize across all classes is impacted by this imbalance. Although employing class-weight adjustments, oversampling, or undersampling may be beneficial, it has proven difficult to do so without adding bias.

**2. Feature Engineering:**
Although the numerical features offer a foundation for categorization, there is limited direct interpretability in them. Numerical stability is ensured through standardization using Scikit-learn's StandardScaler; however, more research is needed to fully understand the relationship between features and activities. Developing domain-specific features or transforming existing ones to better capture activity nuances has proven difficult.

**3. Tuning Hyperparameters:**
Although the Hyperband algorithm of the Keras Tuner is useful for exploring hyperparameter space, there is a considerable computational overhead. Several training iterations are needed to adjust the number of units in hidden layers and learning rates in the experiments. Even though GPUs are readily available, the time required for validation and tuning causes delays in achieving ideal configurations.

**4. Generalization of the Model:**
According to preliminary assessments, the model does well on the training set but has trouble staying accurate on the test data. There is clear overfitting, particularly for some classes with small datasets. To lessen this problem, strategies like dropout regularization and simplifying the model are being explored.

**5. Data Preprocessing for Test Sets:**
It is essential that training and test datasets be consistent. However, errors can occasionally arise from differences in label encoding and feature scaling during preprocessing. Careful monitoring of preprocessing pipelines is necessary to guarantee that the transformations for the two datasets are identical.

**6. Interpretation of Output:**
Complexity is added when predictions are mapped to labels that are readable by humans. The process needs to be smooth and error-free to prevent output file inconsistencies.

### **Initial Findings**
The accuracy of the current model is 95.4% on validation data, and it shows a consistent improvement over 50 training epochs. On test data, however, performance is roughly 68%, raising concerns about overfitting. The most frequently misclassified activities are those with fewer samples, like "Talking" and "Ball Throwing." "No Activity" and other dominant classes have more accurate predictions. These results highlight the need for improved feature refinement and management of class imbalance.

### **Next Actions**
To solve the issues found and enhance the project's outcome, I plan to implement the following tasks:

1. Analyzing the features' distribution and significance in predicting various activities by conducting statistical analyses. By doing this, the feature set can be improved for better model training by identifying features that are redundant or less informative.
2. Creating strong evaluation metrics, such as F1-score, precision, and recall, to give a thorough picture of model performance in all classes, particularly underrepresented ones.
3. This issue of predicted activity readibility can be resolved by inverse label encoder transformations, but in order to avoid inconsistent output files, the procedure must be seamless and error-free.



# Part-4: Second update
### **Improving Generalization with SMOTE:**

By using SMOTE (Synthetic Minority Oversampling Technique) and improving the neural network architecture, I tried to address class imbalance and enhance model generalization. Although the original model's test accuracy was 74.95%, it displayed overfitting (training accuracy: 87.77%, validation accuracy: 67.74%).

Important modifications consist of:


Class Balancing: To address dataset imbalance, use SMOTE from the imbalanced-learn library.

Hyperparameter tuning: enlarged search space for learning rates (1e-3 to 1e-4) and hidden layer units (64-256).

### **Challenges Encountered**
**1. Validation accuracy collapse:** 
Validation accuracy collapsed to 0% as a result of a custom SMOTE implementation because of
a) Incorrect tensor operations when creating synthetic samples.
b) Incorrect binary classification configuration (using sigmoid activation rather than softmax).

So, as a probable fix, the battle-tested imbalanced-learn SMOTE implementation was used as a solution and it helped.

**2. Leakage of Validation Data:** 

Performance was artificially inflated by the initial SMOTE application, which included both training and validation data. So, before using SMOTE on synthetic data, training and validation splits were separated as a fix.

**3. Unstable Hyperparameters:**

At first, larger architectures (256 units) produced unpredictable loss behavior. So, I lessened the learning rate search space and added gradient clipping.

**4. Misunderstanding of the class:**

Due to lingering imbalance, minority classes ("Ball Throwing", "Talking") were commonly misclassified. As an enhancement, SMOTE and class-weighted loss functions were combined.

### ** Evaluation:**

Training Accuracy is 85.58% and Validation Accuracy 62.20%.

<img width="504" alt="Screenshot 2025-04-24 at 1 36 00 AM" src="https://github.com/user-attachments/assets/5b532741-d69a-42c3-8121-55382fe9ec3e" />

**Confusion matrix:**





![conf_mat](https://github.com/user-attachments/assets/114d0c53-62aa-446e-af2b-9720ceffef04)


### **Evaluation Method Justification:**

Precision/Recall/F1: Critical for imbalanced classes (e.g., "Walking" vs "No Activity") to measure true class-specific performance beyond accuracy.

Confusion Matrix: Identifies misclassification patterns (e.g., "Walking" confused with "No Activity").

### **Analysis and Short Commentary**
Training Accuracy (85.58%) being larger than Validation Accuracy (62.20%) points to overfitting which means model memorizes training data but fails to generalize. This is happening likely due to insufficient regularization and class imbalance. There is room for improvement in this area.

Test Accuracy (77.69%) indicates good performance on majority classes (No Activity, Paused).But it fails somewhat when it comes to minority classes (Walking: 35% F1). Current features likely lack discriminative power for Walking.


### **How to Run:**
Files: (All files are pushed to the git)

Train: train_merged.csv 

Test: test_vrclass_2.csv 

Code: Demo_VR_Activity_classf.ipynb = https://colab.research.google.com/drive/1BWVUEJ-G_8Vm7XP5q6oWq5cjjv4f5sh3?usp=sharing

1. Run first cell for requirements
2. Run Second cell for the main training and results
3. First provide the training file (train_merged.csv) when asked.
4. Then provide the testing file (test_vrclass_2.csv) when asked.



# VR Activity Classification Final Report

### **Project Overview**

The goal of this project is to use neural networks trained on network traffic data to classify user activities in the VR game GYM CLASS-BASKETBALL VR. The selected game offers a controlled environment for gathering data. Targeted activities include talking, walking, throwing a ball, pausing, and No Activity (Idle). A neural network model is trained using network traffic features that are taken from Wireshark records, such as packet count, inter-packet time, timestamps, and packet length. Analyzing network traffic in virtual reality settings provides a fresh perspective on user behavior. This study investigates the viability of employing neural networks trained on Wireshark-captured network traffic data to recognize VR gaming actions. The test set's final results, difficulties encountered, and suggested enhancements are compiled in this final report.


### **Objectives**
The primary objectives of this work were:

1. Dataset Development: Creating a trainable dataset of VR network traffic data manually labeled with corresponding user activities.

2. Neural Network Training: Training models such as Recurrent Neural Networks (RNNs) to classify VR activities.

3. Performance Analysis: Using metrics like accuracy, precision, recall, and F1-score to evaluate the effectiveness of the proposed model.

### **Methodology**

**Data Collection:**

While engaging in specific game activities, Wireshark was used to record network traffic data on a Meta Quest Pro VR headset. In order to remove outside interference from multiplayer interactions, the training mode of the game was selected. Among the activities were:

- **Walking**
- **Talking**
- **Ball Throwing**
- **Paused State**
- **No Activity(Idle State)**

Packet Capture (PCAPNG) files were used to capture network traffic, and these files were later converted to CSV format. Every gaming second represents a data point or, a sample, with features manually retrieved and categorized.

**Feature Extraction:**
UDP (User Datagram Protocol) packets, which are frequently utilized for time-sensitive applications like gaming, were used to derive the following characteristics:

- **Time (Seconds):** Recorded time for keeping track of starting and ending of an activity.
- **No. of Bits:** The sum or, total number of bits sent over the period.
- **No. of Packets:** The sum or, total number of packets sent or received during a specific period of time.
- **Average Packet Length (Bytes):** The average packet size during a certain period of time. Average Packet Length is calculated as the mean of all packet lengths in a given interval (mean = sum(packet lengths) / total packets).
- **Average Inter-Packet Arrival Time:** The average amount of time that passes between successive packets. Average Inter-Packet Arrival Time is derived by calculating the mean of these recorded inter-packet times over the specified interval (mean = sum(inter-packet times) / total inter-packet intervals).

Data is divided into 1-second windows (e.g., 12:30:45.000000 to 12:30:46.000000) with the objective of treating each interval as a single data point.At one-second intervals, each feature was aggregated (averaged). During gameplay, this structured dataset was manually labeled and saved for analysis.

## **Dataset**
The processed dataset consists of structured rows where each row represents a single second of gameplay:

| Time (Seconds) | No. of Packets | No. of Bits | Activity      | Avg. Packet Length (Bytes) | Avg. Inter-packet Arrival Time |
|----------------|----------------|-------------|---------------|-----------------------------|--------------------------------|
| 97             | 70             | 113840      | Walking       | 203.29                     | 0.0133                         |
| 98             | 72             | 118064      | Walking       | 204.97                     | 0.0127                         |
| 99             | 69             | 119128      | Walking       | 215.81                     | 0.0134                         |
| 100            | 70             | 116816      | No Activity   | 208.60                     | 0.0131                         |
| 163            | 70             | 110608      | No Activity   | 194.19                     | 0.0110                         |
| 164            | 71             | 108120      | Talking       | 190.35                     | 0.0125                         |
| 165            | 69             | 126520      | Talking       | 229.20                     | 0.0126                         |
| 166            | 71             | 135008      | Talking       | 237.69                     | 0.0135                         |

**Details**
- Total Samples: 5,781

**Splits:**

- Training: 5,348 samples (92.5% of total data, including validation).

- Validation: 1,069 samples (18.5% of total data, extracted from training).

- Test: 433 samples (7.5% of total data).

**Test Set Class Distribution:**

**Activity	Proportion:**
- No Activity	65.4%
- Talking	10.4%
- Ball Throwing	8.8%
- Walking	7.9%
- Paused	7.6%

**Key Differences from Training/Validation:**
1. Extreme imbalance exists in the test set; walking and pausing are relatively rare, whereas no activity predominates at 65.4%.
2. The test set represents raw real-world distributions, whereas the training data consists of synthetic SMOTE-augmented samples for minority groups.
3. Gameplay sessions captured during model training are included in the test data, which introduces hidden network patterns (such as variations in server latency).

### **Model Development**

**Preprocessing:**
1. Scaling: For numerical stability, features were standardized using StandardScaler.
2. Encoding: To ensure model compatibility, activity labels were encoded.
3. Balancing: SMOTE (Synthetic Minority Oversampling Technique) wasc used to address class imbalance.


**Neural Network Architecture:**
TensorFlow/Keras was used to create a Recurrent Neural Network (RNN). Key characteristics:


- **Input Layer:** Scaled features are processed by the input layer.
- **Hidden Layers:** Enhanced through hyperparameter tuning using the Hyperband method of the Keras Tuner.
- **Output Layer:** Uses a softmax activation function to predict activity types.


**Tuning Hyperparameters:**
The hyperparameters listed below were optimized:

- Number of hidden units: 64–256
- Learning rate: 1e-3–1e-4
- Dropout rate: 0.2–0.5




### **Test Results**

**Evaluation Metrics**
- **Test Accuracy**: The average of the overall accuracy.
- **Macro F1**: Unweighted average of F1-scores across all classes.  
- **Weighted F1**: F1-score averaged by class support (accounts for imbalance).  

| Metric             | Value   |
|--------------------|---------|
| Test Accuracy      | 77.37%  |
| Macro F1-Score     | 69%     |
| Weighted F1-Score  | 77%     |



**Class-Specific Performance**

| Activity        | Precision | Recall | F1-Score |
|-----------------|-----------|--------|----------|
| **Ball Throwing** | 0.79      | 1.00   | 0.88     |
| **No Activity**   | 0.89      | 0.79   | 0.84     |
| **Paused**        | 0.75      | 1.00   | 0.86     |
| **Talking**       | 0.52      | 0.71   | 0.60     |
| **Walking**       | 0.31      | 0.24   | 0.27     |

**Confusion Matrix**


![download (2)](https://github.com/user-attachments/assets/af1b9642-524c-45de-b979-9c7bcf2fc2df)







### **Challenges Encountered**
1. **Unbalanced Data:**
There were notable class disparities in the sample, with "No Activity" and "Paused State" being overrepresented. Due to a lack of samples, minority classes such as "Talking" and "Ball Throwing" produced biased model predictions.

Solution: SMOTE was used as a solution to artificially oversample minority classes. Although this lessened imbalance, it also brought about computational difficulties and sporadic collapses in validation accuracy.

2. **Feature Engineering:**
It was not possible to immediately evaluate features obtained from network traffic. While standardization guaranteed stability, it is still difficult to find more domain-specific characteristics.

Future work: To find latent correlations, investigate sophisticated feature transformations and statistical analyses.

3. **Overfitting:**
Early models had lower test accuracy because they had trouble generalizing after doing well on training data.

Solution: To lessen overfitting, regularization strategies like dropout were implemented and model complexity was decreased.

4. **Computational Overheads:**
Extensive computational resources were needed for hyperparameter adjustment with the Keras Tuner, which delayed testing.

Solution: For quicker iterations, use GPU acceleration and a constrained hyperparameter search space.

 
5. **Preprocessing Consistency:** Errors occasionally occurred as a result of different preprocessing procedures used for training and testing datasets.

Solution: Automated validation tests and standardized preprocessing methods were the solutions.


### **Conclusion**
This experiment shows how network traffic analysis may be used to identify VR activities. The findings demonstrate the viability of categorizing activities according to traffic characteristics, notwithstanding ongoing issues with data imbalance and model generalization. Additional study may strengthen the model's resilience and expand its use in other VR scenarios.onclusion
This experiment shows how network traffic analysis may be used to identify VR activities. The findings demonstrate the viability of categorizing activities according to traffic characteristics, notwithstanding ongoing issues with data imbalance and model generalization. Additional study may strengthen the model's resilience and expand its use in other VR scenarios.






## Contact
For questions or confusions, please contact smuhaimi@nd.edu
