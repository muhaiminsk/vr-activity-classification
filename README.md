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

## Contact
For questions or confusions, please contact smuhaimi@nd.edu
