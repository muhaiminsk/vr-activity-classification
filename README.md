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



## Contact
For questions or confusions, please contact smuhaimi@nd.edu
