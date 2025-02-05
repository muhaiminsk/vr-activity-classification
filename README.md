# VR Activity Classification
Identifying user activities in a VR game using neural networks trained on network traffic data collected via Wireshark. The project involves analyzing features like packet count, inter-packet time, timestamps, and packet length.

## Methodology:
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


## Contact
For questions or confusions, please contact smuhaimi@nd.edu
