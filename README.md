# Lane Extraction from Trajectories at Road Intersections Based on Graph Transformer Network

This is the implementation for the paper "Lane Extraction from Trajectories at Road Intersections Based on Graph Transformer Network".



## Data

The following data are provided in the 'Data' folder: 

- \`traj.csv\`: The trajectory data used for model training and testing. Each point consist of 'trajectory_id', 'timestamp', 'longitude(x)', 'latitude(y)', 'azimuth', and 'inter_id' attributes.

- \`trueLane.csv\`: The ground truth lanes for model training and testing. Each lane includes 'geometry'  and 'inter_id' attributes.



## Codes

This repository contains the following Python codes：

- \`data_processing.py\`: Contains the implementation of data processing and feature extraction. It includes functions related to trajectory data processing, trajectory feature extraction, and the calculation of graph node and edge features.
- \`run_process.py\`: Contains the code for executing data processing and feature extraction.
- \`GTN.py\`: Contains the implementation of Graph Transformer Network for lane extraction. It includes classes and functions related to the architecture of the Graph Transformer Network and the set-based lane extraction loss calculation.
- \`train_GTN.py\`: Contains the code for Graph Transformer Network model training.
- \`test_GTN.py\`: Contains the code for model inference and lane extraction.



## Running the Code

#### Data processing and feature extraction

```bash
python run_process.py
```

This step processes trajectory data, extracts graph node features and edge features, and saves them as CSV files in the \`processed_data\` folder.

#### Model training

```bash
python train_GTN.py
```

This step trains the GTN model. The trained model is saved in \`model/GTN_model.pth\`

#### Model inference and lane extraction

```bash
python test_GTN.py
```

This step performs model inference and extracts lanes for the test intersections. The lane extraction result is saved in \`result/predicted_lane.csv`.



## Requirements

The codes use the following dependencies with Python 3.11

* networkx==3.2.1 
* pytorch==2.0.1
* torch-geometric==2.5.3 
* geopandas==1.0.1
