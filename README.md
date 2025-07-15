# Federated Learning for 5G Base Station Traffic Forecasting

This project implements a federated learning approach for forecasting traffic in 5G base stations. It uses data from three base stations (ElBorn, LesCorts, and PobleSec) to predict key metrics like downlink/uplink traffic, RNTI count, and resource block usage.

## Project Overview

The project demonstrates how federated learning can be applied to time series forecasting in telecommunications, comparing it with traditional centralized and individual training approaches.

### Key Features

1. **Data Preparation and Cleaning**
   - Loading and preprocessing time series data from 3 base stations
   - Handling missing values and outliers
   - Temporal train/validation/test splitting
   - Global and local scaling options

2. **Time Series Transformation**
   - Sliding window approach for sequence generation
   - Multi-step forecasting of 5 target variables

3. **Model Architectures**
   - MLP (Multi-Layer Perceptron)
   - RNN (Recurrent Neural Network)
   - LSTM (Long Short-Term Memory)
   - GRU (Gated Recurrent Unit)
   - CNN (Convolutional Neural Network)

4. **Training Settings**
   - Individual Training (separate model per base station)
   - Centralized Training (single model on combined data)
   - Federated Learning (FedAvg algorithm)

5. **Evaluation**
   - MAE and RMSE for all targets
   - NRMSE for DownLink and UpLink
   - Per-station and global performance metrics

6. **Advanced Analyses**
   - Preprocessing impact analysis
   - Aggregation algorithm comparison (FedAvg, SimpleAvg, MedianAvg, FedProx, FedAvgM, FedNova, FedAdagrad, FedYogi, FedAdam)
   - Local fine-tuning after federated learning
   - Carbon & energy consumption analysis
   - Scalability experiments with synthetic base stations

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Usage

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook: `jupyter notebook sample.ipynb`

## Dataset

The dataset contains LTE measurements from 3 base stations with the following features:
- Timestamps (at 2-minute intervals)
- Base station identifier (ElBorn, LesCorts, PobleSec)
- 11 input features per timestep including:
  - Downlink/Uplink traffic
  - RNTI count (connected devices)
  - MCS (Modulation and Coding Scheme) metrics
  - Resource Block (RB) metrics

## Results

The notebook compares the performance of different training approaches and model architectures, demonstrating:
- How federated learning compares to centralized and individual training
- The impact of different preprocessing strategies
- The effectiveness of various aggregation algorithms (including advanced methods like FedProx, FedAdam, etc.)
- The benefits of local fine-tuning after federated learning
- Energy consumption and carbon footprint comparison between federated and centralized approaches
- Scalability analysis with increasing numbers of base stations

## Future Work

- Implement personalized federated learning approaches
- Explore differential privacy techniques for enhanced privacy guarantees
- Investigate asynchronous federated learning for more realistic network conditions
- Extend to more complex model architectures (e.g., Transformers, Graph Neural Networks)
- Develop adaptive client sampling strategies based on data quality and network conditions
