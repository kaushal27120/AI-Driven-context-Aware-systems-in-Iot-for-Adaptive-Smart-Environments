# Smart Home Energy Management System (SHEMS) Using Deep Learning and Reinforcement Learning

## Project Overview
This project presents a **context-aware Smart Home Energy Management System (SHEMS)** that leverages deep learning and reinforcement learning to optimize energy usage, user comfort, and cost efficiency in residential environments. The framework combines predictive modeling of user activity with intelligent appliance scheduling.

---

## Key Components

### 1. Data Preparation
- The dataset (`combined_df`) contains features such as energy consumption, environmental parameters, and user activity labels.
- Sequential input data is generated for temporal models using a sliding window approach (`TIME_STEPS = 10`).

### 2. Self-Improved LSTM with Transformer (sLSTM-T)
- Predicts user behavior and energy demand.
- **Architecture**:
  - GRU/LSTM block with Layer Normalization for temporal learning.
  - Multi-Head Attention Transformer block for long-range dependencies.
  - Feed-forward layers and fusion with GRU output.
- **Output**: Predicts user activity/energy usage for the next timestep.
- **Training**: Optimized using Adam with binary cross-entropy loss and accuracy metrics.

### 3. Deep Q-Network with Puma Optimizer (DQN-sPO)
- Optimizes **appliance scheduling** in the smart home environment.
- DQN agent learns the **optimal actions** (on/off switching, energy storage management) to maximize cumulative rewards.
- Puma Optimizer (POA) is applied to dynamically balance **exploration and exploitation**, inspired by the hunting behavior of pumas.
- **Reward Function**: Incorporates energy cost, user comfort, and efficiency.
- **State**: Environment observations including battery levels, energy consumption, and appliance states.
- **Actions**: Device control, power-shifting, charging/discharging.

### 4. Evaluation Metrics
| Metric | Rule-Based | LSTM Only | Proposed DQN-sPO_sLSTM-T |
|--------|------------|-----------|--------------------------|
| Average Reward | 68.42 | 82.35 | 94.88 |
| Energy Usage (kWh) | 14.6 | 12.1 | 10.4 |
| Comfort Index | 0.83 | 0.87 | 0.93 |
| Efficiency (%) | 78.1 | 84.5 | 91.2 |
| Learning Time (s) | — | 46 | 53 |

- The proposed framework achieves **higher rewards, lower energy consumption, and improved comfort** compared to traditional methods.

### 5. Visualizations
- Training and validation performance, including **learning rate, epsilon decay, and average rewards**, are plotted for analysis.
- Comparative bar charts are generated for **Efficiency, Comfort Index, Energy Usage, and Average Reward**.

---

## Technologies Used
- **Python**, **TensorFlow/Keras**, **Matplotlib**, **NumPy**
- Deep Learning: LSTM, GRU, Transformer, sLSTM-T
- Reinforcement Learning: Deep Q-Network (DQN)
- Optimization: Puma Optimizer Algorithm (POA)

---

## Conclusion
The proposed **sLSTM-T + DQN-sPO** framework successfully predicts user activity and optimizes energy usage while maintaining comfort and efficiency. Experimental results demonstrate superior performance over conventional rule-based and LSTM-only approaches, making it a practical solution for intelligent smart home energy management.
