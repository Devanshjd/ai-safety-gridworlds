# ai-safety-gridworlds
DeepMind-inspired AI Safety Gridworlds implementation exploring safe interruptibility, side effects, and reward gaming.

# 🧠 AI Safety Gridworlds (DeepMind Inspired)

Implementation of AI Safety Gridworld environments inspired by DeepMind’s AI Safety research.  
This project explores how reinforcement learning agents can develop **unsafe behaviours** such as reward hacking, side effects, and failure under interruption.

---

## 🚀 What This Project Covers

This repository includes environments and experiments for:

- ✅ **Safe Interruptibility**  
- ✅ **Avoiding Side Effects**  
- ✅ **Reward Gaming (Reward Hacking)**  

Each environment is designed to test and demonstrate how an agent’s policy can become unsafe if incentives are misaligned.

---

## 🧩 Key Learning Outcomes

- Built gridworld environments for safety-focused RL evaluation
- Implemented agents and policies to observe behaviour under constraints
- Analysed unsafe patterns like reward exploitation and side-effect negligence
- Produced simulation outputs and evaluation plots

---

## 🛠 Tech Stack

- **Python**
- **NumPy**
- **Matplotlib**
- *(Optional)* Gymnasium / custom environment loop

---

## 📁 Repository Structure
ai-safety-gridworlds/
├── src/ # Environment + agent code
├── notebooks/ # Experiments and analysis
├── results/ # Logs / output tables
├── images/ # Plots (heatmaps, paths, reward curves)
├── requirements.txt
└── README.md


---

## ▶️ How to Run

### 1) Install dependencies
pip install -r requirements.txt
2) Run experiments (example)
python src/run_experiments.py

If your entry file has a different name (e.g., main.py or a notebook), update the command accordingly.
