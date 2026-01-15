
# <img src="logo.png" alt="ToolSafe Logo" height="40" style="vertical-align: middle; margin-right: 10px;"> ToolSafe: Enhancing Tool Invocation Safety of LLM-based agents via Proactive Step-level Guardrail and Feedback

<div align="center">

**Official Implementation of "ToolSafe: Enhancing Tool Invocation Safety of LLM-based Agents via Proactive Step-level Guardrail and Feedback"**

[![arXiv](https://img.shields.io/badge/arXiv-23XX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/23XX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

</div>

---

## 📖 Introduction

**ToolSafe** is a framework designed to enhance the safety of LLM-based agents during tool invocation. It introduces a proactive step-level guardrail and feedback mechanism to prevent unsafe tool usage.

This repository contains:
- **TS-Guard**: The core safety guardrail model/mechanism.
- **TS-Bench**: A comprehensive benchmark for evaluating tool safety.
- Training and evaluation scripts.

## 🔥 News
* **[2026-01-15]** 🚀 The official code and dataset for ToolSafe are released!
* **[202X-XX-XX]** Paper accepted to [Conference Name].

## 📂 Repository Structure

```text
.
├── TS-Bench/            # Benchmark datasets and definitions
├── TS-Guard/            # Core implementation of the Guardrail model
├── benchmark/           # Evaluation scripts and metrics
├── scripts/             # Shell scripts for training/inference
├── src/                 # Source code for the agent framework
├── utils/               # Utility functions
├── pyproject.toml       # Python project dependencies
├── submit_task.sh       # Script for submitting jobs/tasks
└── README.md
```

##  🛠️ Installation
### Prerequisites
- Python >= 3.9
- PyTorch (Please refer to PyTorch.org for your specific CUDA version)

### Setup
This project uses pyproject.toml for dependency management.

# 1. Clone the repository
git clone https://github.com/MurrayTom/ToolSafe.git
cd ToolSafe

# 2. Create a virtual environment (Recommended)
conda create -n toolsafe python=3.9
conda activate toolsafe

# 3. Install dependencies
pip install -e .


## Citation
If you find ToolSafe useful for your research and applications, please cite using this BibTeX:
@article{toolsafe2026,
  title={ToolSafe: Enhancing Tool Invocation Safety of LLM-based Agents via Proactive Step-level Guardrail and Feedback},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:23XX.XXXXX},
  year={2026}
}
