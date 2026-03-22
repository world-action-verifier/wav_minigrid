<div align="center">
<h2><center>👉 World Action Verifier: Self-Improving World Models via Asymmetric Forward-Inverse Consistency </h2>

[Yuejiang Liu](https://sites.google.com/view/yuejiangliu)<sup>1</sup>, 
[Lingjing Kong](https://scholar.google.com/citations?hl=en&view_op=search_authors&mauthors=Lingjing+Kong+Carnegie+Mellon)<sup>2</sup>, 
[Fan Feng](https://scholar.google.com/citations?hl=en&view_op=search_authors&mauthors=Fan+Feng+UC+San+Diego)<sup>3</sup>, 
[Weifeng Lu](https://scholar.google.com/citations?hl=en&view_op=search_authors&mauthors=Weifeng+Lu+Shanghai+Jiao+Tong)<sup>4</sup>, 
[Jinzhou Tang](https://scholar.google.com/citations?hl=en&view_op=search_authors&mauthors=Jinzhou+Tang+UC+San+Diego)<sup>3</sup>,  
[XiangCheng Zhang](https://scholar.google.com/citations?hl=en&view_op=search_authors&mauthors=XiangCheng+Zhang+Tsinghua)<sup>5</sup>, 
[Kun Zhang](https://scholar.google.com/citations?hl=en&view_op=search_authors&mauthors=Kun+Zhang+Carnegie+Mellon)<sup>2</sup>, 
[Kevin Patrick Murphy](https://www.cs.ubc.ca/~murphyk/)<sup>6</sup>, 
[Yilun Du](https://yilundu.github.io/)<sup>7</sup>, 
[Chelsea Finn](https://ai.stanford.edu/~cbfinn/)<sup>1</sup>

<br>

<sup>1</sup> Stanford University &nbsp;&nbsp;
<sup>2</sup> Carnegie Mellon University &nbsp;&nbsp;
<sup>3</sup> University of California, San Diego &nbsp;&nbsp;
<br>
<sup>4</sup> School of AI, Shanghai Jiao Tong University  
<sup>5</sup> Tsinghua University &nbsp;&nbsp;
<sup>6</sup> Google DeepMind &nbsp;&nbsp;
<sup>7</sup> Harvard University


<a href='https://arxiv.org/abs/2510.10125'><img src='https://img.shields.io/badge/ArXiv-2510.10125-red'></a> 
<a href='https://ctrl-world.github.io/'><img src='https://img.shields.io/badge/Project-Page-Blue'></a> 

</div>

This repository contains the official PyTorch implementation of [**WAV**](https://sites.google.com/view/ctrl-world).

This codebase focuses on the controlled setting experiments in MiniGrid, 
which are designed to provide mechanistic insights and validate the core 
ideas behind WAV under well-controlled environments.

**TL;DR:** WAV improves world models by turning hard prediction into easier verification via forward–inverse consistency, achieving better performance with half the data.

<p>
    <img src="assets/teaser.png" alt="WAV framework" width="100%" />
</p>

## 📋 Content

This repository contains the code for evaluating:

* Robustness under distribution shift
* Sample efficiency comparison
* Robustness under increasing state complexity
* Robustness under environmental noise
* Active acquisition strategy comparison

## 🛠️ Installation 

We recommend using a clean conda environment.

```bash
# create environment
conda create -n wav_minigrid python=3.7
conda activate wav_minigrid

# clone repository
git clone https://github.com/Weifeng2829/WAV-MiniGrid.git
cd WAV-MiniGrid

# install dependencies
pip install -r requirements.txt
pip install -e .
```
## 🎮 MiniGrid Tasks 

We evaluate WAV on three complex tasks in MiniGrid designed to test long-horizon dependencies and compositional logic. Each task requires precise manipulation of objects (Key, Ball, Box) based on their color attributes.

* **Key Delivery**: A multi-stage manipulation task: Match Key color to Box → Insert Key into Box → Swap Box with Ball → Match Ball color to Box → Reach Goal.
* **Ball Delivery**: A structural mirror of Key Delivery: Place the Ball into the Box first, then manipulate the Key according to color constraints before reaching the goal.
* **Object Matching**: A coordination challenge: Synchronize both Key and Ball colors with a reference Box, then arrange all objects together before exiting.

<p align="center">
  <table>
    <tr>
      <td align="center">
        <img src="assets/key_delivery.gif" width="90%" />
        <br>
        <b>(a) Key Delivery</b>
      </td>
      <td align="center">
        <img src="assets/ball_delivery.gif" width="90%" />
        <br>
        <b>(b) Ball Delivery</b>
      </td>
      <td align="center">
        <img src="assets/object_matching.gif" width="90%" />
        <br>
        <b>(c) Object Matching</b>
      </td>
      <td align="center">
        <img src="assets/noise_emptyenv.gif" width="90%" />
        <br>
        <b>(d) Random Play</b>
      </td>
    </tr>
  </table>
</p>

---

### 🧪 Controlled Environment: Random Play in EmptyEnv

To enable controlled mechanistic analysis, we introduce a **random play setting in an EmptyEnv**.  
In this environment, the agent interacts with:

- A variable number of **objects** (Key, Ball, Box)
- A configurable number of **noisy floor tiles**, whose colors **randomly change at every step**

By systematically varying the number of objects and noisy tiles, we can precisely control **state complexity** and **environmental stochasticity**, providing a clean testbed for studying robustness, generalization, and exploration behavior of world models.

---

## 📷 CheckPoints and Datasets 

We provide all pre-trained checkpoints and datasets used in the MiniGrid experiments, which are sufficient to reproduce all results without additional training or data collection.

For reference, we also include the code for data collection and model training:
* **Data Collection**: Code for collecting expert and random playing demonstrations is available in `env/data_collection/`
* **Model Training**: Training scripts are located in `scripts/train/`

**Note**: These scripts are optional and not required to run the experiments, as all necessary checkpoints and datasets are already provided in this repository.

## 📊 MiniGrid Experiments 

### 1. Robustness of Sparse IDM

Evaluate the robustness of Sparse IDM compared to vanilla IDM under distribution shifts.

```bash
python exps/idm_comparison.py
```

**This experiment evaluates:**

* Generalization under unseen action-object compositions
* Performance gap between Sparse IDM and vanilla IDM

---

### 2. Sample Efficiency: Sparse IDM vs. World Model

Compare data efficiency between Sparse IDM and a forward world model.

```bash
python exps/data_efficiency_gap.py
```

**This experiment evaluates:**

* Accuracy vs. training data size
* Relative performance when labeled data is limited

---

### 3. Robustness under Increasing State Complexity

Test model robustness as the environment state complexity increases.

```bash
python exps/state_complexity_gap.py
```

**This experiment evaluates:**

* Scaling behavior under larger state spaces
* Sensitivity to state complexity

---

### 4. Robustness under Increasing State Complexity

Evaluate model robustness under increasing levels of observation noise.

```bash
python exps/noise_robustness.py
```

**This experiment evaluates:**

* Robustness to stochastic perturbations in the environment
* Performance degradation under increasing noise levels

---

### 5. Active Learning Strategy Comparison

Compare acquisition strategies for improving world model performance.

```bash
python exps/wm_active_learning.py
```

**This experiment evaluates:**

* Different data acquisition strategies
* Performance gains under limited labeling budget

---

### Expected Results

Running the experiments should yield results consistent with the figures below, demonstrating WAV's superior performance across metrics:

<p>
    <img src="assets/minigrid_all_results.png" alt="MiniGrid Experimental Results" width="100%" />
</p>

Key Findings:
* **Robustness**: Sparse IDM maintains better generalization under distribution shifts compared to vanilla IDM
* **Sample Efficiency**: Sparse IDM achieves comparable or better performance with significantly less training data than forward world models
* **Complexity Scaling**: Sparse IDM demonstrate stable performance as environment complexity increases
* **Noise Robustness**: Sparse IDM remains robust under increasing environmental noise
* **Active Learning**: WAV's acquisition strategy leads to more efficient data utilization

## 📂 Project Structure

```
WAV-MiniGrid/
├── README.md
├── assets
├── checkpoints
├── data
├── env
│   ├── data_collection
├── exps
│   ├── data_efficiency_gap.py
│   ├── idm_comparison.py
│   ├── state_complexity_gap.py
│   ├── wm_active_learning.py
│   ├── train
├── requirements.txt
├── setup.py
└── src
    ├── wav_minigrid
    │   ├── models
```

---
## Acknowledgement

This codebase is built upon [MiniGrid](https://github.com/Farama-Foundation/MiniGrid) and adapted from [RIDE](https://github.com/facebookresearch/impact-driven-exploration). We thank the authors for their open-source contributions.


## Bibtex 
If you find our work helpful, please leave us a star and cite our paper. Thank you!
```
@article{liu2026selfimproving,
  title={Self-Improving World Models via Asymmetric Forward-Inverse Consistency},
  author={Liu, Yuejiang and Kong, Lingjing and Feng, Fan and Lu, Weifeng and Tang, Jinzhou and Zhang, XiangCheng and Zhang, Kun and Murphy, Kevin Patrick and Du, Yilun and Finn, Chelsea},
  journal={arXiv preprint arXiv:2602.XXXXX},
  year={2026}
}
```