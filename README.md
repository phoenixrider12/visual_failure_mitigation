# Visual Controller Anomalies Detection and Mitigation

## [Project Website](https://phoenixrider12.github.io/failure_mitigation) | [Paper](https://arxiv.org/pdf/2309.13475.pdf)

Here is the codebase for our paper ["Detecting and Mitigating System-Level Anomalies of Vision-Based Controllers"](https://arxiv.org/pdf/2309.13475.pdf) for an aircraft taxiing problem. We present a framework for detecting system-level failures using a trained anomaly detector and implementing a fallback mechanism for safety-critical control.

# Requirements
The code was tested with following setup:
- Ubuntu 20.04
- Python 3
- Anaconda

# Installation
```
git clone https://github.com/phoenixrider12/visual_failure_mitigation.git
cd visual_failure_mitigation
conda create -n xplane pip
conda activate xplane
pip install -r requirements.txt
```

# Dataset
We prepared a dataset with varying environmental conditions using 3 Times of Day(morning, evening, and night), 2 Cloud Conditions(clear and overcast), and 5 different Runways(KMWH, KATL, PAEI, KSFO, and KEWR). We collected 20k images for each case. The total dataset size is 60GB, and you can download it using the following steps:
```
pip install gdown
gdown https://drive.google.com/file/d/1ju_b36NQky_42wPzY5sLQuSNsgLISC8T/view?usp=drive_link
```

# Citation
If you find our work useful for your research, please cite:
```
@article{gupta2023detecting,
  title={Detecting and Mitigating System-Level Anomalies of Vision-Based Controllers},
  author={Gupta, Aryaman and Chakraborty, Kaustav and Bansal, Somil},
  journal={arXiv preprint arXiv:2309.13475},
  year={2023}
}
```
