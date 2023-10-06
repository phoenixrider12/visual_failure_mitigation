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
