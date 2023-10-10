# Visual Controller Failure Detection and Mitigation

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
We prepared a [dataset](https://drive.google.com/file/d/1ju_b36NQky_42wPzY5sLQuSNsgLISC8T/view) with varying environmental conditions using 3 Times of Day(morning, evening, and night), 2 Cloud Conditions(clear and overcast), and 5 different Runways(KMWH, KATL, PAEI, KSFO, and KEWR). We collected 20k images for each case. The total dataset size is 56GB, and you can download it using the following steps:
```
pip install gdown
gdown https://drive.google.com/uc?id=1ju_b36NQky_42wPzY5sLQuSNsgLISC8T
tar -zxvf taxinet_dataset.tar.gz
```
You should have a folder ```dataset``` containing 30 subfolders, whose names informs about TIME_OF_DAY, CLOUD_CONDITION, and AIRPORT_ID respectively, separated by underscore.

# Training Anomaly Detector

Prepare the dataset (need to run only once)
```
python prepare_dataset.py
```
Train classifier
```
python efficientnet_training.py
```
The prepare dataset.py file is specific for training on our proposed training dataset consisting of 3 runways(KMWH, KATL, and PAEI), 2 times of day(morning and night), and both cloud conditions, totaling 12 cases and 240k images. For training on different cases, you can modify the prepare_dataset.py file and then train the network.

# Testing Anomaly Detector
```
python efficientnet_inference.py --time 17.0 --cloud 0 --runway KMWH
```

# Fallback Mechanism Testing
In our work, the training set . Follow the below-mentioned steps for simulation testing of our framework.
- Run the X-Plane simulator and choose the desired airport.
- Run following commands in terminal
```
cd visual_controller_failure
python simulate.py --time 17.0 --cloud 0 -- runway KMWH --use_fallback True
```
You can choose desired time of day and cloud condition while running the above command and select whether you want to run the safety pipeline or the default visual controller.

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
