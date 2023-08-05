# SynapNet_ApplicationOnGripper

Application of the SynapNet algorithm to classify a wide range of objects incrementally in a real-time dynamic environment using a soft pneumatic gripper equipped with two flex sensors and two force sensors. More details about the application are provided in the paper "".

<p align="center">
  <img src="https://github.com/nilay121/SynapNet_ApplicationOnGripper/blob/main/IMG_20230703_122715.jpg" width="350" alt="accessibility text">
</p>

## Dataset

The dataset used for training the SynapNet algorithm offline is provided in the "dataset" folder.

## Install the dependencies in a virtual environment

- Create a virtual environment (Python version 3.8.10) 
  
  ```bash
  python3 -m venv SynapNetApplication
  ```

- Activate the virtual environment
  ```bash
  . SynapNetApplication/bin/activate
  
- Install the dependencies

  ```bash
  pip3 install -r requirements.txt
  ```

## To replicate the application on a gripper
  - Make sure the Arduino board of the control box and the sensors are connected to the proper ports.
  - Train the SynapNet algorithm first on the offline data by setting ```Uk_classExpPhase = False``` in the main.py file.
  - To perform real-time dynamic training and testing set ```Uk_classExpPhase = True```
    ```bash
    python main.py
    ```
  
## To cite the paper
  ```bash
  ```
