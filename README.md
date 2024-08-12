# SynapNet_ApplicationOnGripper

Application of the SynapNet algorithm to classify a wide range of objects incrementally in a real-time dynamic environment using a soft pneumatic gripper equipped with two flex sensors and two force sensors. More details about the application are provided in the paper "".

## Framework
![](https://github.com/nilay121/SynapNet_ApplicationOnGripper/blob/main/synapnet_gif.gif)

## Dataset

The dataset used for training the SynapNet algorithm offline is provided in the "dataset" folder. The new unseen objects are :
- ethernet_adapter
- mouse
- screwdriver
- univ_adapter
- metallic_pipe
- small_beaker
- battery
- pendrive
- silicon_block
- bottle_cap

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
  - Make sure the Arduino board of the control box and the sensors are connected to the proper ports
  - Train the SynapNet algorithm on the offline data incrementally
    ```bash
    python3 main.py --Uk_classExpPhase False --pseudo_exp False
    ```
  - To perform real-time dynamic training on new unseen objects
    ```bash
    python main.py --Uk_classExpPhase True --pseudo_exp False
    ```
  - To perform a pseudo-real-time experiment
    ```bash
    python main.py --Uk_classExpPhase True --pseudo_exp True
    ```
  
## To cite the paper
  ```bash
  ```
