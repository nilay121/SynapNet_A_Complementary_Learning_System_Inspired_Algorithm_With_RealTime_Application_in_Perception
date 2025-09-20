# SynapNet: A Complementary Learning System Inspired Algorithm With Real-Time Application in Multimodal Perception

Catastrophic forgetting is a phenomenon in which a neural network, upon learning a new task, struggles to maintain it's performance on previously learned tasks. It is a common
challenge in the realm of continual learning (CL) through neural networks. The mammalian brain addresses catastrophic
forgetting by consolidating memories in different parts of the
brain, involving the hippocampus and the neocortex. Taking
inspiration from this brain strategy, we present a CL framework
that combines a plastic model simulating the fast learning
capabilities of the hippocampus and a stable model representing
the slow consolidation nature of the neocortex. To supplement
this, we introduce a variational autoencoder (VAE)-based pseudo
memory for rehearsal purposes. In addition by applying lateral
inhibition masks on the gradients of the convolutional layer,
we aim at damping the activity of adjacent neurons and intro-
duce a sleep phase to reorganize the learned representations.
Empirical evaluation demonstrates the positive impact of such
additions on the performance of our proposed framework; we
evaluate the proposed model on several class-incremental and
domain-incremental datasets and compare it with the standard
benchmark algorithms, showing significant improvements. With
the aim to showcase practical applicability, we implement the
algorithm in a physical environment for object classification
using a soft pneumatic gripper. The algorithm learns new classes
incrementally in real time and also exhibits significant backward
knowledge transfer (KT).

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
@ARTICLE{10649896,
  author={Kushawaha, Nilay and Fruzzetti, Lorenzo and Donato, Enrico and Falotico, Egidio},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={SynapNet: A Complementary Learning System Inspired Algorithm With Real-Time Application in Multimodal Perception}, 
  year={2024},
  volume={},
  number={},
  pages={1-15},
  keywords={Task analysis;Training;Heuristic algorithms;Data models;Brain modeling;Real-time systems;Feature extraction;Catastrophic forgetting;complementary learning system (CLS);continual learning (CL);perception;pseudo episodic memory;soft gripper;variational autoencoder (VAE)},
  doi={10.1109/TNNLS.2024.3446171}}
  ```
