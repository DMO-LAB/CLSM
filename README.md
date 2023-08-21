# CLSM Repository

## Repository Structure
This repository is organized into several folders, each serving a distinct purpose in the functioning and evaluation of the CLSM.

1.  Algorithm
    
    [clsm.py](algorithm\clsm.py): Responsible for computing the weighted MSE and determining which points are won by each model.

    [algorithm\create_models.py](create_models.py): A utility to create regression models or neural network models as per the requirements.

    [algorithm\optimizers.py](optimizers.py): Houses the optimization algorithms, like Adam and Newton's method, which drive the accuracy and efficiency of our models.

2.  Example Cases

    This directory contains various test cases that the model was subjected to during the research. Each case trains the model and generates results for plotting. It is crucial to run the scripts in this folder before delving into the Make Plots folder.

3. Flame Speed Data

    This folder holds the flame speed correlation data derived from Cantera simulations. This data serves as the backbone for training the combustion example case provided in the Example Cases directory.

4. Make Plots

    After the results are generated from the Example Cases directory, the scripts in this folder can be used to visualize those results. It offers a variety of plots to better understand the efficiency and accuracy of the models for each test case.

5. Saved-folder

    A crucial aspect of our repository! Whenever you run a script from the Example Cases, the resultant models are automatically saved here. When you move to the Make Plots directory, these saved models are called upon to generate the plots. This ensures a seamless transition between training, saving, and visualizing the models.


Getting Started:

* Clone this repository to your local machine.
* Ensure you have the necessary dependencies installed.
* Navigate to the Example Cases directory and run the desired test cases.
* Once done, head over to the Make Plots directory to visualize your results.

(Optional) Explore the Algorithm directory to get a deeper understanding of the underlying methods and utilities used.
