# 2023_Q3_DL_project: Reproducibility Learning-to-See-in-the-Dark
### Group 10:
#### David Janssen: ...</br> Jurjen Scharringa: 4708652 </br> Xavier Fung-A-Jou


## Introduction
This GitHub project is a reproducion of the Learning to See in the Dark [paper](http://cchen156.github.io/paper/18CVPR_SID.pdf) in CVPR 2018 by [Chen Chen](http://cchen156.github.io/), [Qifeng Chen](http://cqf.io/), [Jia Xu](http://pages.cs.wisc.edu/~jiaxu/), and [Vladlen Koltun](http://vladlen.info/). 

The reproduction of this paper is part of the course Deep Learning (CS4240) of Delft University of Technology. 
The goal of this project is to reproduce the results of the original paper with a new trained model using the same Neural Network structure. 
More explaination on the reproducibility paper can be found [here](link naar reproducibility paper)

Due to original paper being from 2018, a lot of packages the original code uses are no longer compatible with the current Python version. 
This has created a variation of the old code that functions with the present version of Python.
Also, instead of working with TensorFlow as done in the old code, this code uses PyTorch. 

## Setup
This project was made on windows and therefore the setup instructions will be for windows. 
First, the repository needs to be downloaded into a directory of choice:
```
git clone git@github.com:Jurjen63/2023_Q3_DL_project.git
```

### Requirements
- Python version >3.6
- Conda version >23.0.0

To install the further required packages, a requirements.txt file has been added to the project. Install these packages using the following command in the anaconda prompt: 
```
conda create --name myenv --file requirements.txt
```
Note: First go to your local directory of the project before running this line and change 'myenv' to what name you desire for this environment. 

### Dataset
Only the Sony dataset of the original paper has been used in this reproduction. This dataset can be downloaded here for [Sony](https://storage.googleapis.com/isl-datasets/SID/Sony.zip) (25 GB).
This dataset contains raw images. 
The contents of the 'long' directory are the ground truths with a long-exposed image and those of the 'short' directory are the short-exposed images which are going to be used to be enlightened. 
Furthermore, the filenames contain information, if the first digit is '0' the image is used for training, '1' is used for testing and '2' is used for validating. 

### Training
Train the Sony model by running the ```PyTorch_train_Sony.py``` file. The results and model will be saved in '/PyTorch/result_Sony' and '/PyTorch/model_Sony'. 

If you wish to use our pre-trained, use the ```checkpoint_sony_e4000.pth``` file in '/PyTorch/model_Sony' that has already been uploaded to this GitHub project. 

### Testing
To test the model run the ```PyTorch_test_Sony.pu``` file. The results will be saved in '/PyTorch/test_result_Sony'.

### Process results
To process the results, run the ```process_results.py``` file. In that file you can change the dataset variable to either 'test' or 'validate' to decide which results should be processed. 
This file processes the results that are previously made and does a ```skimage.metrics``` structural similarity and peak signal noise ratio calculation on the ground truth compared to the made results. 
These results are stored , for example when using the 'test' dataset, in '/processed_results_test.csv' so that later on analytics can be done on these results. 
