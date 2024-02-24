# AutoMPM Guide

---
## I. Operating Instructions
### Hardware Requirements
> - RAM (>=32GB) 
> - Recommended: CPU (>= 4 GHz)

### Software Requirements

This software is supported for **Linux**.
> - Ubuntu 20.04
> - The installation tool will be **Anaconda** , to install Anaconda: https://docs.anaconda.com/free/anaconda/install/linux/
### 1. Create the virtual python environment 
```
conda create --name autompm python=3.8
conda activate autompm
```

### 2. Install all the necessary packages
```
# This will generally take few minutes to install
conda install requirements.txt
```

### 3. Install a special package for preprocessing

This will generally take around 10-30 minutes to install
```
pip install --global-option=build_ext --global-option="-I/usr/include/gdal" GDAL==`gdal-config --version`
```

If the command above failed, use the commands below instead:
```

sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update
sudo apt-get update
sudo apt-get install gdal-bin
sudo apt-get install libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
pip install GDAL
```

> Detailed instruction web: 
> https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html

### 4. Run the file and check the result
Install from Github
```
git clone https://github.com/Leonardo-1107/autoMPM.git
cd autoMPM
```

#### (1) Data preprocess
High CPU performance recommanded. This will generally take 20 minutes to finnish.
```
cd Bayesian_main/dataset/Nova
python Preprocessing.py
```
Following the completion of this operation, a packed dataset, in the *.pkl* file form, will be stored within the *Bayesian_main/data* directory.

#### (2) Bayesian Optimization
It may take about 5 seconds for each optimization step on the given dataset, depending on your computer’s CPU speed.
```
# you could modify the configuration in Bayesian_main/code/test.py
python Bayesian_main/code/test.py
```

#### (3) Check the result in Bayesian_main/run
```
The result will be stored as .md files.
```

---
## II. Code Files

+ **optimization.py**:  Bayesian optimization realized
+ **constraints.py** :  Encapsulation of hyperparameter settings
+ **interpolation.py**: The optimization for interpolation
+ **method.py**: Automatically select the algorithm
+ **metric.py**: Feature Filtering realized
+ **model.py**:  The model of auto machine learning algorithm
+ **algo.py**:  Encapsulation of algorithms
+ **test.py**:  The template code to run
+ **utils.py**:  Data pre-process and visualization

---
## III. Functions 
### 1. Preprocess
The explanation of some functions in **utils.py**:
+ **preprocess_data**：The standard function to preprocess raw data.
+ **show_result_map**：To demonstrate the predict result.

### 2. Algorithm Prediction 
The algorithm to predict gold mine should be encapsulated into a standard class which defines:
+ **__init__(self, params)**: Take *params* as the parameter of init function, and unpack it to the init function of super class.
+ **predicter(self, X)**: Return both 2-class-result and the probability-result of samples being classified as positive ones.

<!-- ### Hyperparameter Constraints (Setted)
The constraints on hyperparameters of the algorithm, requiring:
+ **Continuous Param**: Require a floating point list length 2 as the lower and upper bound
+ **Discrete Param**: Require an integer list legnth 2 as the lower and upper bound
+ **Categorical Param**: Require a list as the enumeration of all feasible options
+ **Static Param**: Require a value as the static value -->

### 3. Bayesian Optimization in MPM

#### Process of hyperparameters

##### The format of hyperparameters that input, store and use in *optimization.py*.

* Change the input of hyperparamter info into a fully dict-like format, as
    * { #param_name: {
        * 'type': Enum(continuous, discrete, enum, static)
        * 'low': float or int
        * 'high': float or int
        * 'member': IntEnum(#member)
        * 'value': float or int
        * }
    * }
* A encapsulated function for checking the format of hyperparameter info
    * Whether in the params of algorithm
    * continuous and discrete: low and high
    * enum: member
    * static: value
* A encapsulated function for translating between hyperparameter info and value type
    * continuous to uniform
    * discrete and enum to randint


#### Algorithm Library

##### The algorithms to build a model for mine prediction.

* More **encapsulated algorithms** and corresponding **default hyperparamters** in *algo.py*
    * *Random Forest    (RF)*
    * *Logistic Regression  (LGR)*
    * *Multilayer Perceptron    (MLP)*
    * *Support Vector Machine   (SVM)*

* More stable and reliable method for dataset split in *model.py*
    * (**IID**) Spilt by random-spilt strategy.
    * (**OOD**) Spilt by *K-Means* clustering algorithm with scheme to choose certain start point of generating subarea so as to cover all splitting scenarios with less trials.


#### Optimization Logic

##### The logic workflow of hyperparameter optimization in *optimization.py*.

* Automatically choose the best hyperpameters for the maching learning algorithm. 
* Coarse tuning on some non-sensitive hyperparameters. Fidelity on the number of trials required to alleviate randomness of dataset split
* Multi-processing on multiple threads to accelerate the predicting process

#### Method selection

##### The selection on different machine learning methods in *method.py*.

* Evaluate each method with several default configurations

#### Interpolation optimization

##### The selection on different interpolation strategies in *method.py*.

* *scipy.interpolate.interp2d* with interpolation kinds of ['linear', 'cubic', 'quintic']
* *kringing interpolation* with interpolation kinds of ["linear", "gaussian", "exponential", "hole-effect"]
