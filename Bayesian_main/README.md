# Automated System for MPM

## Code files

+ **optimization.py**:  Bayesian optimization
+ **constraints.py** :  Encapsulation of hyperparameter settings
+ **model.py**:  The model of auto machine learning algorithm
+ **algo.py**:  Encapsulation of algorithms
+ **method.py**:  Automatically select the algorithm
+ **utils.py**:  Data pre-process and visualization
+ **metric.py**: Shapley value
+ **interpolation.py**: The optimization for interpolation
+ **test.py**:  The template code to run


## Preprocess
The explanation of some functions in **utils.py**:
+ **preprocess_data**：The standard function to preprocess raw data.
+ **preprocess_all_data**: Preprocess the raw data of all datasets, excluding *Washington*.
+ **preprocess_data_interpolate**: For dataset *Washington*.
+ **show_result_map**：To demonstrate the predict result.

## Algorithm  
The algorithm to predict gold mine should be encapsulated into a standard class which defines:
+ **__init__(self, params)**: Take *params* as the parameter of init function, and unpack it to the init function of super class.
+ **predicter(self, X)**: Return both 2-class-result and the probability-result of samples being classified as positive ones.

## Constraints
The constraints on hyperparameters of the algorithm, requiring:
+ **Continuous Param**: Require a floating point list length 2 as the lower and upper bound
+ **Discrete Param**: Require an integer list legnth 2 as the lower and upper bound
+ **Categorical Param**: Require a list as the enumeration of all feasible options
+ **Static Param**: Require a value as the static value

## Running  
Change *path* in **test.py** before run it.

# Bayesian Optimization in MPM

## Process of hyperparameters

### The format of hyperparameters that input, store and use in *optimization.py*.

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


## Algorithms

### The algorithms to build a model for mine prediction.

* More **encapsulated algorithms** and corresponding **default hyperparamters** in *algo.py*
    * *Random Forest    (RF)*
    * *Logistic Regression  (LGR)*
    * *Multilayer Perceptron    (MLP)*
    * *Support Vector Machine   (SVM)*

* More stable and reliable method for dataset split in *model.py*
    * (**IID**) Spilt by random-spilt strategy.
    * (**OOD**) Spilt by *K-Means* clustering algorithm with scheme to choose certain start point of generating subarea so as to cover all splitting scenarios with less trials.


## Optimization Logic

### The logic workflow of hyperparameter optimization in *optimization.py*.

* Automatically choose the best hyperpameters for the maching learning algorithm. 
* Coarse tuning on some non-sensitive hyperparameters
    * Stop tuning certain hyperparamter after a proper value is chosen, such as *solver* of *MLPClassifier*
* Fidelty on the number of trials required to alleviate randomness of dataset split
* Multi-processing on multiple threads to accelerate the predicting process

## Method selection

### The selection on different machine learning methods in *method.py*.

* Evaluate each method with several default configurations

## Interpolation optimization

### The selection on different interpolation strategies in *method.py*.

* *scipy.interpolate.interp2d* with interpolation kinds of ['linear', 'cubic', 'quintic']
* *kringing interpolation* with interpolation kinds of ["linear", "gaussian", "exponential", "hole-effect"]