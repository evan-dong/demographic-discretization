## Background

This is a general implementation of the methods used in [Addressing Discretization-Induced Bias in Demographic Prediction](https://arxiv.org/abs/2405.16762). Reference the paper for an explanation of the discretization methods and results when applied to specific case study datasets. The purpose of this code repository is to give other practitioners and academics a starting point to carry out similar analyses on their own tasks and model outputs.

Note that this is *not* a package to be imported. This code may not be regularly maintained, and prioritizes readability and simplicity over optimized performance.

## File outline

`run.ipynb` provides an example notebook of how to apply different discretizations to an input dataset of probabilities (with optional ground truth labels and conditional variables of interest).

The other files - `basic_decision_functions.py`, `metrics.py`, and `utils.py` - contain helper functions to discretize probabilities, calculate accuracy and fidelity, and miscellaneous helper functions respectively.
The file `additional_decision_functions.py` primarily contains the integer program decision functions, as well some customizable versions of functions in `basic_decision_functions.py` that are unused in `run.ipynb`.

## Setting Up

The basic codebook is intended to require only standard data science packages and Python 3.11. The list of virtual environment packages is listed in `basic_requirements.txt`.

To make use of the integer program discretization methods, use the `additional_requirements.txt` requirements file instead. The additional package requirements primarily stem from `cvxpy` and `gurobipy`, although substituting Gurobi with another MILP solver package may be possible.

## Running the Notebook

The `run.ipynb` notebook is designed to work with any input dataset of model output probabilities.

The three `TODOs` commented in codebook **cells 2-4** highlight the dataset information and function needed to load your own dataset for use.

### Parameters

Only the capitalized variables in `run.ipynb` defined in **cells 2-7** need to be modified to adjust running parameters. This includes data parameters, which kinds of discretization methods should be run, and where discretization outputs should be saved.

### Outputs

The tables produced in the Assessment section of `run.ipynb` provide summary statistics, population makeup (including bias), and prediction error rates when using discretization methods. Note that not all tables are calculable without ground truth labels available.

### Additional Decision Functions

To run integer program-based discretizations, install the requirements in `additional_requirements.txt` and uncomment the corresponding lines in cells 1 and 7 (`import * from additional_decision_functions` and `Integer Program`).

### Further Customizaton

Other discretization methods are possible and the code is easily extensible by modifying `basic_decision_functions.py` and adding the appropriate methods to `run.ipnyb`.

Note that any functions defined in a jupyter codebook cannot be found and run using the multiprocessing module used to speed up `batched_matching_discretization` and `batched_integer_program_discretization`, so such functions *must* go in `basic_decision_functions.py` or `additional_decision_functions.py`.
