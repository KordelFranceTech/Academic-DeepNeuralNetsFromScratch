# Academic-DeepNeuralNetsFromScratch
A framework that constructs deep neural networks, autoencoders, logistic regressors, and linear networks without the use of any outside machine learning libraries - all from scratch.

This project was constructed for the Introduction to Machine Learning course, class 605.649 section 84 at Johns Hopkins
University. FranceLab4 is a machine learning toolkit that implements several algorithms for classification and 
regression tasks. Specifically, the toolkit coordinates a linear network, a logistic regressor, an autoencoder, and a 
neural network that implements backpropagation; it also leverages data structures built in the preceding labs. 
FranceLab4 is a software module written in Python 3.7 that facilitates such algorithms.

##Notes for Graders
All files of concern for this project (with the exception of `main.py`) may be found in the `Linear_Network`,
`Logistic_Regression`, and `Neural_Network` folders. 
I kept most of my files from Projects 1, 2, and 3 because I ended up using cross validation, encoding, and other helper 
methods. However, these three folders contains the neural network algorithms of interest.

I have created blocks of code for you to test and run each algorithm if you choose to do so. In `__main__.py` scroll
to the bottom and find the `main` function. Simply comment or uncomment blocks of code to test if desired.

Each neural network and autoencoder constructed are sub-classed / inherited from the NeuralNet class in `neural_net.py`.
I simply initialize the class differently in order to construct an autoencoder, a feed-forward neural network, or
a combination of both.

Data produced in my paper were run with KFCV. However within the `main` program, you may notice that the number of 
folds `k` has been reduced to 2 to make the analysis quicker and the console output easier to follow.

The construction of a linear network begins on line 84 in `__main__.py`.

The construction of a logistic regressor begins on line 102 in `__main__.py`.

The construction of an autoencoder only begins on line 128 in `__main__.py`.

The construction of a feed-forward neural network only begins on line 141 in `__main__.py`.

The construction of an autoencoder that is trained, the decoder removed, and the encoder attached to a new hidden layer 
with a prediction layer attached to form a new neural network begins on line 221 in `__main__.py`.

The code for the weight updates and backward and forward propagation may be found in the following files within the 
`Neural_Network` folder:
- `layer.py`
- `optimizer_function.py`
- `neural_net.py`
  

`__main__.py` is the driver behind importing the dataset, cleaning the data, coordinating KFCV, and initializing each
of the neural network algorithms.


## Running FranceLab4
1. **Ensure Python 3.7 is installed on your computer.**
2. **Navigate to the Lab4 directory.** For example, `cd User\Documents\PythonProjects\FranceLab4`.
Do NOT `cd` into the `Lab4` module.
3. **Run the program as a module: `python3 -m Lab4`.**
4. **Input and output files ar located in the `io_files` subdirectory.** 


### FranceLab4 Usage

```commandline
usage: python3 -m Lab4
```


