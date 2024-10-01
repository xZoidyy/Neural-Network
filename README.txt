I have written a code for Neural Network - NN (classification problem), which I constructed by book from Christopher M. Bishop (https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)

It is needed to unzip the library file

1) PARAMETERS OF NN

In directory "inputs" there is file input.txt
There are defined basic parameters that dictates how the NN should look like

INPUT SIZE: How many input variables NN takes + 1 (because there is also a bias term).
	    The format of input looks like this: (1, x1, x2, ..., xN) so the input size = N + 1
	    
NUMBER OF NEURONS: This is a vector of numbers that characterise number of neurons in each hidden layer. So the length of this vector define number of Hidden layers
		   The format: 10 5 5 (I got 3 hidden layers and number of neurons in each of it is 10, 5 and 5)
	
OUTPUT SIZE: For now is only available 1. One output for classification problem

LEARNING RATE: It is a number how much I change the weights of NN. Lower number means more precise steps but It takes longer time to converge 

ALPHA VALUE: It is value from which I can calculate Hessian and Inverse Hessian of weights using Outer product approximation. Also I use this value to set initial alpha regularisation term.

NUM: It is a value where learning and upgrade of weights stops if it takes much time (It is only a relic of pre-versions and learning should not stop by this condition -> but It is for safety)

TRAINING / PREDICTING: 0 - for training NN and its weights, 1 - for predicting, but It needs pre-trained weights
		       If 1 is set but there are no weights (or something for prediction is missing) then training will automatically starts and the it needs to be start again
		       Pre-trained values are saved in files "weights.txt" and "alpha.txt"

2) INPUTS & OUTPUTS DATA

In the directory "datas".
There is a file "training_data.txt". This is the name for training set of inputs (first value is always 1! - bias). Last value is target value.
There is should be also a file "new_data.txt" where you can predict on new data set. (It is without last target value)

NN will create here also files for plotting the decision boundary for example.
File "boundary_decition_trained_data.txt" and "predicted_trained_data.txt" there is an information about prediction on training set.
File "boundary_decition_data.txt" and "predicted_data.txt" there is an information about prediction on new data set.

3) RUNNING THE CODE

For compilation of .cpp code (NN.cpp) there is a script in the file "makefile.sh". You need only execute this file and the code is compiled and libraries loaded.
It will create a binary file "NN" and by this you start the code

Attention: The script and the source code format is done only for Linux or Mac based systems (especially the format of paths to files)

4) PLOTTING

In the directory "source" a keep python scripts for plotting my results.



	     