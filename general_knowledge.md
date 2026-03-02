

# BLOG 1
https://www.datacamp.com/tutorial/tutorial-time-series-forecasting
## TIME SERIES IN GENERAL

2 types: univariate, multivarate

"
When we are dealing with multivariate time series forecasting, the input variables can be of two types:

    Exogenous: Input variables that are not influenced by other input variables and on which the output variable depends.

    Endogenous: Input variables that are influenced by other input variables and on which the output variable depends.
"

Categories in forecasting (broad):
- Classical/STatistical: MA, ARIMA, Exponential smoothing, TBATS
- ML: linear regression, xgboost, and more
- deep learning: RNN, LSTM


### Statistical:
"
#### ARIMA

ARIMA is one of the most popular classical methods for time series forecasting. It stands for autoregressive integrated moving average and is a type of model that forecasts given time series based on its own past values, that is, its own lags and the lagged forecast errors. ARIMA consists of three components:

    Autoregression (AR): refers to a model that shows a changing variable that regresses on its own lagged, or prior, values.
    Integrated (I): represents the differencing of raw observations to allow for the time series to become stationary (i.e., data values are replaced by the difference between the data values and the previous values).
    Moving average (MA): incorporates the dependency between an observation and a residual error from a moving average model applied to lagged observations
"

#### SARIMA
"
SARIMA adds three new hyperparameters to specify the autoregression (AR), differencing (I), and moving average (MA) for the seasonal component of the series.
"

#### TBATS
"
TBATS models are for time series data with multiple seasonality. For example, retail sales data may have a daily pattern and weekly pattern, as well as an annual pattern.

In TBATS, a Box-Cox transformation is applied to the original time series, and then this is modeled as a linear combination of an exponentially smoothed trend, a seasonal component, and an ARMA component.
"
### ML
 nothing of interest
### DL
 nothing of interest

END OF https://www.datacamp.com/tutorial/tutorial-time-series-forecasting
-------


# BLOG 2
WHAT is Deep Learning?
https://www.datacamp.com/tutorial/tutorial-deep-learning-tutorial


Summary:
"
Deep learning is a type of machine learning that teaches computers to perform tasks by learning from examples, much like humans do. Imagine teaching a computer to recognize cats: instead of telling it to look for whiskers, ears, and a tail, you show it thousands of pictures of cats. The computer finds the common patterns all by itself and learns how to identify a cat

deep learning uses something called "neural networks," which are inspired by the human brain. These networks consist of layers of interconnected nodes that process information. The more layers, the "deeper" the network, allowing it to learn more complex features and perform more sophisticated tasks.
"


## ML to DL
Dl is a subset of ML, that has NN with 3 or more layers. It learns from large amounts of data.

For more info, see:
https://www.datacamp.com/tutorial/machine-deep-learning

**Feature engineering**
Selecting and adjusting most important data (**features**) from raw data, to use in training the model.

### Building blocks of DL

**Neural Networks**

Interconnected Nodes (*neurons*). consists of layers of nodes, each layer for a specific task.
See also: https://www.datacamp.com/blog/what-are-neural-networks

**Deep neural networks**

NN with many layers (deep). Also called *DNN*

**Activation functions**
Decision maker inside a node. Decide, which information will be passed along, and which will be discarded.


Input layers contain raw data, which is passed to one or more hidden layers and then to the output layer.
IN the Hidden layers, the data is classified based on broader target inforamtion. with each layer, scope of target value narrows down.

Output layer uses this information from the hidden layers to select most probable result/label.

### Different DL Models

**Supervised learning**
uses labeled dataset, to train models for classification of data or prediction of values. dataset contains features and target labels, which allows model, to learn over time by minimiing the loss between predicted and actual albels.
Can be **classification** or **regression** problem

**Regression**
Regression model learns the relationship between input and output variables to predict the outcome.

### Deeper look to DL concepts:
https://www.datacamp.com/tutorial/tutorial-deep-learning-tutorial#:~:text=project%2E-,A%20Deeper%20Look%20at%20Deep%20Learning%20Concepts

**Activation functions**
Decides weather input should pass through a neuron/node or not, based on its significatnce.
Non-linear.
Common activaito functions are Tanh, ReLU, Sigmoid, Softmax and more.

See here: **https://towardsdatascience.com/what-are-activation-functions-in-deep-learning-cc4f01e1cf5c/**

**Loss function**
Difference between actual and predicted values.
For different problems, different loss functions are chosen, for example MSE:
Loss = Sum(Predicted - Actual)²

**Backpropagation**
First, in forward propagation, NN is initialized with random inputs to produce a random output.
To increase performance of model, weights are randomly adjusted with *backpropagation*. 
To track optimization, loss function is used to find global minima for maximizing models accuracy.

**Stochastic gradient Descent**
*Gradient descent* is used to optimize the loss function, by changing weights in a controlled way to achieve minimum loss (objective). To achieve this objective, we need direction wether to increase/decrease weights for better performance.
Derivative of lsos function gives direction to update weights.

IN *stochastic gradient descent*, samples are divided into barches instead of using the entire dataset, to optimize gd (for faster computation).

**Hyperparameter**
Tunable parameters adjusted before running the **training process**. Directly affect the model performance.
Most used HP:
- learning rate: step size of each iteration.
- batch size: number of samples passed through a nn at a time.
- number of epochs: an iteration of how many times a model changes weights. too many epochs can cause models to overfit and too few can cause models to underfit, so we have to choose a medium number.

### Popular algorithms

#### Convolutional NN (CNN)
Feed forward NN. used for computer vision (CV) image classification.
Good at recognizing patterns, lines, shapes.
Consists of a 
- input layer
- convolutional layer 
- pooling layer
- output layer

after a conv. layer follows a pooling layer.

#### Recurrent NN (RNN)
Different from Feed-forward networks:
Output of the layer is fed back into the input to predict the output of the layer.
This helps with sequential data, as it can store the informaiton of previous samples to predict future samples.

#### Long short-term memory Networks
Advanced types of RNN, that can retain greater information on past values. solves vanishing gradient problems of RNN.


## Frameworks

### Tensorflow

### Keras

### Pytorch


----------------------

# BLOG 3 
AI Time Series Forecasting:
A Beginners Guide

https://www.datacamp.com/blog/ai-time-series-forecasting


Intro:
[One of the benefits of] "neural networks (RNNs) and long short-term memory (LSTM) networks, is their capacity to recognize intricate patterns in time series data. These models can identify seasonal trends, cyclical behaviors, and anomalies that may otherwise be difficult to detect.

For example, while conventional approaches may rely on predefined assumptions about seasonality or trend direction, AI models dynamically learn from the data itself, discovering hidden relationships between variables, and leading to more precise forecasting. 
"

eher zum vergessen.




--------

Super gut mit code anleitung für GRU und LSTM:
(erklärung is so mittel, aber code ist gut)
https://www.datacamp.com/tutorial/tutorial-for-recurrent-neural-network

Next steps: videos zu FNN, RNN und dann LSTM und GRU

RNN/LSTM:
very good series:https://www.youtube.com/watch?v=Mdp5pAKNNW4
+ episdoe 164, 165, 166, 166b, 181. insbes. 164, 166
also 3blue1brown, for overview of calculus (optional), overview of linear algebra (optional) and Neural networks 
also this: https://www.youtube.com/watch?v=WCUNPb-5EYI



------
Add section regarding computational stuff (UMsetzung) of the pipeline, like managing memory leaks etc.
Speeding operations (vectorized vs loops).

------
Add section about tensorflow, difference eager<>graph mode. how to speed up etc.

-----
Design philosophys (u.a.):
design composite functino (orchestrator function) in a 'passing data' style:
A = get_A()
B = get_B(A)
...

or a stateful design:
self.step_one()
self.step_two()

argue why whcih was chosen -- second one, because you can make sure, no data gets carried over? and this is why the reset_state() function is needed ig.

--------------


note: trend parameter in (s)arima build_model: can be n, c, t, ct. talk about why which was chosen.
defaults are c or n. n or t is possible, others throw error?


# Environments

For deployment reasons, i switched to conda (miniconda3).

IN terminal:
deactivate all envs 
deactive

activate conda:
conda activate thesis_ls 
oder generell:
conda activate base 


get requirements (only installed modules):
pipreq --forece . --ignore .venv,notebooks,data,logs,docs,plots,results
to search all files here (.), and ignore some others (especially notebooks). forece to overwrite exisitng requirements.txt