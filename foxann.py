from datetime import time
from sklearn.model_selection import KFold
import numpy as np


class NeuralNetwork:
    """
    Neural Network implementation with training, evaluation, and cross-validation capabilities.
    """

    def __init__(self, nn_params, epochs):
        """
        Initialize the neural network with the given parameters.

        Parameters:
        - nn_params: Tuple containing input size, hidden sizes, and output size.
        - epochs: Number of training epochs.
        """
        self.input_size, self.hidden_sizes, self.output_size = nn_params

        # Initialize weights and biases for each layer
        self.weights = [np.random.randn(self.input_size, self.hidden_sizes[0])]
        self.biases = [np.zeros((1, self.hidden_sizes[0]))]
        for i in range(1, len(self.hidden_sizes)):
            self.weights.append(np.random.randn(self.hidden_sizes[i-1], self.hidden_sizes[i]))
            self.biases.append(np.zeros((1, self.hidden_sizes[i])))
        self.weights.append(np.random.randn(self.hidden_sizes[-1], self.output_size))
        self.biases.append(np.zeros((1, self.output_size)))

        self.epochs = epochs

    def evaluate(self, predictions, y_test):
        """
        Evaluate the model's predictions against the true labels.

        Parameters:
        - predictions: Predicted values from the model.
        - y_test: True labels.

        Returns:
        - accuracy, precision, recall, f1_score: Evaluation metrics.
        """
        # Threshold predictions at 0.5
        binary_predictions = (predictions > 0.5).astype(int)

        # Calculate accuracy
        accuracy = np.mean(binary_predictions == y_test)

        # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
        TP = np.sum(np.logical_and(binary_predictions == 1, y_test == 1))
        FP = np.sum(np.logical_and(binary_predictions == 1, y_test == 0))
        FN = np.sum(np.logical_and(binary_predictions == 0, y_test == 1))

        # Calculate precision, recall, and F1-score
        precision = TP / (TP + FP + 1e-9)  # Add a small value to avoid division by zero
        recall = TP / (TP + FN + 1e-9)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)

        return accuracy, precision, recall, f1_score

    def softmax(self, x):
        """
        Apply softmax activation function.

        Parameters:
        - x: Input array.

        Returns:
        - Softmax probabilities.
        """
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def sigmoid(self, x):
        """
        Apply sigmoid activation function.

        Parameters:
        - x: Input array.

        Returns:
        - Sigmoid activations.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Compute the derivative of the sigmoid function.

        Parameters:
        - x: Input array.

        Returns:
        - Derivative of sigmoid activations.
        """
        return x * (1 - x)

    def forward(self, inputs):
        """
        Perform forward propagation through the network.

        Parameters:
        - inputs: Input data.

        Returns:
        - Output activations.
        """
        self.activations = [inputs]
        for i in range(len(self.weights) - 1):
            self.activations.append(self.sigmoid(np.dot(self.activations[-1], self.weights[i]) + self.biases[i]))
        self.activations.append(self.softmax(np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]))
        return self.activations[-1]

    def backward(self, targets, learning_rate):
        """
        Perform backward propagation and update weights and biases.

        Parameters:
        - targets: True labels.
        - learning_rate: Learning rate for weight updates.
        """
        errors = [targets - self.activations[-1]]
        deltas = [errors[-1] * self.sigmoid_derivative(self.activations[-1])]

        for i in range(len(self.weights) - 2, -1, -1):
            errors.insert(0, deltas[0].dot(self.weights[i+1].T))
            deltas.insert(0, errors[0] * self.sigmoid_derivative(self.activations[i+1]))

        for i in range(len(self.weights)):
            self.weights[i] += self.activations[i].T.dot(deltas[i]) * learning_rate
            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * learning_rate

    def get_weights_vector(self):
        """
        Concatenate all weight matrices and bias vectors into a single vector.

        Returns:
        - Flattened vector of weights and biases.
        """
        return np.concatenate([w.flatten() for w in self.weights] + [b.flatten() for b in self.biases])

    def set_weights_vector(self, weights_vector):
        """
        Set weights and biases from a flattened vector.

        Parameters:
        - weights_vector: Flattened vector of weights and biases.
        """
        weight_shapes = [w.shape for w in self.weights]
        bias_shapes = [b.shape for b in self.biases]
        split_points = np.cumsum([np.prod(shape) for shape in weight_shapes + bias_shapes[:-1]])
        split_weights = np.split(weights_vector, split_points)
        self.weights = [split_weight.reshape(shape) for split_weight, shape in zip(split_weights[:len(weight_shapes)], weight_shapes)]
        self.biases = [split_weight.reshape(shape) for split_weight, shape in zip(split_weights[len(weight_shapes):], bias_shapes)]

    def train_nn(self, X, y):
        """
        Train the neural network using cross-validation.

        Parameters:
        - X: Input data.
        - y: True labels.

        Returns:
        - mean_losses, mean_accuracy, mean_precision, mean_recall, mean_f1_score, mean_elapsed_time: Training metrics.
        """
        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        validation_losses = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        elapsed_times = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            start = time.time()
            fold_validation_loss = []

            for _ in range(self.epochs):
                self.forward(X_train)
                self.backward(y_train, 0.01)

                self.forward(X_test)
                fold_validation_loss.append(np.mean(np.square(y_test - self.activations[-1])))

            validation_losses.append(fold_validation_loss)
            end = time.time()
            elapsed_time = end - start
            elapsed_times.append(elapsed_time)

            predictions = self.forward(X_test)
            accuracy, precision, recall, f1_score = self.evaluate(predictions, y_test)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)

        mean_accuracy = np.mean(accuracies)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_f1_score = np.mean(f1_scores)
        mean_elapsed_time = np.mean(elapsed_times)
        mean_losses = np.mean(validation_losses, axis=0)

        return mean_losses, mean_accuracy, mean_precision, mean_recall, mean_f1_score, mean_elapsed_time

class FOX:
    """
    FOX optimization algorithm for solving optimization problems.

  
    Cite as: 
    Mohammed, H., Rashid, T. FOX: a FOX-inspired optimization algorithm. Appl Intell (2022). 
    https://doi.org/10.1007/s10489-022-03533-0
    """

    def __init__(self, num_agents, max_iter, lower_bound, upper_bound, dimension, objective_function):
        """
        Initialize the FOX optimizer with the given parameters.
        
        Parameters:
        - num_agents: Number of agents in the population.
        - max_iter: Maximum number of iterations.
        - lower_bound: Lower bound of the search space.
        - upper_bound: Upper bound of the search space.
        - dimension: Dimensionality of the search space.
        - objective_function: Objective function to be optimized.
        """
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dimension = dimension
        self.objective_function = objective_function
        self.best_score = np.inf  # Best score initialized to infinity (for minimization)
        self.best_position = np.zeros((dimension))  # Best position initialized

    def initialization(self):
        """
        Generate random initial positions for agents.
        The positions are sampled from a uniform distribution between lower and upper bounds.
        
        Returns:
        - An array of shape (num_agents, dimension) containing initial positions of agents.
        """
        return np.random.uniform(self.lower_bound, self.upper_bound, size=(self.num_agents, self.dimension))

    def update_fitness_for_agents(self, X, it):
        """
        Update the fitness of each agent and determine the best score and position.
        
        Parameters:
        - X: Current positions of the agents.
        - it: Current iteration number.
        """
        for i in range(self.num_agents):
            agent = X[i, :]
            fitness = self.objective_function(agent)
            if fitness < self.best_score:
                self.best_score = np.copy(fitness)
                self.best_position = np.copy(agent)
                print(f"[{it+1}] Best Score:  ", np.round(self.best_score, 4),
                      "     Best Position:  ", np.round(self.best_position, 4))

    def optimize(self):
        """
        Optimize the objective function using the FOX algorithm.
        This function performs the optimization over the specified number of iterations.
        """
        X = self.initialization()
        for it in range(self.max_iter):
            a = 2 * (1 - (it / self.max_iter))  # Adaptive coefficient
            Jump = 0
            MinT = np.inf

            self.update_fitness_for_agents(X, it)

            for i in range(self.num_agents):
                r = np.random.rand()
                p = np.random.rand()
                if r > 0.5:  # Exploitation phase
                    Time = np.random.rand(1, self.dimension)
                    distance_fox_rat = 0.5 * self.best_position
                    t = (np.sum(Time) / self.dimension) / 2
                    Jump = 0.5 * 9.81 * t**2

                    if p > 0.18:  # Jump to north
                        X[i] = distance_fox_rat * Jump * 0.108
                    elif p <= 0.18:  # Jump to opposite direction
                        X[i] = distance_fox_rat * Jump * 0.72

                    if MinT > t:
                        MinT = t
                else:  # Exploration phase
                    X[i] = self.best_position + np.random.randn(1, self.dimension) * (MinT * a)

                X = self.initialization()

class FOXANN(NeuralNetwork):
    """
    FOXANN: A neural network that uses the FOX optimizer to enhance performance.
    Inherits from NeuralNetwork.
    
    Cite as: 
    Mahmood A. Jumaah, Yossra H. Ali, Tarik A. Rashid (2024). 
    Q-FOX Learning: Breaking Tradition in Reinforcement Learning, 
    https://doi.org/10.48550/arXiv.2402.16562

    Jumaah, Mahmood A.; Ali, Yossra H.; Rashid, Tarik A.; and Vimal, S. (2024). 
    FOXANN: A Method for Boosting Neural Network Performance, 
    Journal of Soft Computing and Computer Applications: Vol. 1: Iss. 1, Article 2. 
    https://jscca.uotechnology.edu.iq/jscca/vol1/iss1/2
    """

    def __init__(self, nn_params, epochs):
        """
        Initialize FOXANN with given parameters.

        Parameters:
        - nn_params: Tuple containing input size, hidden sizes, and output size.
        - epochs: Number of training epochs.
        """
        super().__init__(nn_params, epochs)

    def optimize_weight(self, X, y):
        """
        Optimize weights using the FastFOX optimizer.

        Parameters:
        - X: Input data.
        - y: True labels.

        Returns:
        - fox.scores: Scores from the FOX optimization process.
        """
        def obj(weights):
            self.set_weights_vector(weights)
            self.forward(X)
            loss = np.mean(np.square(y - self.activations[-1]))
            return loss

        fox = FOX(100, self.epochs, -1.5, 1.5, len(self.get_weights_vector()), obj)
        fox.optimize()
        self.set_weights_vector(fox.best_position)

        return fox.scores

    def train_foxann(self, X, y):
        """
        Train FOXANN using cross-validation.

        Parameters:
        - X: Input data.
        - y: True labels.

        Returns:
        - mean_losses, mean_accuracy, mean_precision, mean_recall, mean_f1_score, mean_elapsed_time: Training metrics.
        """
        kf = KFold(n_splits=4, shuffle=True, random_state=42)

        validation_losses = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        elapsed_times = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            start = time.time()
            training_loss = self.optimize_weight(X_train, y_train)
            end = time.time()

            validation_losses.append(training_loss)

            predictions = self.forward(X_test)
            accuracy, precision, recall, f1_score = self.evaluate(predictions, y_test)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)
            elapsed_times.append(end - start)

        mean_accuracy = np.mean(accuracies)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)
        mean_f1_score = np.mean(f1_scores)
        mean_elapsed_time = np.mean(elapsed_times)
        mean_losses = np.mean(validation_losses, axis=0)

        return mean_losses, mean_accuracy, mean_precision, mean_recall, mean_f1_score, mean_elapsed_time
