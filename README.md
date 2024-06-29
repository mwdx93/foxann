# FOXANN
FOXANN, a novel classification model that combines the recently developed Fox optimizer(FOX) with ANN to solve ML problems. FOX replaces the backpropagation algorithm in ANN; optimizes synaptic weights; and achieves high classification accuracy with a minimum loss, improved model generalization, and interpretability.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/mwdali93/foxann.git
   cd foxann

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Example usage
   You can use the FOXANN class to train a neural network with cross-validation using the FOX optimizer.
   ```bash
   from FOXANN import FOXANN
   import numpy as np
   
   # Define neural network parameters
   nn_params = (input_size, hidden_sizes, output_size)
   epochs = 100
   
   # Create FOXANN instance
   foxann = FOXANN(nn_params, epochs)
   
   # Generate dummy data
   X = np.random.randn(100, input_size)
   y = np.random.randint(0, 2, (100, output_size))
   
   # Train FOXANN
   mean_losses, mean_accuracy, mean_precision, mean_recall, mean_f1_score, mean_elapsed_time = foxann.train_foxann(X, y)
   
   # Print results
   print("Mean Losses:", mean_losses)
   print("Mean Accuracy:", mean_accuracy)
   print("Mean Precision:", mean_precision)
   print("Mean Recall:", mean_recall)
   print("Mean F1 Score:", mean_f1_score)
   print("Mean Elapsed Time:", mean_elapsed_time)


## Citation
If you use FOXANN in your research or project, please cite:

Mahmood A. Jumaah, Yossra H. Ali, Tarik A. Rashid (2024). Q-FOX Learning: Breaking Tradition in Reinforcement Learning, https://doi.org/10.48550/arXiv.2402.16562

Jumaah, Mahmood A.; Ali, Yossra H.; Rashid, Tarik A.; and Vimal, S. (2024). FOXANN: A Method for Boosting Neural Network Performance, Journal of Soft Computing and Computer Applications: Vol. 1: Iss. 1, Article 2. https://jscca.uotechnology.edu.iq/jscca/vol1/iss1/2/

Mohammed, H., Rashid, T. FOX: a FOX-inspired optimization algorithm. Appl Intell (2022). https://doi.org/10.1007/s10489-022-03533-0

