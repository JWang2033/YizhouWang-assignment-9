import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # TODO: define layers and initialize weights
        # Initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)

        # Initialize biases
        self.b1 = np.zeros((1, hidden_dim))
        self.b2 = np.zeros((1, output_dim))

        # Layers information
        self.layers = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim
        }

        # To store activations and gradients for visualization
        self.activations = {}
        self.gradients = {}

        # Apply activation function to compute A1
    def forward(self, X):
        # Compute Z1 (pre-activation for hidden layer)
        Z1 = np.dot(X, self.W1) + self.b1  # Shape: (num_samples, hidden_dim)
        self.activations["Z1"] = Z1

        # Apply activation function to compute A1
        if self.activation_fn == 'tanh':
            A1 = np.tanh(Z1)
        elif self.activation_fn == 'relu':
            A1 = np.maximum(0, Z1)
        elif self.activation_fn == 'sigmoid':
            A1 = 1 / (1 + np.exp(-Z1))
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_fn}")

        self.activations["A1"] = A1

        # Compute Z2 (pre-activation for output layer)
        Z2 = np.dot(A1, self.W2) + self.b2  # Shape: (num_samples, output_dim)
        self.activations["Z2"] = Z2

        # Compute A2 (output layer activations) using sigmoid
        A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation
        self.activations["A2"] = A2

        return A2

    '''def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1  # Linear combination for hidden layer
        self.A1 = self.activation(self.Z1)     # Activation for hidden layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # Linear combination for output layer
        self.A2 = self.Z2                       # Linear output (for regression or raw logits)

        # Store activations for visualization
        self.activations = {
            'Z1': self.Z1,
            'A1': self.A1,
            'Z2': self.Z2,
            'A2': self.A2
        }
        return self.A2'''
    def backward(self, X, y):
        m = X.shape[0]  # Number of samples

        # Extract activations
        A2 = self.activations["A2"]
        A1 = self.activations["A1"]
        Z1 = self.activations["Z1"]

        # Compute gradient of loss with respect to Z2 (output layer pre-activation)
        dZ2 = A2 - y  # Shape: (num_samples, output_dim)

        # Compute gradients for output layer
        dW2 = np.dot(A1.T, dZ2) / m  # (hidden_dim, output_dim)
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m  # (1, output_dim)

        # Compute gradients for hidden layer
        dA1 = np.dot(dZ2, self.W2.T)  # (num_samples, hidden_dim)
        if self.activation_fn == 'tanh':
            dZ1 = dA1 * (1 - np.tanh(Z1) ** 2)  # Derivative of tanh
        elif self.activation_fn == 'relu':
            dZ1 = dA1 * (Z1 > 0).astype(float)  # Derivative of ReLU
        elif self.activation_fn == 'sigmoid':
            sig = 1 / (1 + np.exp(-Z1))
            dZ1 = dA1 * (sig * (1 - sig))  # Derivative of sigmoid
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_fn}")

        # Compute gradients for input-to-hidden layer
        dW1 = np.dot(X.T, dZ1) / m  # (input_dim, hidden_dim)
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m  # (1, hidden_dim)

        # Update weights with gradient descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        # Store gradients for visualization
        self.gradients = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    '''def backward(self, X, y):
        # TODO: compute gradients using chain rule
        # Compute gradients for output layer
        m = X.shape[0]
        dZ2 = self.A2 - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Compute gradients for hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.activation_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # TODO: update weights with gradient descent
        # Update weights and biases
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        # TODO: store gradients for visualization
        self.gradients = {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
        "hidden_pre_activation": dZ1,
        "output": dZ2
        }'''


def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    # Clear previous frame's content
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # Hidden Space Graph
    hidden_features = mlp.activations["A1"]
    ax_hidden.scatter(
        hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2],
        c=y.ravel(), cmap='bwr', alpha=0.7
    )
    ax_hidden.set_xlim(-1, 1)
    ax_hidden.set_ylim(-1, 1)
    ax_hidden.set_zlim(-1, 1)
    ax_hidden.set_title(f"Hidden Space at Step {frame * 10}")

    # Input Space Graph
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = mlp.forward(grid).reshape(xx.shape)

    ax_input.contourf(xx, yy, predictions, levels=50, cmap='bwr', alpha=0.5)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k')
    ax_input.set_xlim(-3, 3)
    ax_input.set_ylim(-3, 3)
    ax_input.set_title(f"Input Space at Step {frame * 10}")

    # Gradients Visualization Graph
    node_positions = {
        'x1': (0.1, 0.1),
        'x2': (0.1, 0.9),
        'h1': (0.4, 0.3),
        'h2': (0.4, 0.6),
        'h3': (0.4, 0.9),
        'y': (0.9, 0.5)
    }

    for node, pos in node_positions.items():
        ax_gradient.scatter(*pos, color='blue', s=500, zorder=3)
        ax_gradient.text(pos[0], pos[1], node, color='black', ha='center', va='center', fontsize=12)

    weights = [
        ('x1', 'h1', np.linalg.norm(mlp.gradients['dW1'][0, 0])),
        ('x1', 'h2', np.linalg.norm(mlp.gradients['dW1'][0, 1])),
        ('x1', 'h3', np.linalg.norm(mlp.gradients['dW1'][0, 2])),
        ('x2', 'h1', np.linalg.norm(mlp.gradients['dW1'][1, 0])),
        ('x2', 'h2', np.linalg.norm(mlp.gradients['dW1'][1, 1])),
        ('x2', 'h3', np.linalg.norm(mlp.gradients['dW1'][1, 2])),
        ('h1', 'y', np.linalg.norm(mlp.gradients['dW2'][0, 0])),
        ('h2', 'y', np.linalg.norm(mlp.gradients['dW2'][1, 0])),
        ('h3', 'y', np.linalg.norm(mlp.gradients['dW2'][2, 0])),
    ]

    max_weight = max(w[2] for w in weights)
    for start, end, weight in weights:
        start_pos = node_positions[start]
        end_pos = node_positions[end]
        intensity = weight / max_weight if max_weight > 0 else 0
        ax_gradient.plot(
            [start_pos[0], end_pos[0]],
            [start_pos[1], end_pos[1]],
            color=(1 - intensity, 0, intensity),
            linewidth=2,
            alpha=0.7,
        )
    ax_gradient.set_title(f"Gradients at Step {frame * 10}")
    ax_gradient.set_xlim(0, 1)
    ax_gradient.set_ylim(0, 1)
    ax_gradient.axis('off')

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "sigmoid"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)