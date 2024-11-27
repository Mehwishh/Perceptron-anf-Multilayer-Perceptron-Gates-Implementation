import numpy as np
# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Feedforward function
def feedforward(inputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    # Input to hidden layer
    hidden_layer_inputs = sigmoid(np.dot(inputs, weights_input_hidden) + bias_hidden)
    
    # Hidden layer to output
    output_input = sigmoid(np.dot(hidden_layer_inputs, weights_hidden_output) + bias_output)
    
    return hidden_layer_inputs, output_input

# Backpropagation function
def backpropagation(inputs, outputs, hidden_layer_inputs, predicted_output, 
                    weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, learning_rate):
    
    # Calculate error for the output
    error = outputs - predicted_output

    # Calculate gradients for the output layer
    delta_output = error * sigmoid_derivative(predicted_output)
    
    # Update weights and bias for the output layer
    weights_hidden_output += learning_rate * np.dot(hidden_layer_inputs.T, delta_output)
    bias_output += learning_rate * np.sum(delta_output, axis=0)
    
    # Calculate error for the hidden layer
    delta_hidden = delta_output.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer_inputs)
    
    # Update weights and biases for the hidden layer
    weights_input_hidden += learning_rate * np.dot(inputs.T, delta_hidden)
    bias_hidden += learning_rate * np.sum(delta_hidden, axis=0)
    
    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, error

# Training function that integrates feedforward and backpropagation
def train(inputs, outputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, learning_rate, epochs):
    for epoch in range(epochs):
        # Perform feedforward pass
        hidden_layer_inputs, predicted_output = feedforward(inputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
        
        # Perform backpropagation and calculate error
        weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, error = backpropagation(
            inputs, outputs, hidden_layer_inputs, predicted_output, 
            weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, learning_rate)
        
        # Print error at each epoch
        if (epoch + 1) % 1000 == 0:  # Print every 1000 epochs
            print(f"Epoch {epoch + 1}/{epochs}, Error: {np.mean(np.abs(error))}")
    
    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

# Define a function to test the neural network after training
def test_model(weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    while True:
        # Ask the user if they want to test the model
        user_input = input("\nDo you want to test the model? (yes/no): ").lower()
        if user_input != "yes":
            print("Exiting test mode")
            break
        
        # Get user inputs for testing (simultaneously)
        user_inputs = input("Enter two binary inputs separated by a space (e.g., '0 1'): ")
        try:
            input1, input2 = map(int, user_inputs.split())  # Split input into two integers
        except ValueError:
            print("Invalid input. Please enter two binary values separated by a space.")
            continue
        
        if input1 not in [0, 1] or input2 not in [0, 1]:
            print("Invalid input. Please enter binary values (0 or 1).")
            continue

        test_inputs = np.array([input1, input2])
        
        # Perform feedforward pass
        _, predicted_output = feedforward(test_inputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
        predicted_value = 1 if predicted_output >= 0.5 else 0
        print(f"Prediction: {predicted_value} (Output: {predicted_output[0]:.4f})")

# Main function to take user inputs and run the model
def main():
    print("Welcome to the Neural Network Trainer for XOR and XNOR gates!")
    
    # Take user input to select the gate type
    gate_type = input("Which gate do you want to train? (XOR/XNOR): ").upper()

    # XOR and XNOR gate inputs are the same
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    # Expected outputs for XOR and XNOR gates
    if gate_type == "XOR":
        outputs = np.array([[0], [1], [1], [0]])
    elif gate_type == "XNOR":
        outputs = np.array([[1], [0], [0], [1]])
    else:
        print("Invalid gate type! Please enter XOR or XNOR.")
        return
    
    # Initialize weights and biases
    print("\nUsing default weights for input to hidden layer (2 neurons, 2 weights each) and hidden to output.")
    weights_input_hidden = np.array([[0.5, 0.2], [0.3, 0.7]]) 
    weights_hidden_output = np.array([[0.6], [0.9]])  
    
    bias_hidden = np.array([0.5, 0.6])  # biases
    bias_output = np.array([0.5])  #  output bias
    
    # Take user input for learning rate and epochs
    learning_rate = float(input("\nEnter learning rate (e.g., 0.1): "))
    epochs = int(input("Enter the number of training epochs (e.g., 10000): "))
    
    # Train the model
    weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = train(
        inputs, outputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, learning_rate, epochs)
    
    # Allow the user to test the model 
    test_model(weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)

if __name__ == "__main__":
    main()
