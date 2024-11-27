# Activation function (Step function)
def activation_function(weighted_sum):
    return 1 if weighted_sum >= 0 else 0
# Function to train the perceptron model
def train_logic_gate_perceptron(input_table, expected_outputs, weights, learning_rate, epochs, bias):
    num_samples = len(input_table)
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        all_correct = True  # Flag to track if all predictions are correct
        for i in range(num_samples):
            inputs = input_table[i]  # Current input
            expected = expected_outputs[i]  # Expected output for this input      
            # Step 1: Compute weighted sum (net input)
            weighted_sum = sum(inputs[j] * weights[j] for j in range(len(weights))) + bias
            prediction = activation_function(weighted_sum)  # Step 2: Make prediction       
            # Step 3: Calculate the error
            error = expected - prediction       
            # Step 4: Update weights and bias if there is an error
            if error != 0:
                weights = [weights[j] + learning_rate * error * inputs[j] for j in range(len(weights))]
                bias += learning_rate * error  # Bias update added here
                all_correct = False  # Set flag to false as there was an error
            print(f"Inputs: {inputs}, Predicted: {prediction}, Expected: {expected}, Updated Weights: {weights}, Updated Bias: {bias}")
        # Step 5: If all predictions are correct, stop the training
        if all_correct:
            print(f"Training completed in {epoch + 1} epochs.")
            break       
    return weights, bias
# Function to test the trained perceptron
def predict_logic_gate_output(test_inputs, trained_weights, bias):
    weighted_sum = sum(test_inputs[j] * trained_weights[j] for j in range(len(test_inputs))) + bias
    prediction = activation_function(weighted_sum)
    return prediction
if __name__ == "__main__":
    input_table = [ [0, 0], [0, 1],[1, 0], [1, 1]]
    gate = input("Choose a logic gate to train (AND, OR, NOR, NAND): ").strip().upper() 
    if gate == "AND":
        expected_outputs = [0, 0, 0, 1]  # AND gate truth table
        bias = 0
    elif gate == "OR":
        expected_outputs = [0, 1, 1, 1]  # OR gate truth table
        bias = 0  
    elif gate == "NOR":
        expected_outputs = [1, 0, 0, 0]  # NOR gate truth table
        bias = 1
    elif gate == "NAND":
        expected_outputs = [1, 1, 1, 0]  # NAND gate truth table (opposite of AND)
        bias = 1
    else:
        print("Invalid gate type. Please choose AND, OR, NOR, or NAND.")
        exit()
    # Get user input for initial weights, learning rate, and epochs
    initial_weights = [float(i) for i in input("Enter initial weights (e.g., '0.0 0.0'): ").split()]
    learning_rate = float(input("Enter learning rate (e.g., 0.1): "))
    epochs = int(input("Enter number of epochs: "))
    # Train the perceptron for the chosen logic gate
    trained_weights, trained_bias = train_logic_gate_perceptron(input_table, expected_outputs, initial_weights, learning_rate, epochs, bias)
    # Testing phase
    while True:
        # Get test inputs from the user
        test_inputs = [float(i) for i in input("Enter test inputs (e.g., '0 1'): ").split()]      
        # Predict output 
        prediction = predict_logic_gate_output(test_inputs, trained_weights, trained_bias)
        print(f"Prediction for inputs {test_inputs} (Gate: {gate}): {prediction}")
        if input("Do you want to test another input? (yes/no): ").lower() == 'no':
            break
