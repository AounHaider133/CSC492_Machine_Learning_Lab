import torch
import torch.nn as nn
import torch.optim as optim

# Generate random input data
torch.manual_seed(42)
X = torch.randn(100, 1)  # Input features
Y = 2 + 3 * X  + torch.randn(100, 1) * 0.1  # True labels

# Define the linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # Input size: 1, Output size: 1

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = LinearRegression()

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, Y)

    # Backward and optimize
    optimizer.zero_grad()  # Clear gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Print the final parameters
print('Final Parameters:')
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

# Test the model
with torch.no_grad():
    test_input = torch.tensor([[2.0]])
    predicted_output = model(test_input)
    print(f'Predicted output for test input {test_input.item():.1f}: {predicted_output.item():.2f}')
