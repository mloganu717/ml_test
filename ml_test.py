import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- Configuration ---
# Dataset parameters (same as your script)
n_samples = 200000
n_features = 100
n_informative = 50
n_redundant = 20
random_state_dataset = 42

# Neural Network and Training parameters
N_EPOCHS = 30  # Adjust this to make training longer or shorter
BATCH_SIZE = 512
LEARNING_RATE = 0.001
HIDDEN_SIZE1 = 256
HIDDEN_SIZE2 = 128
HIDDEN_SIZE3 = 64

# --- 0. Device Configuration ---
# For Apple Silicon (M2 Mac)
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using Apple MPS (GPU) device.")
# For Intel Arc (or other GPUs via CUDA/DirectML if PyTorch supports it transparently)
# Note: For Intel Arc with PyTorch-DirectML, it might pick it up automatically if installed.
# For Intel specific XPU via intel_extension_for_pytorch (IPEX), you'd use "xpu"
# This is a general check; specific Intel Arc setup might need IPEX or specific PyTorch builds.
elif torch.cuda.is_available(): # General CUDA check, DirectML might register as CUDA-like
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("MPS or CUDA not available. Using CPU. Ensure PyTorch for your Intel Arc is set up correctly for GPU.")
    print("For Intel Arc on Windows with DirectML, ensure 'torch-directml' is installed.")
    print("For Intel Arc on Linux with oneAPI, ensure 'intel-extension-for-pytorch' is installed and try device='xpu'.")


# --- 1. Generate a More Intensive Synthetic Dataset ---
print(f"Generating a dataset with {n_samples} samples and {n_features} features...")
X_np, y_np = make_classification(n_samples=n_samples,
                                 n_features=n_features,
                                 n_informative=n_informative,
                                 n_redundant=n_redundant,
                                 random_state=random_state_dataset)
print("Dataset generation complete.")

# --- 2. Feature Scaling ---
scaler = StandardScaler()
print("Scaling features...")
X_scaled_np = scaler.fit_transform(X_np)
print("Feature scaling complete.")

# --- 3. Split the Data ---
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X_scaled_np, y_np, test_size=0.3, random_state=42)

# --- 4. Convert to PyTorch Tensors and Create DataLoaders ---
X_train = torch.tensor(X_train_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1) # For BCELoss
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 5. Define the Neural Network Model (MLP) ---
class MLP(nn.Module):
    def __init__(self, input_features, hidden1, hidden2, hidden3):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_features, hidden1)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.layer3 = nn.Linear(hidden2, hidden3)
        self.layer4 = nn.Linear(hidden3, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.layer4(x))
        return x

model = MLP(n_features, HIDDEN_SIZE1, HIDDEN_SIZE2, HIDDEN_SIZE3).to(device)
print(f"\nModel defined and moved to {device}.")
# print(model)

# --- 6. Define Loss Function and Optimizer ---
criterion = nn.BCELoss() # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 7. Training Loop ---
print(f"Starting model training for {N_EPOCHS} epochs on device: {device}...")
start_time = time.time()

for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    if (epoch + 1) % 5 == 0 or epoch == N_EPOCHS -1 : # Print every 5 epochs or last epoch
        print(f"Epoch [{epoch+1}/{N_EPOCHS}], Loss: {avg_epoch_loss:.4f}")

end_time = time.time()
training_time = end_time - start_time
print(f"Training completed in: {training_time:.4f} seconds")

# --- 8. Evaluate the Model ---
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        predicted = (outputs > 0.5).float()
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

accuracy = 100 * correct / total
print(f"Model accuracy on the test set: {accuracy:.2f}%")