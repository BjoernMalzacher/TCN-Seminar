import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
import numpy as np # Added for generating the time series data
import matplotlib.pyplot as plt # New: Added for plotting the results

# --- 1. Causal 1D Convolution Helper ---
class CausalConv1d(nn.Conv1d):
    """
    Implements a 1D Causal Convolution.

    The key difference from a standard Conv1d is the padding applied to the
    input sequence before convolution to ensure the output y_t only depends
    on inputs x_0, ..., x_t, thus preserving temporal causality.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs):
        # Calculate required padding (K - 1) * d
        self.padding_size = (kernel_size - 1) * dilation
        
        # We pass padding=0 to the base Conv1d, as we handle padding manually.
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, **kwargs)

    def forward(self, x):
        # Pad the input tensor on the left side (temporal axis)
        # (batch_size, channels, sequence_length) -> pad left
        x = nn.functional.pad(x, (self.padding_size, 0))
        
        # Perform the convolution (which is now causal due to the padding)
        return super().forward(x)


# --- 2. TCN Residual Block ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        
        # --- Layer 1 ---
        self.conv1 = weight_norm(CausalConv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        ))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # --- Layer 2 ---
        self.conv2 = weight_norm(CausalConv1d(
            out_channels, out_channels, kernel_size, dilation=dilation
        ))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # --- Residual Connection (Downsample/Alignment) ---
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu_final = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        # Recommended practice: Initialize weights for stability
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if isinstance(self.downsample, nn.Conv1d):
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # Store input x for the residual connection
        res = x 

        # --- 1. Main Path F(x) ---
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # --- 2. Residual Connection (x) ---
        # Align dimensions if necessary using 1x1 convolution
        res = self.downsample(res)

        # --- 3. Activation(x + F(x)) ---
        # Element-wise addition of the main path output and the residual connection
        out = self.relu_final(out + res)
        
        return out
    
# --- 3. TCN Model Assembly ---
class TCN(nn.Module):
    """
    The complete TCN architecture, stacking multiple Residual Blocks
    with exponentially increasing dilation rates (d=1, 2, 4, 8, ...).
    """
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            input_size (int): The number of input features (channels).
            output_size (int): The number of output features/classes.
            num_channels (list): A list where each element is the number of 
                                 hidden channels for that residual layer.
            kernel_size (int): The size of the convolution filter.
            dropout (float): Dropout rate applied within the residual blocks.
        """
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        in_channels = input_size
        
        # Build the sequential stack of Residual Blocks
        for i in range(num_levels):
            # Dilation factor increases exponentially: d = 2^i
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            
            layers += [ResidualBlock(
                in_channels, 
                out_channels, 
                kernel_size, 
                dilation=dilation_size, 
                dropout=dropout
            )]
            
            in_channels = out_channels

        self.network = nn.Sequential(*layers)
        
        # Final Linear Layer for Prediction, mapping the last TCN channels 
        # to the required output size (e.g., 1 for single-step forecast)
        self.linear = nn.Linear(in_channels, output_size)

    def forward(self, x):
        # x shape: (N, C_in, L) -> (Batch Size, Input Channels, Sequence Length)
        out = self.network(x)
        
        # The TCN predicts based on the *last* time step of the output sequence.
        # Shape: (N, C_out)
        out = out[:, :, -1] 
        
        # Output shape: (N, output_size)
        return self.linear(out)
    
# --- 4. Training Function ---
def train_model(model, train_data, criterion, optimizer, num_epochs=10):
    """
    Trains the TCN model.
    """
    print("\n--- Starting Training ---")
    
    X_train, y_train = train_data
    model.train() # Set model to training mode

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad() # Zero the gradients
        
        # Forward pass
        output = model(X_train) 
        
        # Calculate loss (e.g., Mean Squared Error for regression)
        loss = criterion(output, y_train)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Print stats
        if epoch % 5 == 0 or epoch == num_epochs:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

    print("--- Training Finished ---")

# --- 5. Sequential Data Preparation Function ---
def create_sequential_data(long_series, lookback_window):
    """
    Converts a single long time series into (input sequence, target value) pairs
    using a sliding window.
    """
    X, y = [], []
    total_length = long_series.shape[0]
    
    # Iterate through the series, stopping before we run out of space for the target
    for i in range(total_length - lookback_window):
        # X is the segment from time i to i + lookback_window - 1
        x_segment = long_series[i: i + lookback_window]
        
        # y is the value at the next time step, i + lookback_window
        y_target = long_series[i + lookback_window]
        
        X.append(x_segment)
        y.append(y_target)

    # Stack them into tensors
    X_tensor = torch.stack(X)
    y_tensor = torch.stack(y).squeeze(-1) # Output shape (N, 1)

    # PyTorch TCN expects (N, C, L). 
    # Current X_tensor is (N, L, C). We need to permute the dimensions.
    X_tensor = X_tensor.permute(0, 2, 1) # -> (N, C, L)
    
    print(f"Total Sequential Samples (N): {X_tensor.shape[0]}")
    return X_tensor, y_tensor


if __name__ == '__main__':
    # --- Hyperparameters ---
    input_channels = 1      
    lookback_window = 100   # L: The length of the sequence the TCN looks back
    output_size = 1         
    kernel_size = 3         
    dropout_rate = 0.1
    learning_rate = 0.005
    num_epochs = 100
    num_channels_list = [16, 32, 64, 128] 
    
    # --- 1. Generate Long Continuous Time Series Data (Simple Count up to 1000) ---
    SERIES_LENGTH = 1000
    # Create the series [0, 1, 2, ..., 999] and normalize it to [0, 1]
    raw_series = np.arange(SERIES_LENGTH, dtype=np.float32) / (SERIES_LENGTH - 1)
    
    # Convert to Tensor and reshape to (Total_Length, Features=1)
    long_series = torch.from_numpy(raw_series).unsqueeze(1) 
    
    # --- 2. Create Sequential Input/Target Pairs ---
    # N = SERIES_LENGTH - lookback_window = 1000 - 100 = 900 samples
    X_train, y_train = create_sequential_data(long_series, lookback_window)
    
    # --- 3. Instantiate Model and Setup Training ---
    model = TCN(
        input_size=input_channels, 
        output_size=output_size, 
        num_channels=num_channels_list, 
        kernel_size=kernel_size, 
        dropout=dropout_rate
    )
    
    # Use Mean Squared Error Loss for forecasting the next value (regression)
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 4. Run Training Loop
    print(f"Input batch shape (N, C, L): {X_train.shape}")
    print(f"Target batch shape (N, Output_Size): {y_train.shape}")
    
    train_model(model, (X_train, y_train), criterion, optimizer, num_epochs=num_epochs)
    
    # 5. Final Check (Predicting the very first and last target value)
    model.eval()
    with torch.no_grad():
        # Predict on the first sample
        prediction_start = model(X_train[0:1])
        prediction_end = model(X_train[-1:])

    print(f"\n--- Prediction Check ---")
    print(f"Target (Start of series): {y_train[0].item():.4f}")
    print(f"Prediction (Start of series): {prediction_start.item():.4f}")
    
    print(f"Target (End of series): {y_train[-1].item():.4f}")
    print(f"Prediction (End of series): {prediction_end.item():.4f}")
    
    print("\n--- Model Architecture ---")
    print(model)

    # --- 6. Plot the entire predicted series (NEW SECTION) ---
    model.eval()
    with torch.no_grad():
        # Predict the output for all samples in the training set
        full_predictions = model(X_train) 





    

