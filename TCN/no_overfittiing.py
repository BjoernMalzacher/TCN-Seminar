import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils.data import TensorDataset, DataLoader
import os

# --- 1. Causal 1D Convolution Helper ---
class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs):
        self.padding_size = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, **kwargs)
    def forward(self, x):
        x = nn.functional.pad(x, (self.padding_size, 0))
        return super().forward(x)

# --- 2. TCN Residual Block ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = weight_norm(CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu_final = nn.ReLU()
        self.init_weights()
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if isinstance(self.downsample, nn.Conv1d):
            self.downsample.weight.data.normal_(0, 0.01)
    def forward(self, x):
        res = x 
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = self.downsample(res)
        out = self.relu_final(out + res)
        return out
    
# --- 3. TCN Model Assembly ---
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        in_channels = input_size
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            layers += [ResidualBlock(in_channels, out_channels, kernel_size, dilation=dilation_size, dropout=dropout)]
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(in_channels, output_size)
   
    def forward(self, x):
        out = self.network(x)
        out = out[:, :, -1] 
        return self.linear(out)
    
# --- 4. Training Function ---
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    print("\n--- Starting Training ---")
    model.train() 

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad() 
            output = model(batch_X) 
            loss = criterion(output, batch_y) 
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        if epoch % 10 == 0 or epoch == num_epochs:
            print(f"Epoch [{epoch}/{num_epochs}], Avg. Loss: {avg_epoch_loss:.6f}")

    print("--- Training Completed ---")

# --- 5. Data Preparation Helper ---
def create_sequential_data(long_series, lookback_window):
    """
    Creates Input (X) and Target (y) pairs from a long time series.
    """
    X, y = [], []
    total_length = long_series.shape[0]
    # Ensure we don't go out of bounds
    if total_length <= lookback_window:
        raise ValueError("Data series is shorter than the lookback window!")

    for i in range(total_length - lookback_window):
        x_segment = long_series[i: i + lookback_window]
        y_target = long_series[i + lookback_window]
        X.append(x_segment)
        y.append(y_target)
    
    X_tensor = torch.stack(X)
    y_tensor = torch.stack(y) 
    X_tensor = X_tensor.permute(0, 2, 1) # -> (Batch, Channels, Length)
    return X_tensor, y_tensor

# --- 6. Helper: Load File ---
def load_data_from_file(filename):
    """
    Robustly loads comma-separated or newline-separated integers from a file.
    """
    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found. Returning empty list.")
        return []
        
    with open(filename, "r") as f:
        content = f.read()
    
    # Handle both newlines and commas
    content = content.replace('\n', ',')
    numbers = [float(x.strip()) for x in content.split(",") if x.strip() != ""]
    return numbers

# --- 7. Autoregressive Forecasting Function ---
def forecast_autoregressive(model, start_sequence, forecast_steps):
    model.eval()
    generated_values = []
    current_window = start_sequence.clone() # Shape (L, C)
    
    # Ensure current_window is (Lookback, Channels)
    if current_window.shape[0] != start_sequence.shape[0]:
         current_window = current_window.permute(1, 0)

    with torch.no_grad():
        for _ in range(forecast_steps):
            # TCN expects (Batch, Channels, Length)
            input_tensor = current_window.permute(1, 0).unsqueeze(0) 
            prediction = model(input_tensor) # Shape (1, 1)
            
            generated_values.append(prediction.item())
            
            # Update window: Drop oldest, add new prediction
            new_value_tensor = prediction.view(1, 1) 
            current_window = torch.cat((current_window[1:], new_value_tensor), dim=0)
            
    return generated_values

# ==========================================
#               MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    
# --- A. CONFIGURATION (FIXED) ---
    input_channels = 1      
    lookback_window = 768    
    output_size = 1         
    kernel_size = 3        
    dropout_rate = 0.01  
    learning_rate = 0.002    
    num_epochs = 120       
    
    num_channels_list = [64] * 8 
    
    BATCH_SIZE  = 32  # Smaller batch size often generalizes better for time series
    
    # Define your files here
    files_to_load = [
        "parkende_autos_30tage_schule_kirche_restaurant_48protag.txt",
        #"parking_data.txt",  # Uncomment if you want to Merge this into training
        "test.txt"           # Uncomment if you want to Merge this into training
    ]

    # Percentage of data to use for Testing (The end of the data)
    TEST_SPLIT_RATIO = 0.15  # 15% of data will be Test
    
    # --- B. DATA LOADING ---
    print("--- Loading Data ---")
    raw_data_list = []
    for file_name in files_to_load:
        data = load_data_from_file(file_name)
        print(f"Loaded {len(data)} points from {file_name}")
        raw_data_list.extend(data)
    
    if len(raw_data_list) == 0:
        raise ValueError("No data loaded. Check filenames.")

    raw_series = np.array(raw_data_list)
    total_len = len(raw_series)
    print(f"Total Combined Series Length: {total_len}")

    # --- C. PERFECT DATA SPLIT (Train vs Test) ---
    split_index = int(total_len * (1 - TEST_SPLIT_RATIO))
    
    train_raw = raw_series[:split_index]
    test_raw = raw_series[split_index:]
    
    print(f"Training Samples: {len(train_raw)}")
    print(f"Test Samples:     {len(test_raw)}")

    # --- D. NORMALIZATION (Prevent Data Leakage) ---
    # CRITICAL: Calculate Min/Max ONLY on Training Data
    train_min = train_raw.min()
    train_max = train_raw.max()
    
    # Apply to Train
    train_norm = (train_raw - train_min) / (train_max - train_min)
    # Apply to Test (using Train metrics)
    test_norm = (test_raw - train_min) / (train_max - train_min)
    
    # Convert to Tensors (Length, Channels)
    train_tensor = torch.from_numpy(train_norm).unsqueeze(1).float()
    
    # --- E. CREATE DATASETS ---
    print("\nCreating Sequential Datasets...")
    X_train, y_train = create_sequential_data(train_tensor, lookback_window)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # --- F. MODEL SETUP ---
    model = TCN(
        input_size=input_channels, 
        output_size=output_size, 
        num_channels=num_channels_list, 
        kernel_size=kernel_size, 
        dropout=dropout_rate
    )
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # --- G. TRAINING ---
    train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)
    
    # --- H. EVALUATION / FORECASTING ---
    # The "Seed" is the LAST 'lookback_window' points of the TRAINING set.
    # This allows us to predict the very first point of the Test set, and continue.
    forecast_seed = train_tensor[-lookback_window:]
    
    num_test_steps = len(test_raw)
    print(f"\n--- Starting Autoregressive Forecast for {num_test_steps} steps ---")
    
    # Run Forecast
    generated_values_norm = forecast_autoregressive(model, forecast_seed, num_test_steps)
    
    # Denormalize Forecast
    generated_values_denorm = (np.array(generated_values_norm) * (train_max - train_min)) + train_min
    
    # --- I. PLOTTING ---
    plt.figure(figsize=(14, 6))
    
    # 1. Plot Training Data (End of it)
    # Let's show only the last 500 points of training to keep the plot readable
    vis_lookback = 300
    plt.plot(range(split_index - vis_lookback, split_index), train_raw[-vis_lookback:], label='Training Data (Tail)', color='blue')
    
    # 2. Plot True Test Data
    plt.plot(range(split_index, split_index + num_test_steps), test_raw, label='Actual Test Data', color='green', alpha=0.6)
    
    # 3. Plot Forecast
    plt.plot(range(split_index, split_index + num_test_steps), generated_values_denorm, label='TCN Forecast', color='red', linestyle='--')
    
    plt.axvline(x=split_index, color='black', linestyle=':', label='Train/Test Split')
    plt.title(f"TCN Forecast:(Train on {len(train_raw)}, Test on {len(test_raw)})")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    plot_filename = "perfect_split_forecast.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.show()

    # --- J. METRICS ---
    # Calculate MSE on the test set
    mse = np.mean((test_raw - generated_values_denorm)**2)
    print(f"\nTest Set MSE: {mse:.4f}")