import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
import numpy as np 
import matplotlib.pyplot as plt # Für das Plotten der Ergebnisse

# NEUE IMPORTS FÜR MINI-BATCHING
from torch.utils.data import TensorDataset, DataLoader

# --- 1. Causal 1D Convolution Helper ---
# ... (Keine Änderungen hier) ...
class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, **kwargs):
        self.padding_size = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, **kwargs)
    def forward(self, x):
        x = nn.functional.pad(x, (self.padding_size, 0))
        return super().forward(x)

# --- 2. TCN Residual Block ---
# ... (Keine Änderungen hier) ...
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
# ... (Keine Änderungen hier) ...
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
    
# --- 4. Training Function (MODIFIZIERT FÜR MINI-BATCHING) ---
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    """
    Trainiert das TCN-Modell mit Mini-Batching.
    """
    print("\n--- Starte Training (mit Mini-Batches) ---")
    
    model.train() # Modell in den Trainingsmodus versetzen

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        
        # Iteriere über die Mini-Batches
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad() 
            
            output = model(batch_X) # Shape [Batch_Size, 1]
            loss = criterion(output, batch_y) # Vergleicht [Batch_Size, 1]
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Berechne den durchschnittlichen Verlust der Epoche
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        if epoch % 10 == 0 or epoch == num_epochs:
            print(f"Epoche [{epoch}/{num_epochs}], Avg. Loss: {avg_epoch_loss:.6f}")

    print("--- Training Abgeschlossen ---")

# --- 5. Sequential Data Preparation Function ---
# ... (Keine Änderungen hier) ...
def create_sequential_data(long_series, lookback_window):
    X, y = [], []
    total_length = long_series.shape[0]
    for i in range(total_length - lookback_window):
        x_segment = long_series[i: i + lookback_window]
        y_target = long_series[i + lookback_window]
        X.append(x_segment)
        y.append(y_target)
    X_tensor = torch.stack(X)
    y_tensor = torch.stack(y) 
    X_tensor = X_tensor.permute(0, 2, 1) # -> (N, C, L)
    print(f"Sequentielle Samples (N): {X_tensor.shape[0]}")
    return X_tensor, y_tensor


# --- 6. Autoregressive Forecasting Function ---
# ... (Keine Änderungen hier) ...
def forecast_autoregressive(model, start_sequence, forecast_steps):
    model.eval()
    generated_values = []
    current_window = start_sequence.clone() # Shape (L, C)
    with torch.no_grad():
        for _ in range(forecast_steps):
            input_tensor = current_window.permute(1, 0).unsqueeze(0) 
            prediction = model(input_tensor) # Shape (1, 1)
            generated_values.append(prediction.item())
            new_value_tensor = prediction.view(1, 1) 
            current_window = torch.cat((current_window[1:], new_value_tensor), dim=0)
    return generated_values


if __name__ == '__main__':
    # --- Hyperparameter ---
    input_channels = 1      
    lookback_window = 100   
    output_size = 1         
    kernel_size = 3         
    dropout_rate = 0.1
    learning_rate = 0.005
    num_epochs = 200            
    num_channels_list = [32] * 9 # 9 Lagen
    
    BATCH_SIZE = 128 # Kleinere Zahl, z.B. 32, 64, 128
    one_cycle =  [3, 3, 2, 1, 2, 2, 3, 1, 2, 2, 4, 2, 6, 11, 18, 27, 40, 47, 41, 42, 39, 41, 42, 41, 30, 28, 30, 30, 40, 40, 38, 39, 37, 39, 36, 32, 29, 19, 12, 5, 3, 5, 6, 4, 4, 6, 4, 5, 1, 4, 3, 2, 0, 1, 2, 4, 0, 2, 3, 2, 8, 11, 19, 28, 38, 44, 40, 40, 40, 39, 39, 39, 30, 31, 30, 28, 39, 40, 40, 39, 40, 40, 36, 33, 29, 18, 11, 6, 7, 5, 7, 5, 5, 6, 7, 7, 3, 0, 2, 2, 3, 3, 1, 3, 0, 1, 3, 1, 8, 9, 16, 28, 36, 45, 42, 40, 41, 38, 41, 40, 30, 30, 29, 32, 36, 36, 40, 37, 36, 37, 37, 35, 26, 19, 9, 4, 4, 6, 4, 4, 5, 3, 3, 7, 1, 1, 3, 3, 4, 1, 3, 4, 0, 3, 3, 1, 6, 11, 17, 26, 38, 44, 40, 40, 39, 39, 41, 39, 30, 28, 31, 29, 36, 37, 37, 40, 37, 37, 36, 35, 28, 19, 11, 4, 5, 6, 3, 4, 4, 4, 6, 5, 4, 2, 2, 2, 3, 2, 3, 0, 2, 1, 1, 2, 7, 11, 18, 29, 38, 44, 40, 41, 42, 41, 39, 42, 31, 28, 31, 32, 38, 39, 37, 37, 39, 36, 37, 34, 26, 17, 11, 6, 4, 7, 3, 6, 5, 3, 4, 5, 5, 5, 2, 4, 4, 3, 5, 3, 2, 2, 4, 4, 8, 9, 12, 14, 21, 24, 22, 22, 23, 22, 20, 21, 15, 17, 15, 15, 19, 21, 21, 19, 21, 18, 18, 20, 15, 11, 8, 4, 5, 4, 3, 4, 5, 4, 5, 5, 3, 2, 5, 3, 5, 5, 3, 3, 4, 3, 3, 4, 8, 9, 11, 15, 22, 24, 19, 19, 22, 19, 19, 21, 17, 16, 16, 17, 21, 20, 20, 22, 19, 21, 19, 17, 14, 13, 8, 7, 5, 6, 4, 5, 4, 3, 7, 4, 1, 1, 2, 2, 1, 4, 4, 1, 2, 3, 4, 2, 9, 10, 17, 27, 39, 43, 40, 39, 39, 42, 42, 41, 28, 29, 29, 31, 39, 38, 39, 39, 36, 36, 38, 32, 27, 17, 8, 6, 4, 4, 4, 6, 7, 5, 7, 5, 2, 2, 3, 2, 3, 2, 3, 0, 2, 2, 3, 1, 4, 3, 5, 8, 10, 11, 7, 8, 9, 8, 7, 10, 9, 6, 7, 8, 7, 9, 7, 8, 10, 7, 6, 7, 7, 6, 4, 3, 1, 4, 2, 2, 2, 0, 2, 1, 0, 2, 3, 0, 2, 1, 2, 3, 2, 4, 2, 2, 8, 10, 16, 28, 38, 46, 39, 42, 41, 42, 41, 41, 31, 30, 31, 29, 39, 38, 39, 38, 39, 36, 34, 35, 28, 21, 12, 5, 5, 4, 3, 6, 7, 6, 6, 5, 1, 0, 2, 2, 4, 4, 2, 0, 3, 3, 0, 2, 8, 8, 19, 29, 39, 43, 38, 39, 38, 39, 38, 42, 31, 28, 32, 29, 37, 39, 40, 36, 36, 37, 34, 34, 29, 18, 9, 4, 7, 4, 7, 4, 5, 4, 7, 7, 3, 1, 4, 0, 2, 1, 4, 4, 3, 1, 1, 3, 6, 10, 15, 26, 36, 43, 38, 40, 40, 41, 39, 40, 31, 28, 30, 29, 40, 36, 38, 38, 39, 37, 35, 33, 28, 21, 11, 4, 5, 6, 6, 5, 5, 6, 5, 4, 3, 5, 2, 3, 4, 3, 2, 1, 4, 4, 1, 3, 7, 8, 11, 14, 21, 23, 22, 22, 20, 22, 20, 23, 15, 18, 15, 16, 20, 20, 20, 19, 19, 19, 20, 20, 16, 10, 9, 4, 5, 5, 4, 7, 4, 5, 4, 5, 4, 4, 4, 3, 4, 3, 5, 5, 3, 3, 4, 2, 7, 7, 11, 14, 21, 23, 21, 21, 21, 22, 20, 22, 15, 16, 17, 15, 21, 18, 20, 22, 18, 20, 18, 20, 16, 13, 8, 3, 3, 6, 7, 7, 6, 6, 4, 6]
    MIN_VAL = max(one_cycle)
    MAX_VAL = min(one_cycle)
    SERIES_LENGTH = len(one_cycle)    

    cycle_len = len(one_cycle)
    num_repeats = int(np.ceil(SERIES_LENGTH / cycle_len))
    raw_series = np.tile(one_cycle, num_repeats)[:SERIES_LENGTH]
    print(f"Rohe Serie (Ausschnitt): {raw_series[:5]}... {raw_series[498:503]}... {raw_series[996:1001]}")
    normalized_series = (raw_series - MIN_VAL) / (MAX_VAL - MIN_VAL)
    long_series = torch.from_numpy(normalized_series).unsqueeze(1).float()
    
    # --- 2. Erstelle Sequentielle Input/Target Paare ---
    X_train, y_train = create_sequential_data(long_series, lookback_window)
    
    # --- 3. Instanziiere Modell und richte Training ein ---
    model = TCN(
        input_size=input_channels, 
        output_size=output_size, 
        num_channels=num_channels_list, 
        kernel_size=kernel_size, 
        dropout=dropout_rate
    )
    
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # --- 4. Führe Trainingsschleife aus (MODIFIZIERT) ---
    
    # Erstelle TensorDataset und DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Input Batch Shape (N, C, L): {X_train.shape}")
    print(f"Target Batch Shape (N, Output_Size): {y_train.shape}")
    print(f"Verwende Mini-Batches der Größe: {BATCH_SIZE}")
    
    # Übergebe den LOADER an die Trainingsfunktion
    train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)
    
    # --- 5. Autoregressive Prognose ---
    FORECAST_STEPS = 200 
    last_window_from_data = long_series[-lookback_window:] 
    print(f"\n--- Starte Autoregressive Prognose für {FORECAST_STEPS} Schritte ---")
    generated_values_norm = forecast_autoregressive(model, last_window_from_data, FORECAST_STEPS)

    # --- 6. Ergebnisse Denormalisieren und Kombinieren ---
    original_series_denorm = (long_series.squeeze().numpy() * (MAX_VAL - MIN_VAL)) + MIN_VAL
    generated_series_denorm = (np.array(generated_values_norm) * (MAX_VAL - MIN_VAL)) + MIN_VAL
    
    print("\n--- Ergebnis ---")
    print(f"Letzte 5 Werte der Originalserie (Input):")
    print(original_series_denorm[-5:])
    print("\nErste 5 Werte der Prognose (Output):")
    print(generated_series_denorm[:5])
    
    # --- 7. Plotten der Ergebnisse ---
    print("\n--- Erstelle Plot (tcn_forecast_plot_pattern.png) ---")
    plt.figure(figsize=(15, 6))
    plt.plot(original_series_denorm, label="Originaldaten (Input)", color='blue')
    forecast_x_axis = range(SERIES_LENGTH, SERIES_LENGTH + FORECAST_STEPS)
    plt.plot(forecast_x_axis, generated_series_denorm, label="Prognose (Output)", color='red', linestyle='--')
    plt.title(f"TCN Autoregressive Prognose (Sägezahn-Muster, 9 Lagen)")
    plt.xlabel("Zeitschritt")
    plt.ylabel("Wert")
    plt.legend()
    plt.grid(True)
    plt.xlim(SERIES_LENGTH - 500, SERIES_LENGTH + FORECAST_STEPS)
    try:
        plt.savefig("tcn_forecast_plot_pattern.png")
        print("Plot erfolgreich in 'tcn_forecast_plot_pattern.png' gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern des Plots: {e}")