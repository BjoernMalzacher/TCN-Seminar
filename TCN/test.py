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
    lookback_window = 800   
    output_size = 1         
    kernel_size = 5        
    dropout_rate = 0.1
    learning_rate = 0.005
    num_epochs = 200            
    num_channels_list = [32] * 7 
    #num_channels_list = [1,2,4,8,16,32,64,128]  # 9 Lagen
    #num_channels_list = [32,64,128,256,512]  # 9 Lagen
    BATCH_SIZE =128
    
    # *** NEUER PARAMETER ***
    NUM_TEST_STEPS = 100 # Die letzten 100 Werte für den Vergleich
    with open("parkende_autos_30tage_schule_kirche_restaurant_48protag.txt", "r") as f:
        content = f.read()

# In Liste umwandeln
    numbers = [int(x.strip()) for x in content.split(",") if x.strip() != ""]
    
    mid = len(numbers) // 2

    # Split into two halves
    first_half = numbers[:mid]
    second_half = numbers[mid:]
    one_cycle = first_half
    # Korrekte Normalisierungswerte
    REAL_MIN_VAL = min(one_cycle)
    REAL_MAX_VAL = max(one_cycle)

    SERIES_LENGTH = len(one_cycle)    

    cycle_len = len(one_cycle)
    num_repeats = int(np.ceil(SERIES_LENGTH / cycle_len))
    raw_series = np.tile(one_cycle, num_repeats)[:SERIES_LENGTH]
    print(f"Rohe Serie (Ausschnitt): {raw_series[:5]}... {raw_series[498:503]}... {raw_series[996:1001]}")
    
    # Verwende korrekte Min/Max-Werte für die Normalisierung
    normalized_series = (raw_series - REAL_MIN_VAL) / (REAL_MAX_VAL - REAL_MIN_VAL)
    long_series = torch.from_numpy(normalized_series).unsqueeze(1).float()
    
    # --- 1.5. Aufteilen in Training- und Test-Serie (MODIFIZIERT) ---
    # Wir trainieren auf allem BIS auf die letzten 100 Schritte
    train_series = long_series[:-NUM_TEST_STEPS]
    
    # --- 2. Erstelle Sequentielle Input/Target Paare (MODIFIZIERT) ---
    # Erstelle Sequenzen NUR aus der Trainings-Serie
    print("Erstelle Trainings-Sequenzen...")
    X_train, y_train = create_sequential_data(train_series, lookback_window)
    
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
    
    # --- 4. Führe Trainingsschleife aus ---
    
    # Erstelle TensorDataset und DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Input Batch Shape (N, C, L): {X_train.shape}")
    print(f"Target Batch Shape (N, Output_Size): {y_train.shape}")
    print(f"Verwende Mini-Batches der Größe: {BATCH_SIZE}")
    
    # Übergebe den LOADER an die Trainingsfunktion
    train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)
    
    # --- 5. Autoregressive Prognose (MODIFIZIERT) ---
    # Nimm das letzte Fenster VOR dem Test-Set als Seed
    # Dieses Fenster ist: long_series[-(NUM_TEST_STEPS + lookback_window) : -NUM_TEST_STEPS]
    
    
    with open("parking_data.txt", "r") as f:
        content = f.read()

# In Liste umwandeln
    numbers = [int(x.strip()) for x in content.split(",") if x.strip() != ""]
    one_cycle = numbers;
    
    # Korrekte Normalisierungswerte
    REAL_MIN_VAL = min(one_cycle)
    REAL_MAX_VAL = max(one_cycle)

    SERIES_LENGTH = len(one_cycle)    

    cycle_len = len(one_cycle)
    num_repeats = int(np.ceil(SERIES_LENGTH / cycle_len))
    raw_series = np.tile(one_cycle, num_repeats)[:SERIES_LENGTH]
    print(f"Rohe Serie (Ausschnitt): {raw_series[:5]}... {raw_series[498:503]}... {raw_series[996:1001]}")
    
    # Verwende korrekte Min/Max-Werte für die Normalisierung
    normalized_series = (raw_series - REAL_MIN_VAL) / (REAL_MAX_VAL - REAL_MIN_VAL)
    long_series = torch.from_numpy(normalized_series).unsqueeze(1).float()
    forecast_seed = long_series[-(NUM_TEST_STEPS + lookback_window) : -NUM_TEST_STEPS]
    
    print(f"\n--- Starte Autoregressive Prognose für {NUM_TEST_STEPS} Test-Schritte ---")
    generated_values_norm = forecast_autoregressive(model, forecast_seed, NUM_TEST_STEPS)

    # --- 6. Ergebnisse Denormalisieren und Kombinieren (MODIFIZIERT) ---
    
    # Denormalisiere die *gesamte* Originalserie für den Plot
    original_series_denorm = (long_series.squeeze().numpy() * (REAL_MAX_VAL - REAL_MIN_VAL)) + REAL_MIN_VAL
    
    # Denormalisiere die *vorhergesagten* Werte
    generated_series_denorm = (np.array(generated_values_norm) * (REAL_MAX_VAL - REAL_MIN_VAL)) + REAL_MIN_VAL
    
    # Hol dir die *echten* Test-Werte zum Vergleichen
    actual_test_values_denorm = original_series_denorm[-NUM_TEST_STEPS:]
    
    print("\n--- Ergebnis (Vergleich) ---")
    print(f"Letzte 5 'echte' Werte (Test-Set):")
    print(actual_test_values_denorm[-5:])
    print("\nLetzte 5 'vorhergesagte' Werte (Test-Set):")
    print(generated_series_denorm[-5:])
    
    # --- 7. Plotten der Ergebnisse (MODIFIZIERT) ---
    print("\n--- Erstelle Plot (tcn_forecast_plot_vergleich.png) ---")
    plt.figure(figsize=(15, 6))
    
    # Plotte die gesamten Originaldaten
    plt.plot(original_series_denorm, label="Original Data (Train + Test)", color='blue', alpha=0.7)
    
    # Definiere die X-Achse für die Prognose (die letzten 100 Schritte)
    forecast_x_axis = range(SERIES_LENGTH - NUM_TEST_STEPS, SERIES_LENGTH)
    
    # Plotte die Prognose ÜBER die echten Daten
    plt.plot(forecast_x_axis, generated_series_denorm, label=f"forcast (last {NUM_TEST_STEPS})", color='red', linestyle='--')
    
    # Markiere den Start der Prognose
    plt.axvline(x=SERIES_LENGTH - NUM_TEST_STEPS, color='green', linestyle=':', label='Start of forecast')
    
    plt.title(f"TCN forecast (last {NUM_TEST_STEPS} steps) vs. syntatic data")
    plt.xlabel("30-minute time steps")
    plt.ylabel("loading car count")
    plt.legend()
    plt.grid(True)
    
    # Zoome auf den relevanten Bereich: 200 Schritte vor dem Test + das Test-Set
    plt.xlim(SERIES_LENGTH - NUM_TEST_STEPS - 200, SERIES_LENGTH + 5) 
    
    try:
        plt.savefig("tcn_forecast_plot_vergleich.png") # Neuer Dateiname
        print("Plot erfolgreich in 'tcn_forecast_plot_vergleich.png' gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern des Plots: {e}")