import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tcn import TCN

data_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
data = np.array(data_list)
slidWind_Length = 3  # Input-Fenstergröße (Rezeptives Feld)

X = []
y = []

# Erstellung der Sequenzen (Sliding Window)
for i in range(len(data) - slidWind_Length):
    # Features (X): die vorherigen 'seq_length' Werte
    X.append(data[i:i + slidWind_Length])
    # Label (y): der unmittelbar nächste Wert
    y.append(data[i + slidWind_Length])

X = np.array(X)
y = np.array(y)

# TCN Input-Shape: (Samples, Timesteps, Features)
# Die aktuelle Form ist (Samples, Timesteps). Wir brauchen (Samples, Timesteps, 1) für univariante Daten.
X = X.reshape(X.shape[0], X.shape[1], 1)

print(f"Original-Datenlänge: {len(data_list)}")
print(f"X-Shape (Samples, Timesteps, Features): {X.shape}")
print(f"Erste X-Sequenz: {X[0].flatten()} -> Label y: {y[0]}")
# Beispiel-Output bei seq_length=3: [0, 1, 2] -> 3


# Modellparameter
input_dim = 1  # Anzahl der Features pro Zeitschritt (univariant)
timesteps = slidWind_Length  # Die Sequenzlänge (3 in diesem Beispiel)
nb_filters = 64  # Anzahl der Filter in den Faltungs-Layern
kernel_size = 3  # Kernelgröße der 1D-Faltung
dilations = [1, 2, 4]  # Dilatationsraten (werden gestapelt)
padding = 'causal' # Wichtig für Zeitreihen: Das Output-Element hängt nur von den vorherigen/aktuellen Input-Elementen ab

model = Sequential([
    # TCN-Layer
    TCN(
        input_shape=(timesteps, input_dim),
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        dilations=dilations,
        padding=padding,
        return_sequences=False  # Nur den Output des letzten Zeitschritts zurückgeben
    ),
    # Dense-Layer für die finale Ausgabe (z.B. die Vorhersage des nächsten Werts)
    Dense(1)
])

model.summary()
model.compile(optimizer='adam', loss='mse')
# Training des Modells
epochs = 50
model.fit(X, y, epochs=epochs, verbose=0)
print(f"Modell trainiert für {epochs} Epochen.")

# Neueste Sequenz aus den Originaldaten als Input
last_sequence = data[-slidWind_Length:]
# Umformen auf die benötigte Shape (1, timesteps, features)
X_new = last_sequence.reshape(1, slidWind_Length, 1)

# Vorhersage
prediction = model.predict(X_new)

print("-" * 30)
print(f"Letzte Input-Sequenz: {last_sequence}")
print(f"Vorhersage für den nächsten Wert: {prediction.flatten()[0]:.2f}")
print(f"Tatsächlicher nächster Wert (falls bekannt): 11 (nicht in den Trainingsdaten)")