import matplotlib.pyplot as plt
import numpy as np

def load_series(filename):
    """Liest eine kommagetrennte Textdatei ein und gibt eine Liste von Zahlen zurück."""
    try:
        with open(filename, 'r') as f:
            content = f.read()
            # Zeilenumbrüche entfernen und am Komma trennen
            content = content.replace('\n', ' ')
            # In Integers umwandeln
            data = [int(x.strip()) for x in content.split(',') if x.strip().isdigit()]
        return np.array(data)
    except FileNotFoundError:
        print(f"Fehler: Die Datei '{filename}' wurde nicht gefunden.")
        return np.array([])

# Dateinamen (Stelle sicher, dass diese Dateien im selben Ordner liegen)
file_1 = 'test.txt'
file_2 = 'parkende_autos_30tage_schule_kirche_restaurant_48protag.txt'

# Daten laden
series_1 = load_series(file_1)
series_2 = load_series(file_2)

if len(series_1) > 0 and len(series_2) > 0:
    # Plot erstellen
    plt.figure(figsize=(15, 6))
    
    # Wir zoomen standardmäßig auf die ersten 7 Tage (7 * 48 Messpunkte), 
    # damit man Details sieht. Nimm '[:]' statt '[:limit]', um alles zu sehen.
    limit = 30 * 48 
    
    # Erste Zeitreihe plotten
    plt.plot(series_1[:limit], label='Synthetisch 1 (Rauschen & Skalierung)', color='blue', alpha=0.7)
    
    # Zweite Zeitreihe plotten
    plt.plot(series_2[:limit], label='Synthetisch 2 (Verzerrung & Varianz)', color='green', alpha=0.7, linestyle='--')
    
    # Beschriftungen
    plt.title('Vergleich der synthetischen Zeitreihen (Ausschnitt erste 7 Tage)')
    plt.xlabel('Zeit (in 30-Minuten-Schritten)')
    plt.ylabel('Anzahl parkende Autos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Anzeigen
    plt.tight_layout()
    plt.show()
else:
    print("Konnte keine Daten zum Plotten finden.")