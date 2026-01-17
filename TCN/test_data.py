# Generiere und zeige die Zahlenfolge für 14 Tage (48 Messungen/Tag).
# Annahme: Tag 1 = Montag. Der 2. Dienstag ist damit Tag 9 (index 8, 0-basiert).
# Ausgabe: 1D-Liste mit 672 Ganzzahlen (Anzahl parkender Autos, 0..50).
days = 14
per_day = 48

import math, random
random.seed(0)

vals = []
for day in range(days):
    is_weekend = (day%7 in (5,6))
    is_holiday = (day==8)  # zweiter Dienstag (Tag 9)
    for slot in range(per_day):
        hour = slot/2.0
        if hour < 6:
            base = 2 if not is_weekend else 1
        elif 6 <= hour < 9:
            x = (hour-6)/3.0
            base = 5 + 45*(1/(1+math.exp(-6*(x-0.5))))
        elif 9 <= hour < 12:
            base = 40
        elif 12 <= hour < 14:
            base = 30
        elif 14 <= hour < 17:
            base = 38
        elif 17 <= hour < 20:
            x = (hour-17)/3.0
            base = 38*(1 - 1/(1+math.exp(-6*(x-0.5))))
        else:
            base = 5
        if is_weekend:
            base = base * 0.45 + 3
        if is_holiday:
            base = base * 0.20 + 1
        val = int(round(max(0, min(50, base + random.uniform(-2,2)))))
        vals.append(val)

# Ausgabe als eine einzelne 1D-Liste (Zeilen zu je 48 Werten für Lesbarkeit)
lines = []
for d in range(days):
    day_vals = vals[d*per_day:(d+1)*per_day]
    lines.append(', '.join(str(x) for x in day_vals))

print("Annahmen: Tag 1 = Montag. Der 2. Dienstag ist Tag 9 (zweiter Dienstag, ganztägig Feiertag).")
print("Gesamtanzahl Werte:", len(vals))
print("\n--- Werte (je Zeile = 1 Tag, 48 Messungen/Tag) ---\n")
for i, line in enumerate(lines, start=1):
    print(f"Tag {i}: {line}")

# Und zum einfachen Kopieren: komplette 1D-Liste in einer Zeile
print("\n--- Komplette 1D-Liste ---")
print("[" + ", ".join(str(x) for x in vals) + "]")

