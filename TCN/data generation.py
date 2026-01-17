import numpy as np
import random

def generate_parking_data():
    # --- Configuration ---
    days = 365
    intervals_per_day = 48  # 30-minute intervals
    total_steps = days * intervals_per_day
    max_capacity = 50
    
    # Initialize array (Time steps)
    data = np.zeros(total_steps)
    
    # --- 1. Base Noise (Random variations) ---
    # A few random cars are always parking or leaving
    noise = np.random.normal(loc=2, scale=2, size=total_steps)
    data += noise

    # --- 2. Restaurant Logic ---
    # Opens 17:00 (Index 34), Capacity ~30. 
    # We model a curve peaking around 19:00 (Index 38) and dropping by 23:00
    restaurant_curve = [0, 5, 15, 25, 28, 30, 25, 15, 5, 0] # 5 hours duration approx
    rest_start_index = 34 # 17:00
    
    for d in range(days):
        day_offset = d * intervals_per_day
        # Add restaurant pattern every day
        for i, val in enumerate(restaurant_curve):
            if rest_start_index + i < intervals_per_day:
                data[day_offset + rest_start_index + i] += val + np.random.randint(-3, 3)

    # --- 3. School Logic ---
    # Weekdays only. Morning drop-off (07:30-08:30) and Pickup (13:00-14:30)
    for d in range(days):
        day_offset = d * intervals_per_day
        weekday = d % 7 # 0=Monday, 6=Sunday
        
        if weekday < 5: # Mon-Fri
            # Morning Spike (Index 15 = 07:30)
            data[day_offset + 15] += np.random.randint(10, 15)
            data[day_offset + 16] += np.random.randint(5, 10)
            
            # Afternoon Spike (Index 26 = 13:00)
            data[day_offset + 26] += np.random.randint(10, 18)
            data[day_offset + 27] += np.random.randint(8, 12)

    # --- 4. Church Logic ---
    # Randomly selected times, irregular spikes.
    # Let's assume ~50 random services a year.
    num_services = 50
    service_days = np.random.choice(range(days), num_services, replace=False)
    
    for d in service_days:
        day_offset = d * intervals_per_day
        # Service time random between 09:00 (18) and 20:00 (40)
        start_time = np.random.randint(18, 40)
        duration = 4 # 2 hours
        intensity = np.random.randint(20, 40)
        
        for i in range(duration):
            if start_time + i < intervals_per_day:
                data[day_offset + start_time + i] += intensity

    # --- 5. Three Random Holidays ---
    # Deviate substantially. We will flatten the usual pattern and add a weird block.
    holidays = np.random.choice(range(days), 3, replace=False)
    
    for h in holidays:
        day_offset = h * intervals_per_day
        # Clear existing data for this day
        data[day_offset : day_offset + intervals_per_day] = 0
        
        # Add "Holiday Pattern" (e.g., consistent high usage all afternoon)
        data[day_offset + 20 : day_offset + 40] = np.random.randint(35, 45, size=20)

    # --- Final Cleanup ---
    # Round to integers
    data = np.round(data)
    # Clip to be between 0 and 50 (Physical constraints)
    data = np.clip(data, 0, max_capacity)
    
    return data.astype(int)

# Generate and Save
parking_data = generate_parking_data()

# Save as comma separated list
output_string = ", ".join(map(str, parking_data))

with open("parking_data.txt", "w") as f:
    f.write(output_string)

print(f"Generated {len(parking_data)} data points.")
print("First 48 points (Day 1):")
print(output_string[:200] + "...")