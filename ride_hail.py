import math

drivers = [
    ("D1", 12, 15),
    ("D2", 5, 8),
    ("D3", 20, 10)
]

passenger = (10, 12)

best_driver = None
best_dist = float('inf')

for name, x, y in drivers:
    d = math.sqrt((x-passenger[0])**2 + (y-passenger[1])**2)

    if d < best_dist:
        best_dist = d
        best_driver = name

print("Assigned Driver:", best_driver)
