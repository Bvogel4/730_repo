import math

# Constants
G = 4 * math.pi**2  # AU^3 / (M_sun * year^2)
M = 1  # Total mass of the system in solar masses (1 + 1)
r = 10  # Orbital radius in AU

# Calculate orbital velocity
v = math.sqrt(G * M /(4 *r))

T = 2 * math.pi * r/ v

print(f"The orbital velocity is {v} AU/year")
print(f"The orbital period is {T} years")