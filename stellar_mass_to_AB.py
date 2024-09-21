import numpy as np


def estimate_absolute_magnitude(stellar_mass, mass_to_light_ratio=1.0):
    """
    Estimate the absolute magnitude of a galaxy from its stellar mass.

    Parameters:
    stellar_mass (float): Stellar mass of the galaxy in solar masses
    mass_to_light_ratio (float): Mass-to-light ratio in solar units (default: 1.0)

    Returns:
    float: Estimated absolute magnitude
    """
    # Convert stellar mass to luminosity
    luminosity = stellar_mass / mass_to_light_ratio

    # Calculate absolute magnitude using the luminosity
    # We use the Sun's absolute magnitude in the V-band as reference (M_sun = 4.83)
    M_sun = 4.83
    absolute_magnitude = M_sun - 2.5 * np.log10(luminosity)

    return absolute_magnitude


# Example usage
if __name__ == "__main__":
    # Example stellar masses (in solar masses)
    galaxy_masses = np.logspace(6,10,5)

    for mass in galaxy_masses:
        abs_mag = estimate_absolute_magnitude(mass)
        print(f"Galaxy with stellar mass {mass:.2e} M_sun:")
        print(f"Estimated absolute magnitude: {abs_mag:.2f}")
        print()