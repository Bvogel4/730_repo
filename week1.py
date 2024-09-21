# distance = 100
# effeciency = .1 #L/km
# energy_litre = 3e14 #ergs
# n_cars = 1.3e9
#
# #energy in 1 day
#
# energy = distance * effeciency * energy_litre * n_cars
# print(f'Energy in 1 day: {energy:.2e} ergs/day')
#
#
# luminosity_sun = 4e33
#
# #how many years to produce the same energy as the sun produces in 1 second
#
# energy_sun = luminosity_sun * 1
# energy_year = energy * 365
# years = energy_sun / energy_year
# print(f'Years to produce the same energy as the sun produces in 1 second: {years:.2e} years')
#

mass_loss_rate = 8e-14 #solar masses/year

#how much mass has it lost in 10^10 years
mass_loss = mass_loss_rate * 1e10
print(f'Mass lost in 10^10 years: {mass_loss:.1e} solar masses')




