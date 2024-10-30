

phi = .007
m_sun = 1.898e33
c = 2.998e10
L_sun = 3.846e33



t_nuc = m_sun*.007*.13*.7*c**2 / L_sun
print(t_nuc)


print(f'The time it takes for the sun to burn 7% of its mass is {t_nuc:.2e} seconds')
#years
t_nuc = t_nuc/(60*60*24*365)
print(f'The time it takes for the sun to burn 7% of its mass is {t_nuc:.5} years')

import numpy as np

