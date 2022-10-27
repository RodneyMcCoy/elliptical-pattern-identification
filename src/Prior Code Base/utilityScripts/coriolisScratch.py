import metpy.calc as mpcalc
from metpy.units import units

lat = 90 * units.degrees

f = mpcalc.coriolis_parameter(lat)
f_period = 1/f

print("f: ", f)
print("f period: ", f_period)