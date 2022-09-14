# Emulsification kernel

# This kernel emulsifies the oil particles. This is done by calculating the water fraction using the interfacial area
# between the oil and the water.
# This is a process that only occurs at the surface, so only acts on particles with a depth lower than (z = 0.001 m).
# Furthermore, the process of emulsification only begins once a certain fraction of the oil has been evaporated, known as
# the emulsification onset percentage. This is known for some oils, but for a lot of oils it is unknown. If it is unknown, it can be set to 0, or approximated using a similar
# oil. There is also a maximum water fraction that can be reached, which cannot be higher than 90%.

# Variables needed from environment: wind speed
# Variables needed from oil: depth, fraction evaporated, emulsification onset percentage, max water fraction, minimum droplet
#                            diameter, maximum droplet diameter
# Kernel updates: water fraction

def Emulsification(particle, fieldset, time):

    if particle.depth < 0.001 and particle.fraction_evaporated > particle.emulsification_onset_percentage:

        max_interfacial_area = (6 / particle.d_min) * (particle.max_water_fraction / (1 - particle.max_water_fraction))

        k_emul = particle.wind_speed**2 * (6 * 2.02e-6)/(1e-5)

        particle.interfacial_area += particle.dt * (k_emul * (1 - (particle.interfacial_area/max_interfacial_area)))
        if particle.interfacial_area > max_interfacial_area:
            particle.interfacial_area = max_interfacial_area

        particle.water_fraction = particle.interfacial_area * particle.d_max / (6 + (particle.interfacial_area * particle.d_max))
        if particle.water_fraction > particle.max_water_fraction:
            particle.water_fraction = particle.max_water_fraction
