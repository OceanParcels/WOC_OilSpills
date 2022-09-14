# Entrainment Kernel

# This kernel calculates the entrainment of oil particles. This is done by calculating an entrainment rate for the particle,
# which is then converted to a probability of the particle being entrained during time step dt. If a random number is lower than
# this probability, then the particle will be entrained. The depth to which the particles are entrained depends on the wind speed.
# This is a process which only occurs for particles at the surface, so only for particles with a depth of lower than z = 0.001 m.

# Variables needed from environment: wind speed, acceleration due to gravity (g), water density
# Variables needed from oil: depth, interfacial tension, density, viscosity
# Kernel updates: depth, entrainment time

def Entrainment(particle, fieldset, time):

    particle.U10 = fieldset.U10[particle]
    particle.V10 = fieldset.V10[particle]
    particle.wind_speed = math.sqrt(particle.U10**2 + particle.V10**2)

    if particle.depth < 0.001:

        d_0 = 4 * math.sqrt(particle.interfacial_tension / (fieldset.g*(fieldset.density_water - particle.density)))
        weber = fieldset.density_water * fieldset.g * 0.0246 * particle.wind_speed**2 * d_0 / particle.interfacial_tension
        ohns = (particle.viscosity * particle.density) / (math.sqrt(particle.density * particle.interfacial_tension * d_0))
        if particle.wind_speed > 5:
            T_p = 2*math.pi/(0.877 * fieldset.g * (1/(1.17*particle.wind_speed)))
            F_bw = 0.032 * (particle.wind_speed - 5)/T_p
        else:
            F_bw = 0

        entrainment_rate = 4.604e-10 * weber**1.805 * ohns**(-1.023) * F_bw
        if (entrainment_rate * particle.dt) < 0.01:
            entrainment_probability = entrainment_rate * particle.dt
        else:
            entrainment_probability = 1 - math.exp(-1 * entrainment_rate * particle.dt)

        if ParcelsRandom.random() < entrainment_probability:
            particle.depth = 1.5 * 0.0246 * particle.wind_speed**2

    elif particle.depth > 0.001 and particle.entrainment_time == 0:
        particle.entrainment_time += 1
