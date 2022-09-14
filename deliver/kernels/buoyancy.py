# Buyonacy velocity rise Kernel

# Kernel to update the depth of a particle due to the buoyancy rise velocity. This process only occurs for particles under
# the surface. If the particles depth is below zero after the rise velocity is applied, it is automatically set to be at zero,
# the sea surface.

# Variables needed from environment: water density, water viscosity, acceleration due to gravity (g)
# Variables needed from oil: depth, entrainment time, minimum diameter, density
# Kernel updates: depth, entrainment time

def Buoyancy(particle, fieldset, time):

    if particle.depth > 0.001 and particle.entrainment_time != 0:

        r = particle.min_diameter
        particle.rise_velocity = 2 * fieldset.g * (1 - (particle.density/fieldset.density_water)) * r**2 / (9 * fieldset.viscosity_water)

        if (particle.rise_velocity * 2 * r / fieldset.viscosity_water) > 50:
            particle.rise_velocity = math.sqrt((16/3) * fieldset.g * (1 - (particle.density/fieldset.density_water)) * r)

        particle.depth = particle.depth - particle.rise_velocity * particle.dt

        if particle.depth < 0.001:
            particle.depth = 0
            particle.entrainment_time = 0
