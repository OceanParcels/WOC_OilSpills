# Update Properties Kernel

# This is a kernel to update the properties of the oil, namely the viscosity and density.

# Variables needed from environment: water density
# Variables needed from oil: density, viscosity, water fraction, fraction evaporated, original density, original viscosity
# Kernel updates: density, viscosity

def Update_oil_properties(particle, fieldset, time):

    kv1 = math.sqrt(particle.viscosity) * 1.5e3
    if kv1 > 10:
        kv1 = 10
    elif kv1 < 1:
        kv1 = 1

    particle.density = particle.water_fraction * fieldset.density_water + (1 - particle.water_fraction) * particle.original_density

    particle.viscosity = particle.original_viscosity * math.exp(kv1 * particle.fraction_evaporated) * (1 + (particle.water_fraction/0.84)/(1.187 - (particle.water_fraction/0.84)))**2.49
