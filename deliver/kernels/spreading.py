# Spreading kernel

# This kernel is used to update the area of the spill, which is subsequently used in the evaporation kernel. This
# kernel also updates the particle age, the length of time the particle has been at the surface since the initial release
# of the particle.
# This is a process that only occurs at the surface, so only acts on particles with a depth lower than (z = 0.001 m).

# Variables needed from environment: water density, water viscosity, acceleration due to gravity (g)
# Variables needed from oil: depth, oil density, initial spill volume, particle age, max area
# Kernel updates: area, age

def Spreading(particle, fieldset, time):

    if particle.depth < 0.001:

        relative_buoyancy = (fieldset.density_water - particle.density)/fieldset.density_water
        t_0 = ((1.21/1.53)**4 * particle.volume_initial/(fieldset.viscosity_water * fieldset.g * relative_buoyancy))**(1/3)

        if particle.age < t_0 or particle.age == 0:
            particle.area = math.pi * (1.21**4)/(1.53**2) * ((particle.volume_initial**5 * fieldset.g * relative_buoyancy)/(fieldset.viscosity_water))**(1/6)

        else:
            particle.area = math.pi * 1.45**2 * (particle.volume_initial**2 * fieldset.g * relative_buoyancy/(math.sqrt(fieldset.viscosity_water)))**(1/3) * (1/math.sqrt(particle.age))

        if particle.area > particle.max_area:
            particle.area = particle.max_area

        particle.age += particle.dt
