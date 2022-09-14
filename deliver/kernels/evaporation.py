# Evaporation kernel

# This kernel calculates the fraction of oil that is evaporated. For this, the different mass components are required.
# The oil is split up into the different components and evaporation occurs on each component individually. The total fraction
# evaporated is then calculated as the sum of the mass which is evaporated from each component. The different mass components
# and their mass, vapour pressures, and molecular weights can be found for different oil types from the NOAA oil database. If no data
# on a specific oil is known, it can be approximated using a similar oil.
# This is a process that only occurs at the surface, so only acts on particles with a depth lower than (z = 0.001 m).

# Variables needed from environment: wind speed,
# Variables needed from oil: depth, mass of oil components, molecular weight of oil components, vapour pressure of oil
#                            components, area, initial mass
# Kernel updates: mass, fraction evaporated

def Evaporation(particle, fieldset, time):

    particle.U10 = fieldset.U10[particle]
    particle.V10 = fieldset.V10[particle]
    particle.wind_speed = math.sqrt(particle.U10**2 + particle.V10**2)

    if particle.depth < 0.001:

        # summed mass components
        summed_mass_components = (particle.mass_cut_1/particle.molecular_weight_cut_1) + (particle.mass_cut_2/particle.molecular_weight_cut_2) + (particle.mass_cut_3/particle.molecular_weight_cut_3) + (particle.mass_cut_4/particle.molecular_weight_cut_4) + (particle.mass_cut_5/particle.molecular_weight_cut_5) + (particle.mass_cut_6/particle.molecular_weight_cut_6) + (particle.mass_cut_7/particle.molecular_weight_cut_7) + (particle.mass_cut_8/particle.molecular_weight_cut_8) + (particle.mass_cut_9/particle.molecular_weight_cut_9) + (particle.mass_cut_10/particle.molecular_weight_cut_10) + (particle.mass_cut_11/particle.molecular_weight_cut_11) + (particle.mass_cut_12/particle.molecular_weight_cut_12) + (particle.mass_cut_13/particle.molecular_weight_cut_13) + (particle.mass_cut_14/particle.molecular_weight_cut_14) + (particle.mass_cut_15/particle.molecular_weight_cut_15) + (particle.mass_cut_16/particle.molecular_weight_cut_16) + (particle.mass_cut_17/particle.molecular_weight_cut_17) + (particle.mass_cut_18/particle.molecular_weight_cut_18) + (particle.mass_cut_19/particle.molecular_weight_cut_19) + (particle.mass_cut_20/particle.molecular_weight_cut_20)

        # Mass Transport Coefficient, K
        if abs(particle.wind_speed) < 10:
            K = 0.0025 * particle.wind_speed**0.78
        else:
            K = 0.06 * particle.wind_speed**2

        particle.mass_cut_1 = particle.mass_cut_1 * math.exp(-1 * particle.dt * (particle.area * K * particle.vapor_pressure_cut_1)/(fieldset.R * fieldset.sea_water_temperature * summed_mass_components))
        particle.mass_cut_2 = particle.mass_cut_2 * math.exp(-1 * particle.dt * (particle.area * K * particle.vapor_pressure_cut_2)/(fieldset.R * fieldset.sea_water_temperature * summed_mass_components))
        particle.mass_cut_3 = particle.mass_cut_3 * math.exp(-1 * particle.dt * (particle.area * K * particle.vapor_pressure_cut_3)/(fieldset.R * fieldset.sea_water_temperature * summed_mass_components))
        particle.mass_cut_4 = particle.mass_cut_4 * math.exp(-1 * particle.dt * (particle.area * K * particle.vapor_pressure_cut_4)/(fieldset.R * fieldset.sea_water_temperature * summed_mass_components))
        particle.mass_cut_5 = particle.mass_cut_5 * math.exp(-1 * particle.dt * (particle.area * K * particle.vapor_pressure_cut_5)/(fieldset.R * fieldset.sea_water_temperature * summed_mass_components))
        particle.mass_cut_6 = particle.mass_cut_6 * math.exp(-1 * particle.dt * (particle.area * K * particle.vapor_pressure_cut_6)/(fieldset.R * fieldset.sea_water_temperature * summed_mass_components))
        particle.mass_cut_7 = particle.mass_cut_7 * math.exp(-1 * particle.dt * (particle.area * K * particle.vapor_pressure_cut_7)/(fieldset.R * fieldset.sea_water_temperature * summed_mass_components))
        particle.mass_cut_8 = particle.mass_cut_8 * math.exp(-1 * particle.dt * (particle.area * K * particle.vapor_pressure_cut_8)/(fieldset.R * fieldset.sea_water_temperature * summed_mass_components))
        particle.mass_cut_9 = particle.mass_cut_9 * math.exp(-1 * particle.dt * (particle.area * K * particle.vapor_pressure_cut_9)/(fieldset.R * fieldset.sea_water_temperature * summed_mass_components))
        particle.mass_cut_10 = particle.mass_cut_10 * math.exp(-1 * particle.dt * (particle.area * K * particle.vapor_pressure_cut_10)/(fieldset.R * fieldset.sea_water_temperature * summed_mass_components))
        particle.mass_cut_11 = particle.mass_cut_11 * math.exp(-1 * particle.dt * (particle.area * K * particle.vapor_pressure_cut_11)/(fieldset.R * fieldset.sea_water_temperature * summed_mass_components))
        particle.mass_cut_12 = particle.mass_cut_12 * math.exp(-1 * particle.dt * (particle.area * K * particle.vapor_pressure_cut_12)/(fieldset.R * fieldset.sea_water_temperature * summed_mass_components))
        particle.mass_cut_13 = particle.mass_cut_13 * math.exp(-1 * particle.dt * (particle.area * K * particle.vapor_pressure_cut_13)/(fieldset.R * fieldset.sea_water_temperature * summed_mass_components))
        particle.mass_cut_14 = particle.mass_cut_14 * math.exp(-1 * particle.dt * (particle.area * K * particle.vapor_pressure_cut_14)/(fieldset.R * fieldset.sea_water_temperature * summed_mass_components))
        particle.mass_cut_15 = particle.mass_cut_15 * math.exp(-1 * particle.dt * (particle.area * K * particle.vapor_pressure_cut_15)/(fieldset.R * fieldset.sea_water_temperature * summed_mass_components))
        particle.mass_cut_16 = particle.mass_cut_16 * math.exp(-1 * particle.dt * (particle.area * K * particle.vapor_pressure_cut_16)/(fieldset.R * fieldset.sea_water_temperature * summed_mass_components))
        particle.mass_cut_17 = particle.mass_cut_17 * math.exp(-1 * particle.dt * (particle.area * K * particle.vapor_pressure_cut_17)/(fieldset.R * fieldset.sea_water_temperature * summed_mass_components))
        particle.mass_cut_18 = particle.mass_cut_18 * math.exp(-1 * particle.dt * (particle.area * K * particle.vapor_pressure_cut_18)/(fieldset.R * fieldset.sea_water_temperature * summed_mass_components))
        particle.mass_cut_19 = particle.mass_cut_19 * math.exp(-1 * particle.dt * (particle.area * K * particle.vapor_pressure_cut_19)/(fieldset.R * fieldset.sea_water_temperature * summed_mass_components))
        particle.mass_cut_20 = particle.mass_cut_20 * math.exp(-1 * particle.dt * (particle.area * K * particle.vapor_pressure_cut_20)/(fieldset.R * fieldset.sea_water_temperature * summed_mass_components))


        particle.mass = particle.mass_cut_1 + particle.mass_cut_2 + particle.mass_cut_3 + particle.mass_cut_4 + particle.mass_cut_5 + particle.mass_cut_6 + particle.mass_cut_7 + particle.mass_cut_8 + particle.mass_cut_9 + particle.mass_cut_10 + particle.mass_cut_11 + particle.mass_cut_12 + particle.mass_cut_13 + particle.mass_cut_14 + particle.mass_cut_15 + particle.mass_cut_16 + particle.mass_cut_17 + particle.mass_cut_18 + particle.mass_cut_19 + particle.mass_cut_20
        particle.fraction_evaporated = (particle.mass_initial - particle.mass)/particle.mass_initial
