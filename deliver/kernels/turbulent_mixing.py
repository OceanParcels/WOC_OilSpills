# Turbulent Mixing kernel

# This kernel is slightly adapted from the kernel developed by Victor Onink for turbulent mixing of particles up to the
# mixed layer depth. This process only occurs for particles not at the surface, so with a depth higher than z = 0.001 m.
# The turbulent mixing goes both up and down, however, it is not enough for particles to resurface. If the depth after the
# mixing is less than zero, or 'above' the surface, the particles depth is reflected.

# Variables needed from environment: mixed layer depth, wind speed, air density, wave age, acceleration due to gravity (g),
#                                    water density, surface depth, v_k, theta, phi
# Variables needed from oil: depth
# Kernel updates: depth

def KPP_wind_mixing(particle, fieldset, time):
"""
Author: Victor Onink, 25/01/22
"""
if particle.depth > 0.001 and particle.entrainment_time != 0:
    # Loading the mixed layer depth from the fieldset
    mixed_layer_depth = fieldset.mld

    # Below the MLD there is no wind-driven turbulent diffusion according to KPP theory, so we set both Kz and dKz to zero.
    if particle.depth > mixed_layer_depth:
        Kz = 0
        dKz = 0

    # Within the MLD we compute the vertical diffusion according to Boufadel et al. (2020) https://doi.org/10.1029/2019JC015727
    else:
      # Calculate the wind speed at the ocean surface
        w_10 = particle.wind_speed

      # Drag coefficient according to Large & Pond (1981) https://doi.org/10.1175/1520-0485(1981)011%3C0324:OOMFMI%3E2.0.CO;2
        C_D = min(max(1.2E-3, 1.0E-3 * (0.49 + 0.065 * w_10)), 2.12E-3)

      # Calculate the surface wind stress based on the surface wind speed and the density of air
        tau = C_D * fieldset.density_air * w_10 ** 2

      # Calcuate the friction velocity of water at the ocean surface using the surface wind stress and the surface water density
        U_W = math.sqrt(tau / fieldset.density_water)

      # Calcuate the surface roughness z0 following Zhao & Li (2019) https://doi.org/10.1007/s10872-018-0494-9
        z0 = 3.5153e-5 * fieldset.wave_age ** (-0.42) * w_10 ** 2 / fieldset.g

      # The corrected particle depth, since the depth is not always zero for the surface circulation data
        z_correct = particle.depth - fieldset.surface_z

      # The diffusion gradient at particle.depth
        C1 = (fieldset.v_k * U_W * fieldset.theta) / (fieldset.phi * mixed_layer_depth ** 2)
        dKz = C1 * (mixed_layer_depth - z_correct) * (mixed_layer_depth - 3 * z_correct - 2 * z0)

      # The KPP profile vertical diffusion, at a depth corrected for the vertical gradient in Kz
        C2 = (fieldset.v_k * U_W * fieldset.theta) / fieldset.phi
        Kz = C2 * (z_correct + z0) * math.pow(1 - z_correct / mixed_layer_depth, 2)

    # The Markov-0 vertical transport from Grawe et al. (2012) http://dx.doi.org/10.1007/s10236-012-0523-y
    gradient = dKz * particle.dt
    R = ParcelsRandom.uniform(-1., 1.) * math.sqrt(math.fabs(particle.dt) * 3) * math.sqrt(2 * Kz)

    # Update the particle depth
    particle.depth = particle.depth + gradient + R
    if particle.depth < 0:
        particle.depth = particle.depth * -1
