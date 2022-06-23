def parcels_oil_simulation(output_names, ndays, output_dt, lat_start, lon_start, wind_factor, diameter, oil_types, nparticles):

    import numpy as np
    from parcels import FieldSet, Field, VectorField, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4, ErrorCode, Variable, ParcelsRandom, DiffusionUniformKh, AdvectionDiffusionM1, AdvectionDiffusionEM
    from parcels.tools.converters import GeographicPolar, Geographic
    from datetime import timedelta
    from glob import glob
    import math
    import xarray as xr

    # Particle set
    """
    This sets up the oil particle with different variables used in the kernels.

    The water fraction is the amount of water in the oil droplets from emulsification, and affects the density and the viscosity.
    The evaporation fraction affects the viscosity of the oil.
    """

    # Norman Wells Light Crude Oil
    class LowDensityOil(JITParticle):
        viscosity = Variable('viscosity', dtype=np.float32, initial=0.007)
        density = Variable('density', dtype=np.float32, initial=838)
        mass = Variable('mass', dtype=np.float32, initial=838)
        original_mass = Variable('original_mass', dtype=np.float32, initial=838)
        original_density = Variable('original_density', dtype=np.float32, initial=838)
        original_viscosity = Variable('original_viscosity', dtype=np.float32, initial=0.007)
        water_fraction = Variable('water_fraction', dtype=np.float32, initial=0)
        max_water_fraction = Variable('max_water_fraction', dtype=np.float32, initial=0.9)
        fraction_evaporated = Variable('fraction_evaporated', dtype=np.float32, initial=0)
        is_emulsified = Variable('is_emulsified', dtype=np.float32, initial=0)
        emulsification_time = Variable('emulsification_time', dtype=np.float32, initial=0)
        vapor_pressure = Variable('vapor_pressure', dtype=np.float32, initial=119933.00831083095)
        interfacial_tension = Variable('interfacial_tension', dtype=np.float32, initial=0.0248)
        area = Variable('area', dtype=np.float32, initial = 0.1)
        interfacial_area = Variable('interfacial_area', dtype=np.float32, initial = 0)
        d_min = Variable('d_min', dtype=np.float32, initial = 1e-6)
        d_max = Variable('d_max', dtype=np.float32, initial = 1e-5)
        max_diameter = Variable('max_diameter', dtype=np.float32, initial = diameter)
        min_diameter = Variable('min_diameter', dtype=np.float32, initial = diameter)
        rise_velocity = Variable('rise_velocity', dtype=np.float32, initial = 0)
        U10 = Variable('U10', dtype=np.float32, initial = 0)
        V10 = Variable('V10', dtype=np.float32, initial = 0)
        wind_speed = Variable('wind_speed', dtype=np.float32, initial = 0)
        entrainment_time = Variable('entrainment_time', dtype=np.float32, initial = 0)

    # Dos Cuadros HE-26 [2011] Medium Crude Oil
    class MediumDensityOil(ScipyParticle):
        viscosity = Variable('viscosity', dtype=np.float32, initial=0.021)
        density = Variable('density', dtype=np.float32, initial=879)
        mass = Variable('mass', dtype=np.float32, initial=879)
        original_mass = Variable('original_mass', dtype=np.float32, initial=879)
        original_density = Variable('original_density', dtype=np.float32, initial=879)
        original_viscosity = Variable('original_viscosity', dtype=np.float32, initial=0.021)
        water_fraction = Variable('water_fraction', dtype=np.float32, initial=0)
        max_water_fraction = Variable('max_water_fraction', dtype=np.float32, initial=0.9)
        fraction_evaporated = Variable('fraction_evaporated', dtype=np.float32, initial=0)
        is_emulsified = Variable('is_emulsified', dtype=np.float32, initial=0)
        emulsification_time = Variable('emulsification_time', dtype=np.float32, initial=0)
        vapor_pressure = Variable('vapor_pressure', dtype=np.float32, initial=26155.627736028568)
        interfacial_tension = Variable('interfacial_tension', dtype=np.float32, initial=0.0161)
        area = Variable('area', dtype=np.float32, initial = 0.1)
        interfacial_area = Variable('interfacial_area', dtype=np.float32, initial = 0)
        d_min = Variable('d_min', dtype=np.float32, initial = 1e-6)
        d_max = Variable('d_max', dtype=np.float32, initial = 1e-5)
        max_diameter = Variable('max_diameter', dtype=np.float32, initial = diameter)
        min_diameter = Variable('min_diameter', dtype=np.float32, initial = diameter)
        rise_velocity = Variable('rise_velocity', dtype=np.float32, initial = 0)
        U10 = Variable('U10', dtype=np.float32, initial = 0)
        V10 = Variable('V10', dtype=np.float32, initial = 0)
        wind_speed = Variable('wind_speed', dtype=np.float32, initial = 0)
        entrainment_time = Variable('entrainment_time', dtype=np.float32, initial = 0)
        decay = Variable('decay', dtype=np.float32, initial = 0)

    # Platform Ellen A038 Heavy Crude Oil    
    class HighDensityOil(ScipyParticle):
        viscosity = Variable('viscosity', dtype=np.float32, initial=3.1)
        density = Variable('density', dtype=np.float32, initial=959)
        mass = Variable('mass', dtype=np.float32, initial=959)
        original_mass = Variable('original_mass', dtype=np.float32, initial=959)
        original_density = Variable('original_density', dtype=np.float32, initial=959)
        original_viscosity = Variable('original_viscosity', dtype=np.float32, initial=3.1)
        water_fraction = Variable('water_fraction', dtype=np.float32, initial=0)
        max_water_fraction = Variable('max_water_fraction', dtype=np.float32, initial=0.9)
        fraction_evaporated = Variable('fraction_evaporated', dtype=np.float32, initial=0)
        is_emulsified = Variable('is_emulsified', dtype=np.float32, initial=0)
        emulsification_time = Variable('emulsification_time', dtype=np.float32, initial=0)
        vapor_pressure = Variable('vapor_pressure', dtype=np.float32, initial=48026.29949199871)
        interfacial_tension = Variable('interfacial_tension', dtype=np.float32, initial=0.0223)
        area = Variable('area', dtype=np.float32, initial = 0.1)
        interfacial_area = Variable('interfacial_area', dtype=np.float32, initial = 0)
        d_min = Variable('d_min', dtype=np.float32, initial = 1e-6)
        d_max = Variable('d_max', dtype=np.float32, initial = 1e-5)
        max_diameter = Variable('max_diameter', dtype=np.float32, initial = diameter)
        min_diameter = Variable('min_diameter', dtype=np.float32, initial = diameter)
        rise_velocity = Variable('rise_velocity', dtype=np.float32, initial = 0)
        U10 = Variable('U10', dtype=np.float32, initial = 0)
        V10 = Variable('V10', dtype=np.float32, initial = 0)
        wind_speed = Variable('wind_speed', dtype=np.float32, initial = 0)
        entrainment_time = Variable('entrainment_time', dtype=np.float32, initial = 0)

    # Add MOI field set

    data_path = '/storage/shared/oceanparcels/input_data/MOi/'
    ufiles = sorted(glob(data_path+'psy4v3r1/psy4v3r1-daily_U_2019-06-*.nc'))
    vfiles = [f.replace('_U_', '_V_') for f in ufiles]
    wfiles = [f.replace('_U_', '_W_') for f in ufiles]
    mesh_mask = data_path + 'domain_ORCA0083-N006/coordinates.nc'

    filenames = {'U': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': ufiles},
                 'V': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': vfiles},
                 'W': {'lon': mesh_mask, 'lat': mesh_mask, 'depth': wfiles[0], 'data': wfiles}}
    variables = {'U': 'vozocrtx', 'V': 'vomecrty', 'W': 'vovecrtz'}
    dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
                  'V': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'},
                  'W': {'lon': 'glamf', 'lat': 'gphif', 'depth': 'depthw', 'time': 'time_counter'}}

    fieldset_moi = FieldSet.from_nemo(filenames, variables, dimensions)

#     # Add wind fieldset from netcdf file

    wind_dataset = xr.open_dataset('era_5_wind_data_june_2019.nc')

    filenames_t_wind = {'U': 'era_5_wind_data_june_2019.nc', 'V': 'era_5_wind_data_june_2019.nc'}
    variables_t_wind = {'U': 'u10', 'V': 'v10'}
    dimensions_t_wind = {'U': {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'},
                  'V': {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}}

    fieldset_wind_total = FieldSet.from_netcdf(filenames_t_wind, variables_t_wind, dimensions_t_wind)
    fieldset_wind_factor = FieldSet.from_netcdf(filenames_t_wind, variables_t_wind, dimensions_t_wind)

    # Add wind speed fields to fieldset
    
    u_wind_total, v_wind_total = fieldset_wind_total.UV
    fieldset_moi.add_field(u_wind_total)
    fieldset_moi.add_field(v_wind_total)
    
    fieldset_wind_factor.U.set_scaling_factor(wind_factor)
    fieldset_wind_factor.V.set_scaling_factor(wind_factor)
    
    u_wind_factor, v_wind_factor = fieldset_wind_factor.UV
    fieldset_moi.add_field(u_wind_factor, name='U10_factor')
    fieldset_moi.add_field(v_wind_factor, name='V10_factor')
    
#     u_wind_factor = Field('U10_factor', wind_dataset['u10'] * wind_factor, grid=fieldset_wind.U.grid)
#     v_wind_factor = Field('V10_factor', wind_dataset['v10'] * wind_factor, grid=fieldset_wind.V.grid)

#     u_wind_total = Field('U10', wind_dataset['u10'], grid=fieldset_wind.U.grid)
#     v_wind_total = Field('V10', wind_dataset['v10'], grid=fieldset_wind.V.grid)

#     fieldset_moi.add_field(u_wind_factor)
#     fieldset_moi.add_field(v_wind_factor)
#     fieldset_moi.add_field(u_wind_total)
#     fieldset_moi.add_field(v_wind_total)

#     fieldset_moi.U10_factor.units = GeographicPolar()
#     fieldset_moi.V10_factor.units = Geographic()

    fieldset_moi.add_constant('density_water', 1028)
    fieldset_moi.add_constant('viscosity_water', 0.00122/1028)
    fieldset_moi.add_constant('g', 9.81)
    fieldset_moi.add_constant('wind_threshold', 5)
    fieldset_moi.add_constant('sea_surface_temperature', 283)
    fieldset_moi.add_constant('mld', 50)
    fieldset_moi.add_constant('density_air', 1.22)
    fieldset_moi.add_constant('wave_age', 35)
    fieldset_moi.add_constant('surface_z', 0)
    fieldset_moi.add_constant('v_k', 0.4)
    fieldset_moi.add_constant('theta', 1)
    fieldset_moi.add_constant('phi', 0.9)

    # Delete particles if out of bounds

    def DeleteParticle(particle, fieldset, time):
        particle.delete()

    # Oil advection kernel with wind factor added

    def OilAdvectionRK4(particle, fieldset, time):

        if particle.depth < 0.001:
            (u1, v1) = fieldset.UV[particle]
            u10_1 = fieldset.U10_factor[particle]
            v10_1 = fieldset.V10_factor[particle]
            lon1, lat1 = (particle.lon + (u1 + u10_1)*.5*particle.dt, particle.lat + (v1 + v10_1)*.5*particle.dt)
            (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1, particle]
            u10_2 = fieldset.U10_factor[time + .5 * particle.dt, particle.depth, lat1, lon1, particle]
            v10_2 = fieldset.V10_factor[time + .5 * particle.dt, particle.depth, lat1, lon1, particle]
            lon2, lat2 = (particle.lon + (u2 + u10_2)*.5*particle.dt, particle.lat + (v2 + v10_2)*.5*particle.dt)
            (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2, particle]
            u10_3 = fieldset.U10_factor[time + .5 * particle.dt, particle.depth, lat2, lon2, particle]
            v10_3 = fieldset.V10_factor[time + .5 * particle.dt, particle.depth, lat2, lon2, particle]
            lon3, lat3 = (particle.lon + (u3 + u10_3)*particle.dt, particle.lat + (v3 + v10_3)*particle.dt)
            (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3, particle]
            u10_4 = fieldset.U10_factor[time + particle.dt, particle.depth, lat3, lon3, particle]
            v10_4 = fieldset.V10_factor[time + particle.dt, particle.depth, lat3, lon3, particle]
            particle.lon += ((u1 + u10_1) + 2*(u2 + u10_2) + 2*(u3 + u10_3) + (u4 + u10_4)) / 6. * particle.dt
            particle.lat += ((v1 + v10_1) + 2*(v2 + v10_2) + 2*(v3 + v10_3) + (v4 + v10_4)) / 6. * particle.dt

        else:
            (u1, v1) = fieldset.UV[particle]
            lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)
            (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1, particle]
            lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)
            (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2, particle]
            lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)
            (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3, particle]
            particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
            particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt

    # Evaporation

    def Evaporation(particle, fieldset, time):

        particle.U10 = fieldset.U10[particle]
        particle.V10 = fieldset.V10[particle]
        particle.wind_speed = math.sqrt(particle.U10**2 + particle.V10**2)

        if particle.depth < 0.001:

            # Mass Transport Coefficient, K
            if abs(particle.wind_speed) < 10:
                K = 0.0025 * particle.wind_speed**0.78
            else:
                K = 0.06 * particle.wind_speed**2

            particle.decay = particle.area * K * particle.vapor_pressure / (8.314 * fieldset.sea_surface_temperature * particle.mass)
            particle.mass = particle.original_mass * math.exp(-1 * particle.decay * (particle.emulsification_time + particle.dt))
            particle.emulsification_time += particle.dt

            particle.fraction_evaporated = (particle.original_mass - particle.mass) / particle.original_mass

    # Emulsification

    def Emulsification(particle, fieldset, time):

        if particle.depth < 0.001:

            max_interfacial_area = (6 / particle.d_min) * (particle.max_water_fraction / (1 - particle.max_water_fraction))

            k_emul = particle.wind_speed**2 * (6 * 2.02e-6)/(1e-5)

            particle.interfacial_area += particle.dt * (k_emul * (1 - (particle.interfacial_area/max_interfacial_area)))
            if particle.interfacial_area > max_interfacial_area:
                particle.interfacial_area = max_interfacial_area

            particle.water_fraction = particle.interfacial_area * particle.d_max / (6 + (particle.interfacial_area * particle.d_max))
            if particle.water_fraction > particle.max_water_fraction:
                particle.water_fraction = particle.max_water_fraction

    # Update Properties

    def Update_oil_properties(particle, fieldset, time):

        kv1 = math.sqrt(particle.viscosity) * 1.5e3
        if kv1 > 10:
            kv1 = 10
        elif kv1 < 1:
            kv1 = 1

        particle.density = particle.water_fraction * fieldset.density_water + (1 - particle.water_fraction) * particle.original_density

        particle.viscosity = particle.original_viscosity * math.exp(kv1 * particle.fraction_evaporated) * (1 + (particle.water_fraction/0.84)/(1.187 - (particle.water_fraction/0.84)))**2.49

    # Entrainment

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

    # Turbulent Mixing

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

    # Buyonacy velocity rise

    def Buoyancy(particle, fieldset, time):

        if particle.depth > 0.001 and particle.entrainment_time != 0:

            r = particle.min_diameter
            particle.rise_velocity = 2 * 9.81 * (1 - (particle.density/fieldset.density_water)) * r**2 / (9 * fieldset.viscosity_water)

            if (particle.rise_velocity * 2 * r / fieldset.viscosity_water) > 50:
                particle.rise_velocity = math.sqrt((16/3) * 9.81 * (1 - (particle.density/fieldset.density_water)) * r)

            particle.depth = particle.depth - particle.rise_velocity * particle.dt

            if particle.depth < 0.001:
                particle.depth = 0
                particle.entrainment_time = 0

    number_particles = nparticles

    lonp = lon_start * np.ones(number_particles)
    latp = lat_start * np.ones(number_particles)
    depp = np.zeros(number_particles)

    if 'light' in oil_types:
        
        pset_light = ParticleSet.from_list(fieldset=fieldset_moi, pclass=LowDensityOil, lon=lonp, lat=latp, depth=depp)

        kernels_light = pset_light.Kernel(OilAdvectionRK4) + pset_light.Kernel(Entrainment) + pset_light.Kernel(KPP_wind_mixing) + pset_light.Kernel(Buoyancy)

        output_file_light = pset_light.ParticleFile(name=output_names[0], outputdt=timedelta(minutes=output_dt))

        pset_light.execute(kernels_light, runtime=timedelta(days=ndays), dt=timedelta(seconds=30), recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}, output_file=output_file_light)

        output_file_light.close()
        if 'medium' in oil_types:
            
            pset_medium = ParticleSet.from_list(fieldset=fieldset_moi, pclass=MediumDensityOil, lon=lonp, lat=latp, depth=depp)

            kernels_medium = pset_medium.Kernel(OilAdvectionRK4) + pset_medium.Kernel(Entrainment) + pset_medium.Kernel(KPP_wind_mixing) + pset_medium.Kernel(Buoyancy)

            output_file_medium = pset_medium.ParticleFile(name=output_names[1], outputdt=timedelta(minutes=output_dt))

            pset_medium.execute(kernels_medium, runtime=timedelta(days=ndays), dt=timedelta(seconds=30), recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}, output_file=output_file_medium)

            output_file_medium.close()
            
            if 'heavy' in oil_types:

                pset_heavy = ParticleSet.from_list(fieldset=fieldset_moi, pclass=HighDensityOil, lon=lonp, lat=latp, depth=depp)

                kernels_heavy = pset_heavy.Kernel(OilAdvectionRK4) + pset_heavy.Kernel(Entrainment) + pset_heavy.Kernel(KPP_wind_mixing) + pset_heavy.Kernel(Buoyancy)

                output_file_heavy = pset_heavy.ParticleFile(name=output_names[2], outputdt=timedelta(minutes=output_dt))

                pset_heavy.execute(kernels_heavy, runtime=timedelta(days=ndays), dt=timedelta(seconds=30), recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}, output_file=output_file_heavy)

                output_file_heavy.close()
            
        elif 'medium' not in oil_types and 'heavy' in oil_types:
            
            pset_heavy = ParticleSet.from_list(fieldset=fieldset_moi, pclass=HighDensityOil, lon=lonp, lat=latp, depth=depp)

            kernels_heavy = pset_heavy.Kernel(OilAdvectionRK4) + pset_heavy.Kernel(Entrainment) + pset_heavy.Kernel(KPP_wind_mixing) + pset_heavy.Kernel(Buoyancy)

            output_file_heavy = pset_heavy.ParticleFile(name=output_names[1], outputdt=timedelta(minutes=output_dt))

            pset_heavy.execute(kernels_heavy, runtime=timedelta(days=ndays), dt=timedelta(seconds=30), recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}, output_file=output_file_heavy)

            output_file_heavy.close()
    
    elif 'medium' in oil_types:
        
        pset_medium = ParticleSet.from_list(fieldset=fieldset_moi, pclass=MediumDensityOil, lon=lonp, lat=latp, depth=depp)

        kernels_medium = pset_medium.Kernel(OilAdvectionRK4) + pset_medium.Kernel(Entrainment) + pset_medium.Kernel(KPP_wind_mixing) + pset_medium.Kernel(Buoyancy)

        output_file_medium = pset_medium.ParticleFile(name=output_names[0], outputdt=timedelta(minutes=output_dt))

        pset_medium.execute(kernels_medium, runtime=timedelta(days=ndays), dt=timedelta(seconds=30), recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}, output_file=output_file_medium)

        output_file_medium.close()
            
        if 'heavy' in oil_types:
            pset_heavy = ParticleSet.from_list(fieldset=fieldset_moi, pclass=HighDensityOil, lon=lonp, lat=latp, depth=depp)

            kernels_heavy = pset_heavy.Kernel(OilAdvectionRK4) + pset_heavy.Kernel(Entrainment) + pset_heavy.Kernel(KPP_wind_mixing) + pset_heavy.Kernel(Buoyancy)

            output_file_heavy = pset_heavy.ParticleFile(name=output_names[1], outputdt=timedelta(minutes=output_dt))

            pset_heavy.execute(kernels_heavy, runtime=timedelta(days=ndays), dt=timedelta(seconds=30), recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}, output_file=output_file_heavy)

            output_file_heavy.close()
    
    elif 'heavy' in oil_types:
        pset_heavy = ParticleSet.from_list(fieldset=fieldset_moi, pclass=HighDensityOil, lon=lonp, lat=latp, depth=depp)

        kernels_heavy = pset_heavy.Kernel(OilAdvectionRK4) + pset_heavy.Kernel(Entrainment) + pset_heavy.Kernel(KPP_wind_mixing) + pset_heavy.Kernel(Buoyancy)

        output_file_heavy = pset_heavy.ParticleFile(name=output_names[0], outputdt=timedelta(minutes=output_dt))

        pset_heavy.execute(kernels_heavy, runtime=timedelta(days=ndays), dt=timedelta(seconds=30), recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}, output_file=output_file_heavy)

        output_file_heavy.close()
        
    else:
        print('error, probable typo. Oil types should be light, medium, heavy')