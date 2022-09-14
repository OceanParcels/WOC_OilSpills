def parcels_oil_simulation(output_name, ndays, output_dt, lat_ref, lon_ref, wind_factor, diameter, oil_types):

    import numpy as np
    from parcels import FieldSet, Field, VectorField, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4, ErrorCode, Variable, ParcelsRandom, DiffusionUniformKh, AdvectionDiffusionM1, AdvectionDiffusionEM
    from parcels.tools.converters import GeographicPolar, Geographic
    from datetime import timedelta, datetime
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
    class LowDensityOil(ScipyParticle):
      viscosity = Variable('viscosity', dtype=np.float32, initial=8.353221957040573e-06)
      density = Variable('density', dtype=np.float32, initial=838)
      mass = Variable('mass', dtype=np.float32, initial=1000)
      mass_initial = Variable('mass_initial', dtype=np.float32, initial=1000)
      original_density = Variable('original_density', dtype=np.float32, initial=838)
      original_viscosity = Variable('original_viscosity', dtype=np.float32, initial=8.353221957040573e-06)
      water_fraction = Variable('water_fraction', dtype=np.float32, initial=0)
      max_water_fraction = Variable('max_water_fraction', dtype=np.float32, initial=0.9)
      fraction_evaporated = Variable('fraction_evaporated', dtype=np.float32, initial=0)
      interfacial_tension = Variable('interfacial_tension', dtype=np.float32, initial=0.0248)
      area = Variable('area', dtype=np.float32, initial = 346.93456406896297)
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
      saturates_fraction = Variable('saturates_fraction', dtype=np.float32, initial = 0.71)
      aromatics_fraction = Variable('aromatics_fraction', dtype=np.float32, initial = 0.21)
      resins_fraction = Variable('resins_fraction', dtype=np.float32, initial = 0.07)
      asphaltenes_fraction = Variable('asphaltenes_fraction', dtype=np.float32, initial = 0.006)
      volume_initial = Variable('volume_initial', dtype=np.float32, initial = 1000/838)
      age = Variable('age', dtype=np.float32, initial=0)
      max_area = Variable('max_area', dtype=np.float32, initial=71521.16988722442)
      emulsification_onset_percentage = Variable('emulsification_onset_percentage', dtype=np.float32, initial=0.19)

      # Distillation cuts, which is found from the NOAA oil database

      number_cuts = Variable('number_cuts', dtype=np.float32, initial = 20)
      mass_cut_1 = Variable('mass_cut_1', dtype=np.float32, initial = 50)
      molecular_weight_cut_1 = Variable('molecular_weight_cut_1', dtype=np.float32, initial = 119.70584938)
      vapor_pressure_cut_1 = Variable('vapor_pressure_cut_1', dtype=np.float32, initial = 4.69066845e+04)
      mass_cut_2 = Variable('mass_cut_2', dtype=np.float32, initial = 50)
      molecular_weight_cut_2 = Variable('molecular_weight_cut_2', dtype=np.float32, initial = 126.50508921)
      vapor_pressure_cut_2 = Variable('vapor_pressure_cut_2', dtype=np.float32, initial = 4.35251887e+04)
      mass_cut_3 = Variable('mass_cut_3', dtype=np.float32, initial = 50)
      molecular_weight_cut_3 = Variable('molecular_weight_cut_3', dtype=np.float32, initial = 129.87562488)
      vapor_pressure_cut_3 = Variable('vapor_pressure_cut_3', dtype=np.float32, initial = 2.36946193e+04)
      mass_cut_4 = Variable('mass_cut_4', dtype=np.float32, initial = 50)
      molecular_weight_cut_4 = Variable('molecular_weight_cut_4', dtype=np.float32, initial = 142.88896431)
      vapor_pressure_cut_4 = Variable('vapor_pressure_cut_4', dtype=np.float32, initial = 3.29190505e+03)
      mass_cut_5 = Variable('mass_cut_5', dtype=np.float32, initial = 50)
      molecular_weight_cut_5 = Variable('molecular_weight_cut_5', dtype=np.float32, initial = 157.77561454)
      vapor_pressure_cut_5 = Variable('vapor_pressure_cut_5', dtype=np.float32, initial = 1.52385542e+03)
      mass_cut_6 = Variable('mass_cut_6', dtype=np.float32, initial = 50)
      molecular_weight_cut_6 = Variable('molecular_weight_cut_6', dtype=np.float32, initial = 167.18566083)
      vapor_pressure_cut_6 = Variable('vapor_pressure_cut_6', dtype=np.float32, initial = 6.00350257e+02)
      mass_cut_7 = Variable('mass_cut_7', dtype=np.float32, initial = 50)
      molecular_weight_cut_7 = Variable('molecular_weight_cut_7', dtype=np.float32, initial = 177.44847886)
      vapor_pressure_cut_7 = Variable('vapor_pressure_cut_7', dtype=np.float32, initial = 2.50325988e+02)
      mass_cut_8 = Variable('mass_cut_8', dtype=np.float32, initial = 50)
      molecular_weight_cut_8 = Variable('molecular_weight_cut_8', dtype=np.float32, initial = 188.08865762)
      vapor_pressure_cut_8 = Variable('vapor_pressure_cut_8', dtype=np.float32, initial = 9.63052865e+01)
      mass_cut_9 = Variable('mass_cut_9', dtype=np.float32, initial = 50)
      molecular_weight_cut_9 = Variable('molecular_weight_cut_9', dtype=np.float32, initial = 200.59550024)
      vapor_pressure_cut_9 = Variable('vapor_pressure_cut_9', dtype=np.float32, initial = 3.06620842e+01)
      mass_cut_10 = Variable('mass_cut_10', dtype=np.float32, initial = 50)
      molecular_weight_cut_10 = Variable('molecular_weight_cut_10', dtype=np.float32, initial = 214.62039403)
      vapor_pressure_cut_10 = Variable('vapor_pressure_cut_10', dtype=np.float32, initial = 9.77377165)
      mass_cut_11 = Variable('mass_cut_11', dtype=np.float32, initial = 50)
      molecular_weight_cut_11 = Variable('molecular_weight_cut_11', dtype=np.float32, initial = 230.02946782)
      vapor_pressure_cut_11 = Variable('vapor_pressure_cut_11', dtype=np.float32, initial = 2.65107466)
      mass_cut_12 = Variable('mass_cut_12', dtype=np.float32, initial = 50)
      molecular_weight_cut_12 = Variable('molecular_weight_cut_12', dtype=np.float32, initial = 248.4996907)
      vapor_pressure_cut_12 = Variable('vapor_pressure_cut_12', dtype=np.float32, initial = 5.65929814e-01)
      mass_cut_13 = Variable('mass_cut_13', dtype=np.float32, initial = 50)
      molecular_weight_cut_13 = Variable('molecular_weight_cut_13', dtype=np.float32, initial = 270.2263597)
      vapor_pressure_cut_13 = Variable('vapor_pressure_cut_13', dtype=np.float32, initial = 1.04090954e-01)
      mass_cut_14 = Variable('mass_cut_14', dtype=np.float32, initial = 50)
      molecular_weight_cut_14 = Variable('molecular_weight_cut_14', dtype=np.float32, initial = 295.42003103)
      vapor_pressure_cut_14 = Variable('vapor_pressure_cut_14', dtype=np.float32, initial = 1.51643728e-02)
      mass_cut_15 = Variable('mass_cut_15', dtype=np.float32, initial = 50)
      molecular_weight_cut_15 = Variable('molecular_weight_cut_15', dtype=np.float32, initial = 325.81633305)
      vapor_pressure_cut_15 = Variable('vapor_pressure_cut_15', dtype=np.float32, initial = 1.57425213e-03)
      mass_cut_16 = Variable('mass_cut_16', dtype=np.float32, initial = 50)
      molecular_weight_cut_16 = Variable('molecular_weight_cut_16', dtype=np.float32, initial = 362.75352361)
      vapor_pressure_cut_16 = Variable('vapor_pressure_cut_16', dtype=np.float32, initial = 1.18463227e-04)
      mass_cut_17 = Variable('mass_cut_17', dtype=np.float32, initial = 50)
      molecular_weight_cut_17 = Variable('molecular_weight_cut_17', dtype=np.float32, initial = 406.78376222)
      vapor_pressure_cut_17 = Variable('vapor_pressure_cut_17', dtype=np.float32, initial = 6.64070916e-06)
      mass_cut_18 = Variable('mass_cut_18', dtype=np.float32, initial = 50)
      molecular_weight_cut_18 = Variable('molecular_weight_cut_18', dtype=np.float32, initial = 464.66297459)
      vapor_pressure_cut_18 = Variable('vapor_pressure_cut_18', dtype=np.float32, initial = 1.39138543e-07)
      mass_cut_19 = Variable('mass_cut_19', dtype=np.float32, initial = 50)
      molecular_weight_cut_19 = Variable('molecular_weight_cut_19', dtype=np.float32, initial = 552.2402062)
      vapor_pressure_cut_19 = Variable('vapor_pressure_cut_19', dtype=np.float32, initial = 5.96516670e-10)
      mass_cut_20 = Variable('mass_cut_20', dtype=np.float32, initial = 50)
      molecular_weight_cut_20 = Variable('molecular_weight_cut_20', dtype=np.float32, initial = 705.94346753)
      vapor_pressure_cut_20 = Variable('vapor_pressure_cut_20', dtype=np.float32, initial = 1.01473136e-13)

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

    fieldset_wind = FieldSet.from_netcdf(filenames_t_wind, variables_t_wind, dimensions_t_wind)

    # Add wind speed fields to fieldset

    u_wind_factor = Field('U10_factor', wind_dataset['u10'] * wind_factor, grid=fieldset_wind.U.grid)
    v_wind_factor = Field('V10_factor', wind_dataset['v10'] * wind_factor, grid=fieldset_wind.V.grid)

    u_wind_total = Field('U10', wind_dataset['u10'], grid=fieldset_wind.U.grid)
    v_wind_total = Field('V10', wind_dataset['v10'], grid=fieldset_wind.V.grid)

    fieldset_moi.add_field(u_wind_factor)
    fieldset_moi.add_field(v_wind_factor)
    fieldset_moi.add_field(u_wind_total)
    fieldset_moi.add_field(v_wind_total)

    fieldset_moi.U10_factor.units = GeographicPolar()
    fieldset_moi.V10_factor.units = Geographic()

    fieldset_moi.add_constant('density_water', 1026)
    fieldset_moi.add_constant('viscosity_water', 0.00122/1026)
    fieldset_moi.add_constant('g', 9.81)
    fieldset_moi.add_constant('R', 8.31446261815324)
    fieldset_moi.add_constant('wind_threshold', 5)
    fieldset_moi.add_constant('sea_water_temperature', 283)
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

    # Oil advection kernel

    # This kernel is used to advect particles are each time step using the Runge-Kutta (RK4) integration method.
    # For particles at the surface (with a depth of smaller than z = 0.001 m), the particles are advected with both current data
    # (U, V), and a factor of the wind speed at 10m (U10, V10). For depths larger than this value, the particles are only advected
    # with the current data.

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

    lons_list=[lon_ref-.06, lon_ref, lon_ref+.06]
    lats_list=[lat_ref-.06,  lat_ref, lat_ref+.06]

    lonp, latp = np.meshgrid(np.array(lons_list), np.array(lats_list))

    depp = np.zeros(lonp.shape[0]*lonp.shape[1])

    timep = datetime(2011,9,10)

    pset_light = ParticleSet.from_list(fieldset=fieldset_moi, pclass=LowDensityOil, lon=lonp, lat=latp, depth=depp, time=timep)

    kernels_light = pset_light.Kernel(OilAdvectionRK4) + pset_light.Kernel(Entrainment) + pset_light.Kernel(KPP_wind_mixing) + pset_light.Kernel(Buoyancy) + pset_light.Kernel(Spreading) + pset_light.Kernel(Evaporation) + pset_light.Kernel(Emulsification) + pset_light.Kernel(Update_oil_properties)

    output_file_light = pset_light.ParticleFile(name=output_name, outputdt=timedelta(minutes=output_dt))

    pset_light.execute(kernels_light, runtime=timedelta(days=ndays), dt=timedelta(seconds=30), recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}, output_file=output_file_light)

    output_file_light.close()
