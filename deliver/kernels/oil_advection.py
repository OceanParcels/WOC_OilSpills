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
