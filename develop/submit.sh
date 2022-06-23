#!/bin/bash -l
#
#SBATCH -J jit-sim   # the name of your job   
#SBATCH -p normal           # request normal partition, job takes > 1 hour (this line can also be left out because 'normal' is the default)  
#SBATCH -t 48:00:00         # time in hh:mm:ss you want to reserve for the job
#SBATCH -n 1                # the number of cores you want to use for the job, SLURM automatically determines how many nodes are needed
#SBATCH -o log_pytest.%j.o  # the name of the file where the standard output will be written to. %j will be the jobid determined by SLURM
#SBATCH -e log_pytest.%j.e  # the name of the file where potential errors will be written to. %j will be the jobid determined by SLURM

cd develop/2022-may/plots-june
conda activate envname      # this passes your conda environment to all the compute nodes

python3 -c 'from wind-field-tests import *; parcels_oil_simulation(output_names = ["jit_test"], ndays=0.5, output_dt=20, lat_start=59.876814891293556, lon_start=2.4757442009520174, wind_factor=0.02, diameter=1e-5, oil_types=["light"], nparticles=100)'