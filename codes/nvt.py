import mattertune as mt
import pandas as pd

from ase.io import read, write
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

model = mt.backbones.mattersim.MatterSimM3GNetBackboneModule.load_from_checkpoint(f"../checkpoint/MatterSim-v1.0.0-1M-cspbbr3-nc-geomopt.ckpt")
calculator = model.ase_calculator()

def print_step(dyn, temp, energy):
    atoms = dyn.atoms
    T_inst = atoms.get_temperature()
    E_kin = atoms.get_kinetic_energy()  
    E_pot = atoms.get_potential_energy() 
    E_tot = E_kin + E_pot

    print(f"Step: {dyn.nsteps}, Temperature: {T_inst:.2f} K, Total Energy: {E_tot:.6f} eV")
    
    temp.append(T_inst)
    energy.append(E_tot)
    
def write_xyz(filename, atoms):
    write(filename+".xyz", atoms, format="xyz", append=True)

def MD(thermostat, atoms, T, timestep, time, interval, filename):
    if time < 1000:
        filename = f'{filename}_{T}_{time}fs'
    elif time < 1000000:
        filename = f'{filename}_{T}_{time/1000:.3f}ps'
    else:
        filename = f'{filename}_{T}_{time/1000000:.2f}ns'
    
    atoms.calc = calculator
    atoms.pbc = False
    atoms.center()   

    temp = []
    energy = []
    
    MaxwellBoltzmannDistribution(atoms, temperature_K=T)
    Stationary(atoms)  
    ZeroRotation(atoms)

    if thermostat == "langevin":
        dyn = Langevin(
            atoms,
            timestep=timestep * units.fs,
            temperature_K=T,
            friction=0.02 / units.fs,
        )
    else:
        raise ValueError(f"Unknown thermostat: {thermostat}")
    
    dyn.attach(write_xyz, interval=interval, atoms=atoms, filename=filename)
    dyn.attach(lambda: print_step(dyn, temp, energy), interval=interval)

    open(filename + ".xyz", "w").close()
    dyn.run(round(time/timestep))
    
    df = pd.DataFrame({"T": temp, "E": energy})
    df.to_csv(f"{filename}.csv")

filename = "your_filename"

# Load structure
atoms = read(f'../geometry_opt/{filename}.xyz')

MD(thermostat="langevin",
    atoms=atoms, 
    T=300, # K
    timestep=1, # fs
    time=100000, # fs
    interval=50, # save every n frames
    filename=filename)