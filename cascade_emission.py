#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import numpy as np
import random
import math
import argparse
import time


# # Data Extraction

# In[21]:


def LoadEnergyDependentSpectrum(filepath):
    """
    Loads the energy-dependent spectrum from a .csv file.

    Parameters:
    - filepath: str, a path to the spectrum .csv file.

    Returns:
    - energy_centres: array, the centres of the energy bins.
    - spectrum_centres: array, the centres of the frequency bins.
    - spectrum: array, the energy-dependent spectrum.
    
    """
    # Extracts energy bins from the header
    with open(filepath, "r") as f:
        header = f.readline().strip()
    energy_centres = np.array(header.split(",")[1:], dtype=float)
    
    # Converts the energy centres into eV where 0 eV is the ground state
    bin_width = energy_centres[1] - energy_centres[0]
    ground_state_energy = energy_centres[0] - bin_width/2
    for i in range(len(energy_centres)):
        energy_centres[i] = (energy_centres[i]-ground_state_energy)/8065.541154

    # Loads the remaining data
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)

    # Extracts frequency bins (first column)
    spectrum_centres = data[:, 0]

    # Extracts spectrum (remaining columns)
    spectrum = data[:, 1:]

    return energy_centres, spectrum_centres, np.transpose(spectrum)


# In[6]:


def GetBinEdges(bin_centres):
    """
    Calculates the edges of a set of bins given their centres.

    Parameters:
    - bin_centres: array, the centres of the bins.

    Returns:
    - bin_edges: array, the edges of the bins.
    
    """
    bin_width = bin_centres[1] - bin_centres[0] # Calculates bin width
    
    # Uses half of the bin width to locate the maximum and minimum of the bin range
    E_min = bin_centres[0] - bin_width/2
    E_max = bin_centres[-1] + bin_width/2
    
    # Defines bin edges
    return np.linspace(E_min, E_max, len(bin_centres)+1)


# # Functions

# In[3]:


def AssignBin(E, bins_edges):
    """
    Calculation of bin index for given energy.
    
    Parameters:
    - E: float, the energy that needs assigning to a bin.
    - bin_edges: array, the edges of bins.

    Returns:
    - bin_index: integer, the index of the bin containing E, adjusted to prevent non-real indexes.
    
    """
    bin_index = int((E - bins_edges[0]) / (bins_edges[-1] - bins_edges[0]) * (len(bins_edges)-1))
    return max(0, min(bin_index, len(bins_edges) - 2))


# In[4]:


def TroubleshootFile(troubleshoot_file, progress, iteration):
    """
    Updates the troubleshooting file, giving information on calculation progress.
    
    Parameters:
    - troubleshoot_file: str, the filepath to the troubleshooting file.
    - progress: integer, a percentage value to indicate the progress of the current iteration.
    - iteration: integer, the current iteration of the simulation.
    
    """
    with open(troubleshoot_file, 'w') as f:
        f.write("Calculation progress: " + str(progress) + "%\n")
        f.write("Excitations: " + str(iteration) + "\n")
        f.flush()


# In[5]:


def PrecomputeEmissionProbabilities(spectrum, spectrum_centres):
    """
    Precomputes the emission probabilities and emission times across all internal energies.
    
    Parameters:
    - spectrum: array, the energy-dependent emission spectrum with intensity based on emission cross-section.
    - spectrum_centres: array, the centres of the frequency bins.

    Returns:
    - emission_probabilities: array, the probability of photon emission within each frequency bin at each internal energy.
    - emission_times: array, a constant relative to the amount of time between emissions at each internal energy.
    
    """
    einstein_coefficients = (spectrum_centres*3e10) ** 2 * 8*np.pi/(3e8)**2 # Creates array used to convert intensity in terms of cross section to intensity in terms of Einstein coefficient
    emission_probabilities = []
    emission_times = []

    for energy_idx in range(len(spectrum)):  
        current_spectrum = spectrum[energy_idx]
        if np.sum(current_spectrum) == 0:
            emission_probabilities.append(current_spectrum)  # No emission possible at this energy
            emission_times.append(None) # No calculation of emission time necessary
        else:
            weighted_spectrum = current_spectrum * einstein_coefficients # Applies correction to obtain Einstein coefficient weighted spectrum
            total_rate = np.sum(weighted_spectrum) # Calculates total emission rate at this internal energy
            emission_times.append(1/total_rate) # Calculates total time between emissions at this internal energy
            probabilities = weighted_spectrum / np.sum(weighted_spectrum) # Normalises emission probability
            emission_probabilities.append(probabilities)

    return emission_probabilities, emission_times


# In[7]:


def CascadeEmission(excitation_energy, energy_bins, spectrum_bins, spectrum_centres, spectrum, iterations, troubleshoot_file, time_limit):
    """
    Simulates the cascade emission process theorised to occur to vibrationally-excited molecules in the interstellar medium (ISM).
    
    Parameters:
    - excitation_energy, integer: the initial energy the molecule is excited to.
    - energy_bins: array, the edges of the energy bins.
    - spectrum_bins: array, the edges of the frequency bins.
    - spectrum: array, the energy-dependent emission spectrum.
    - iterations: integer, the number of excitations used to simulate the cascade.
    - troubleshoot_file: str, the filepath to the troubleshooting file.
    - time_limit: integer, the time limit of the simulation in seconds.

    Returns:
    - cascade_spectrum, array: the cascade emission spectrum (an average of all sampled relaxation pathways)
    
    """
    cascade_spectrum = np.zeros(len(spectrum_bins)) # Creates an empty cascade histogram
    last_logged_progress = -1
    emission_probabilities, emission_times = PrecomputeEmissionProbabilities(spectrum, spectrum_centres) # Precomputes emission probabilities and emission times
    
    # Repeats for multiple photon excitations
    for iteration in range(iterations):
        progress = int(100 * iteration / iterations)  # Converts progress to a percentage
        if progress > last_logged_progress:  # Updates only when a new percentage step is reached
            last_logged_progress = progress
            TroubleshootFile(troubleshoot_file, progress, iteration)
        
        # Resets the simulation
        current_energy = excitation_energy
        current_time = 0
        
        # This loop simulates the sequential relaxation of a molecule to its ground state
        while current_energy > 0 and current_time < time_limit:
            energy_idx = AssignBin(current_energy, energy_bins) # Calculates the index of the current internal energy
            current_probabilities = emission_probabilities[energy_idx] # Fetches the emission probabilities at the current internal energy
            
            # Breaks the loop if no further emission can occur to prevent errors
            if not np.any(current_probabilities):
                break
    
            sample_emission = np.random.choice(spectrum_centres, p=current_probabilities) # Samples a random photon emission based on current emission probabilities
            photon_bin = AssignBin(sample_emission, spectrum_bins) # Assigns a frequency bin to the photon emission
            cascade_spectrum[photon_bin] += 1 # Updates cascade histogram at current frequency bin
            current_energy -= sample_emission/8065.541154 # Reduces internal energy to simulate photon emission
            current_time += -np.log(np.random.random()) * emission_times[energy_idx] # Randomly increases current simulation time based on the total emission rate at the current inernal energy
    return cascade_spectrum


# In[8]:


def SaveCascade(filepath, spectrum_centres, cascade_spectrum):
    """
    Saves the cascade emission spectrum to a .txt file.
    
    Parameters:
    - filepath: str, the filepath to save the cascade emission spectrum to.
    - spectrum_centres: array, the centres of the frequency bins.
    - cascade_spectrum: array, the emission intensities associated with each frequency bin.

    """
    with open(filepath, "w") as file:
        # Write header
        file.write("# Wavenumber (cm^-1)   Density\n")
        # Write data
        for frequency, intensity in zip(spectrum_centres, cascade_spectrum):
            file.write("{:}   {:}\n".format(frequency, intensity))


# # Main Function

# In[9]:


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Calculation of an energy-dependent absorption spectrum")
    parser.add_argument("--title", type=str, required=True, help="A title for the calculation.")
    parser.add_argument("--date", type=str, required=True, help="Date of the calculation.")
    parser.add_argument("--spectrum", type=str, required=True, help="A path to the required input file.")
    parser.add_argument("--iterations", type=str, required=True, help="Number of excitations in the simulation.")
    parser.add_argument("--energy", type=str, required=True, help="The excitation energy.")
    parser.add_argument("--time", type=str, required=True, help="The time limit of the simulation.")
    
    # Parse arguments
    args = parser.parse_args()
    date = args.date
    iterations = 10**int(args.iterations)
    excitation_energy = int(args.energy)
    time_limit = int(args.time)
    filepath = "/rds/general/user/ogd21/home/Outputs/energy_dependent_spectra/"+args.spectrum+".csv"
    
    # Extracts data from .csv file (note: this causes a significant slowdown due to the size of the files)
    energy_centres, spectrum_centres, spectrum = LoadEnergyDependentSpectrum(filepath)
    energy_bins = GetBinEdges(energy_centres) # Defines bin edges of energy bins
    spectrum_bins = GetBinEdges(spectrum_centres) # Defines bin edges of frequency bins
    
    # Defines the troubleshooting file
    troubleshoot_file = "/rds/general/user/ogd21/home/Outputs/cascade_spectra/"+date+"/"+args.title+"_"+str(excitation_energy)+"eV_info.txt"
    
    # Simulates cascade emission to generate spectrum
    cascade_spectrum = CascadeEmission(excitation_energy, energy_bins, spectrum_bins, spectrum_centres, spectrum, iterations, troubleshoot_file, time_limit)
    
    # Saves the cascade emission spectrum
    filepath = "/rds/general/user/ogd21/home/Outputs/cascade_spectra/"+date+"/"+args.title+"_"+str(excitation_energy)+"eV.txt"
    SaveCascade(filepath, spectrum_centres, cascade_spectrum)
    
    # Calculate elapsed time
    end_time = time.time()
    elapsed_seconds = int(end_time - start_time)

    # Converts seconds into days, hours, minutes, and seconds
    days, remainder = divmod(elapsed_seconds, 86400)  # 86400 seconds in a day
    hours, remainder = divmod(remainder, 3600)        # 3600 seconds in an hour
    minutes, seconds = divmod(remainder, 60)          # 60 seconds in a minute
    
    # Prints the final calculation information
    with open(troubleshoot_file, 'w') as f:
        f.write("Calculation complete\n")
        f.write("Spectrum source: " + "/rds/general/user/ogd21/home/Outputs/energy_dependent_spectra/"+args.spectrum+".csv" + "\n")
        f.write("Excitations: " + str(iterations) + "\n")
        f.write("Simulation time limit: " + str(time_limit) + ' seconds\n')
        f.write("Calculation started: " + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))) + "\n")
        f.write("Calculation complete: " + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))) + "\n")
        f.write("Elapsed time: {} days, {} hours, {} minutes, {} seconds".format(days, hours, minutes, seconds) + "\n")
        f.flush()

