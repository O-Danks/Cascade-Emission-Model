#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[4]:


import numpy as np
import random
import math
import argparse
import time


# # Data Extraction

# In[5]:


def CouplingConstantToFloat(str_constant):
    """
    Converts the coupling constants from a string in the Gaussian output to a float.
    
    Parameters:
    - str_constant: string, the coupling constant in a string format.

    Returns:
    - coupling_constant: float, the coupling constant in a float format.
    
    """
    temp_float = ''
    temp_power = ''
    power = False
    for character in str_constant:
        if character == 'D':
            power=True
        elif power == True:
            temp_power += character
        elif power == False:
            temp_float += character
    return float(temp_float)*10**int(temp_power)

def AbsorptionCrossSection(dipole_strength, frequency, h = 6.62606896e-34, epsillon0 = 8.854e-12, c = 3e8):
    """
    Converts the dipole strength to the absorption cross section.
    
    Parameters:
    - dipole_strength: float, the dipole strength, extracted from Gaussian output.
    - frequency: float, the harmonic frequency of the mode, extracted from Gaussian output.

    Returns:
    - cross_section: float, the absorption cross section (0>1) of the mode.
    
    """
    # Convert the input to standard units
    dipole_strength = dipole_strength * 1.113 * 1e-63
    frequency = frequency * 3e10
    
    # Returns the absorption cross section
    return (dipole_strength * 2* np.pi**2 * frequency) / (3 * h * epsillon0 * c)

def ExtractGaussianData(file_path):
    """
    Extracts frequencies, coupling constants and dipole strengths from a Gaussian .log file.
    
    Parameters:
    - file_path: str, the filepath to the Gaussian .log file.

    Returns:
    - vibrational modes: array, a list of vibrational modes including labels, frequencies and dipole strengths.
    - coupling_matrix: array, a square matrix containing the anharmonic coupling constants
    
    """
    record_data = False # Controls which lines of data are read
    section = "" # Controls how certain sections are recorded
    vibrational_modes = []
    raw_matrix = []
    
    # Selects the correct section depending on the heading
    with open(file_path, "r") as data:
        for line in data:
            if "Dipole strengths (DS) in 10^-40 esu^2.cm^2" in line:
                record_data = True
            if "Total Anharmonic X Matrix (in cm^-1)" in line:
                record_data = True
                section = "Coupling Matrix"
                
            if "Fundamental Bands" in line:
                section = "Fundamental Bands"
            elif "Overtones" in line:
                section = "Overtones"
            elif "Combination Bands" in line:
                section = "Combination Bands"
            
            # Alters how data is recorded depending on the heading
            if record_data == True:
                if section == "Fundamental Bands":
                    values = line.split()
                    if len(values) == 5 and "Mode(n)" not in values:
                        mode = values[0]
                        freq = float(values[1])
                        dipole_strength = float(values[4])
                        vibrational_modes.append((mode, freq, dipole_strength))
                        
                if section == "Overtones":
                    values = line.split()
                    if len(values) == 4 and "Mode(n)" not in values:
                        mode = values[0]
                        freq = float(values[1])
                        dipole_strength = float(values[3])
                        vibrational_modes.append((mode, freq, dipole_strength))
                        
                if section == "Combination Bands":
                    values = line.split()
                    if len(values) == 5 and "Mode(n)" not in values:
                        mode = values[0]+'+'+values[1]
                        freq = float(values[2])
                        dipole_strength = float(values[4])
                        vibrational_modes.append((mode, freq, dipole_strength))
                    elif 'Berny' in values:
                        section = ""
                
                if section == "Coupling Matrix":
                    values = line.split()
                    raw_matrix.append(values)
                    
                    if "=" in line:
                        record_data = False
        
        raw_matrix=raw_matrix[2:-2] # Removes empty lines from the coupling matrix
        
        # Counts the number of fundamental modes to set the size of the coupling matrix
        fundamental_modes = []
        column_indexes = [0, 1, 2, 3, 4]
        for line in raw_matrix:
            if not line[0] in fundamental_modes:
                fundamental_modes.append(line[0])
        
        column_indexes = [0, 1, 2, 3, 4]
        coupling_matrix = [[0.0 for _ in range(len(fundamental_modes))] for _ in range(len(fundamental_modes))]
        
        for line in raw_matrix:
            if line[-1].isdigit(): # Recognises this is a new row of headers and alters the current indexes accordingly
                for i in range(len(line)):
                    column_indexes[i] = int(line[i])
            else:
                row_index = int(line[0]) # Updates the coupling matrix at the current row based on the current column headers
                for i in range(1, len(line)):
                    coupling_matrix[row_index-1][column_indexes[i-1]-1] = CouplingConstantToFloat(line[i])
                    coupling_matrix[column_indexes[i-1]-1][row_index-1] = CouplingConstantToFloat(line[i])
            
                    
        return vibrational_modes, coupling_matrix


# In[6]:


def LoadDOS(filepath):
    """
    Loads the vibrational density of states (VDOS) from a .txt file.
    
    Parameters:
    - file_path: str, the filepath to the DOS .txt file.

    Returns:
    - energy_centres: array, the centres of the energy bins.
    - dos: array, the logathimic DOS.
    
    """
    append_data = False
    energy_centres = []
    dos = []
    
    with open(filepath, "r") as data:
        for line in data:
            if append_data == True: # Reads only the data, not the heading
                values = line.split()
                energy_centres.append(float(values[0]))
                dos.append(float(values[1]))
            else:
                append_data = True
    
    dos = NormaliseDOS(dos)
    return energy_centres, dos


# In[7]:


def NormaliseDOS(dos):
    """
    Normalises the DOS so that the density of the ground state is 1.
    
    Parameters:
    - dos: array, the logathimic DOS.

    Returns:
    - dos: array, the logathimic DOS (normalised).
    
    """
    dos_0 = dos[0] # Fetches the value of the DOS at the ground state
    for i in range(len(dos)):
        dos[i] -= dos_0  # Normalises the DOS at all positions
    return dos


# # Functions

# In[8]:


def ComputeStateEnergycm(state, frequencies, anharmonic_coupling):
    """
    Computes the energy of a given state in wavenumbers.

    Parameters:
    - state: array, the quantum numbers of the state.
    - frequencies: array, the harmonic frequencies, as calculated by Gaussian.
    - anharmonic_coupling: array, the anharmonic coupling matrix, as calculated by Gaussian.

    Returns:
    - state_energy: float, the energy of the state in wavenumbers.
    
    """
    state = np.asarray(state, dtype=np.float64)  # Convert to NumPy array
    v = state + 0.5  # Precompute (state[i] + 0.5)

    # First term: Sum of frequency contributions
    state_energy = np.dot(frequencies, v)

    # Second term: Anharmonic coupling, summing only over i >= j
    lower_triangle = np.triu(anharmonic_coupling)
    state_energy += np.sum(lower_triangle * np.outer(v, v))

    return state_energy


# In[9]:


def GenerateRandomState(frequencies, anharmonic_coupling, E_max, steps=1000):
    """
    Generates a random state whose energy is below the maximum energy of the simulation. This is done using a Monte Carlo walk.

    Parameters:
    - frequencies: array, the harmonic frequencies, as calculated by Gaussian.
    - anharmonic_coupling: array, the anharmonic coupling matrix, as calculated by Gaussian
    - E_max: integer, the maximum energy of the simulation.

    Returns:
    - state: array, the quantum numbers of the state.

    """
    num_modes = len(frequencies)
    state = [0] * num_modes
    E_min = ComputeStateEnergycm(state, frequencies, anharmonic_coupling) # Sets minimum energy at the ground state
    random_max = random.randint(int(E_min) + 1, int(E_max)) # Randomly shifts the maximum random state energy to ensure fairer sampling of energies

    for step in range(steps):
        # Step 1: Propose a new state
        new_state = list(state)
        mode_to_change = random.randint(0, num_modes - 1)
        new_state[mode_to_change] += random.choice([-1, 1])  # Increment or decrement

        # Step 2: Ensure quantum numbers remain non-negative
        if new_state[mode_to_change] < 0:
            continue

        # Step 3: Compute energy of the proposed state
        current_energy = ComputeStateEnergycm(state, frequencies, anharmonic_coupling)
        proposed_energy = ComputeStateEnergycm(new_state, frequencies, anharmonic_coupling)

        # Step 4: Accept or reject the new state
        if proposed_energy <= random_max:
            state = new_state  # Accept the proposed state

    # Return the final random state
    return state


# In[10]:


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


# In[11]:


def GetUpperBounds(frequencies, anharmonic_coupling):
    """
    Calculates the maximum quantum number for each mode during the walk.
    
    Parameters:
    - frequencies: array, the harmonic frequencies, as calculated by Gaussian.
    - anharmonic_coupling: array, the anharmonic coupling matrix, as calculated by Gaussian.

    Returns:
    - max_n: array, the maximum quantum number for each mode.
    
    """
    max_n = []
    for i in range(len(frequencies)):
        max_n.append(int((-frequencies[i]/np.sum(anharmonic_coupling[i])) - 0.5))
        if max_n[i] < 0:
            max_n[i] = 999 # If max_n is negative, sets the maximum quantum number to 999 as an arbitrarily large value
    return max_n


# In[12]:


def ModifyState(current_state, prob_up_down, max_n):
    """
    Modifies the current state via a walk in the space of quantum numbers.
    
    Parameters:
    - current_state: array, the current quantum numbers.
    - prob_up_down: float, the probability of a state increasing (or decreasing) by one quanta.
    
    """
    change = np.random.choice([-1, 0, 1], size=len(current_state), p=[prob_up_down, 1 - 2*prob_up_down, prob_up_down]) #Proposes a state change
    new_state = np.maximum(current_state + change, 0)  # Ensure n >= 0
    new_state = np.minimum(new_state, max_n) # Ensures n <= max_n

    return new_state


# In[13]:


def TroubleshootFile(troubleshoot_file, lowest_energy_reached, highest_energy_reached, current_visits, average_visits):
    """
    Updates the troubleshooting file, giving information on calculation progress.
    
    Parameters:
    - troubleshoot_file: str, the filepath to the troubleshooting file.
    - lowest_energy_reached: float, the lowest energy reached by the Wang-Landau walk.
    - highest_energy_reached: float, the highest energy reached by the Wang-Landau walk.
    - current_visits: integer, the current average visits per energy bin.
    - average_visits: integer, the desired final average visits per energy bin.
    
    """
    with open(troubleshoot_file, 'w') as f:
        f.write("Calculation progress: " + str(100*current_visits/average_visits) + "%\n")
        f.write("Average internal energy bin visits: " + str(current_visits) + "\n")
        f.write("Lowest energy reached: " + str(lowest_energy_reached) + " eV\n")
        f.write("Highest energy reached: " + str(highest_energy_reached) + " eV\n")
        f.flush()


# In[14]:


def ComputeAbsorptionTransitionEnergy(frequencies, anharmonic_coupling, state, mode):
    """
    Calculates the transition energy for increasing the quantum number of a specific mode by 1.

    Parameters:
    - frequencies: array, the anharmonic frequencies, as calculated by Gaussian.
    - anharmonic_coupling: array, the anharmonic coupling matrix, as calculated by Gaussian.
    - state: array, the quantum numbers of the state.
    - mode: integer, the index of a specific mode.
    
    Returns:
    - delta_e: float, the transition energy.
    
    """
    state = np.asarray(state, dtype=np.float64)  # Converts to NumPy array
    v = state + 0.5  # Precomputes (state[i] + 0.5)

    # Starts with the frequency contribution of the specified mode
    delta_e = frequencies[mode] + 2 * anharmonic_coupling[mode][mode] * (state[mode] + 1)

    # Adds the anharmonic coupling contributions from other modes
    delta_e += np.sum(np.array(anharmonic_coupling[mode]) * (v)) - anharmonic_coupling[mode][mode] * (v[mode])

    return delta_e

def ComputeEmissionTransitionEnergy(frequencies, anharmonic_coupling, state, mode):
    """
    Calculates the transition energy for decreasing the quantum number of a specific mode by 1.

    Parameters:
    - frequencies: array, the anharmonic frequencies, as calculated by Gaussian.
    - anharmonic_coupling: array, the anharmonic coupling matrix, as calculated by Gaussian.
    - state: array, the quantum numbers of the state.
    - mode: integer, the index of a specific mode.
    
    Returns:
    - delta_e: float, the transition energy.
    
    """
    state = np.asarray(state, dtype=np.float64)  # Converts to NumPy array
    v = state + 0.5  # Precomputes (state[i] + 0.5)

    # Starts with the frequency contribution of the specified mode
    delta_e = frequencies[mode] + 2 * anharmonic_coupling[mode][mode] * (state[mode])

    # Adds the anharmonic coupling contributions from other modes
    delta_e += np.sum(np.array(anharmonic_coupling[mode]) * (v)) - anharmonic_coupling[mode][mode] * (v[mode])

    return delta_e


# In[15]:


def UpdateSpectrum(spectrum, spectrum_centres, spectrum_bins, transition_energy, cross_section, quantum, sigma=0.1, cutoff=3):
    """
    Efficiently update the spectrum using a Gaussian-spread Dirac delta function.
    
    Parameters:
    - spectrum: array, a 2D absorption/emission spectrum that varies with the internal energy of the system.
    - spectrum_centres: array, the centres of each frequency bin.
    - spectrum_bins: array, the edges of the frequency bins.
    - transition_energy: integer, the index of a specific mode.
    - cross_section: float, the absorption/emission cross section of the current mode.
    - quantum: int, the quantum number of the current mode.
    
    """
    
    inv_2sigma2 = 1 / (2 * sigma ** 2)  # Precompute constant
    
    # Find relevant bins within the cutoff range
    lower_bound = transition_energy - cutoff * sigma
    upper_bound = transition_energy + cutoff * sigma

    start_idx = AssignBin(lower_bound, spectrum_bins)
    end_idx = AssignBin(upper_bound, spectrum_bins)

    # Compute Gaussian weights only for the affected bins
    relevant_bins = spectrum_centres[start_idx:end_idx]
    diff = relevant_bins - transition_energy
    weights = np.exp(-diff * diff * inv_2sigma2)
    weights /= np.sum(weights)  # Normalise

    # Apply the update in place to avoid unnecessary array creation
    spectrum[start_idx:end_idx] += weights * (quantum * cross_section)


# In[18]:


def GetEnergyDependentSpectrum(frequencies, anharmonic_coupling, cross_sections, dos, energy_bins, spectrum_bins, average_visits, ground_state_energy, troubleshoot_file, calc_type):
    """
    Calculates an energy dependent absorption spectrum using a Wang-Landau Monte Carlo method.

    Parameters:
    - frequencies: array, the harmonic frequencies, as calculated by Gaussian.
    - anharmonic_coupling: array, the anharmonic coupling matrix, as calculated by Gaussian.
    - cross_sections: array, the absorption/emission cross sections associated with each vibrational mode, as calculated from Gaussian output.
    - dos: array, the normalised logarithmic density of states (DOS).
    - energy_bins: array, the edges of the energy bins.
    - spectrum_bins: array, the edges of the frequency bins.
    - average_visits: integer, the number of average visits to each internal energy bin before the simulation considers the spectrum converged.
    - ground_state_energy: float, the energy of the system when all vibrational quantum numbers are zero.
    - troubleshoot_file: str, the filepath to the troubleshooting file.
    - calc_type: str, the type of simulation (absorption/emission)

    Returns:
    - spectrum: a 2D absorption/emission spectrum that varies with the internal energy of the system.

    """
    E_max = energy_bins[-1] # Sets upper energy limit
    E_min = energy_bins[0] # Sets lower energy limit
    spectrum = np.zeros((len(energy_bins)-1, len(spectrum_bins)-1)) # Initialises the spectrum histogram
    prob_up_down = 0.05 # Sets the probability of quantum numbers changing in the Wang-Landau walk
    hist = np.zeros(len(energy_bins)-1) # Initialises the histogram used to track visits to internal energies
    spectrum_centres = (spectrum_bins[:-1] + spectrum_bins[1:]) / 2
    max_n = GetUpperBounds(frequencies, anharmonic_coupling)
    
    current_state = GenerateRandomState(frequencies, anharmonic_coupling, E_max, steps=1000) # Starts at a random state with a random energy
    initial_energy = ComputeStateEnergycm(current_state, frequencies, anharmonic_coupling)
    current_bin = AssignBin(initial_energy, energy_bins)
    
    #Troubleshooting variables
    lowest_energy_reached = initial_energy
    highest_energy_reached = initial_energy
    last_logged_progress = -1
    
    while np.mean(hist) < average_visits: # Ensures all bins are visited by a minimum average number of times to promote convergence towards true spectrum
        progress = int(100 * np.mean(hist) / average_visits)  # Converts calculation progress to a percentage
        if progress > last_logged_progress:  # Updates only when a new percentage step is reached
            last_logged_progress = progress
            TroubleshootFile(troubleshoot_file, (lowest_energy_reached-ground_state_energy)/8065.541154, (highest_energy_reached-ground_state_energy)/8065.541154, np.mean(hist), average_visits)
        
        # Proposes a new state by modifying the vibrational quantum numbers
        new_state = ModifyState(current_state, prob_up_down, max_n)     
        new_energy = ComputeStateEnergycm(new_state, frequencies, anharmonic_coupling)

        if E_min <= new_energy <= E_max: # Only considers new states within the energy range
            new_bin = AssignBin(new_energy, energy_bins)
            P_accept = min(1, np.exp(dos[current_bin]-dos[new_bin])) # Calculates an acceptance probability based upon the normalised DOS
            if random.random() < P_accept:
                if new_energy < lowest_energy_reached: # Verifies if a new lowest energy has been reached
                    lowest_energy_reached = new_energy
                elif new_energy > highest_energy_reached: # Verifies if a new highest energy has been reached
                    highest_energy_reached = new_energy
                current_bin = new_bin
                current_state = list(new_state)  # Accept new state
                
        # Updates the spectrum at this energy
        for mode in range(len(current_state)):
            
            # Updates at the correct transition_energy for absorption/emission in the current mode
            if calc_type == "emission":
                transition_energy = ComputeEmissionTransitionEnergy(frequencies, anharmonic_coupling, current_state, mode)
            else:
                transition_energy = ComputeAbsorptionTransitionEnergy(frequencies, anharmonic_coupling, current_state, mode)
            
            # Updates the spectrum, accounting for change in cross section due to quantum number
            if calc_type == "emission":
                UpdateSpectrum(spectrum[current_bin], spectrum_centres, spectrum_bins, transition_energy, cross_sections[mode], current_state[mode])
            else:
                UpdateSpectrum(spectrum[current_bin], spectrum_centres, spectrum_bins, transition_energy, cross_sections[mode], current_state[mode]+1)
        hist[current_bin]+=1
        
    # Normalises the spectrum based on the number of visits to each internal energy bin
    for x in range(len(energy_bins)-1):
        for y in range(len(spectrum_bins)-1):
            if hist[x] > 0:
                spectrum[x,y]=spectrum[x,y]/hist[x]
                
    return spectrum


# In[17]:


def SaveEnergyDependentSpectrum(filepath, energy_centres, spectrum_bins, spectrum):
    """
    Saves the energy dependent spectrum to a specific filepath.

    Parameters:
    - filepath: string, the filepath that the spectrum is saved to.
    - energy_centres: array, the centres of each energy bin.
    - spectrum_bins: array, the edges of the frequency bins.
    - spectrum: a 2D absorption/emission spectrum that varies with the internal energy of the system.

    """
    # Saves as a CSV
    header = "Frequency/Energy," + ",".join(map(str, energy_centres))  # Adds energy bin centres as header
    frequency_centres = (spectrum_bins[:-1] + spectrum_bins[1:]) / 2 # Creates an array containing the frequency bin centres

    with open(filepath, 'w') as f:
        f.write(header + "\n")
        for energy, freq in enumerate(frequency_centres):
            row = [freq] + list(spectrum[:, energy])  # Records frequency and corresponding spectrum intensities
            f.write(",".join(map(str, row)) + "\n")


# # Main Function

# In[21]:


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Calculation of an energy-dependent absorption spectrum")
    parser.add_argument("--title", type=str, required=True, help="A title for the calculation.")
    parser.add_argument("--data", type=str, required=True, help="A path to the required input file.")
    parser.add_argument("--date", type=str, required=True, help="Date of the calculation.")
    parser.add_argument("--dos", type=str, required=True, help="A path to the DOS file.")
    parser.add_argument("--visits", type=str, required=True, help="Average number of visits to each internal energy bin.")
    parser.add_argument("--type", type=str, required=True, help="The type of spectrum generated (absorption or emission).")
    
    # Parse arguments
    args = parser.parse_args()
    dos_filepath = "/rds/general/user/ogd21/home/Outputs/dos/"+args.dos+".txt"
    date = args.date
    visits = int(args.visits)
    filepath = "/rds/general/user/ogd21/home/Outputs/log_files/"+args.data+".log"

    # Extracts Gaussian data
    vibrational_modes, coupling_matrix = ExtractGaussianData(filepath)
    
    # Processes Gaussian data
    anharmonic_coupling = coupling_matrix
    frequencies = [item[1] for item in vibrational_modes]
    frequencies = frequencies[:len(anharmonic_coupling)] #this code extracts only the fundamental modes for testing purposes
    dipole_strengths = [item[2] for item in vibrational_modes]
    
    # Calculates absorption cross sections
    cross_sections = []
    for i in range(len(frequencies)):
        cross_sections.append(AbsorptionCrossSection(dipole_strengths[i], frequencies[i]))
    
    # Calculates ground state energy
    ground_state = np.zeros(len(frequencies))
    ground_state_energy = ComputeStateEnergycm(ground_state, frequencies, anharmonic_coupling)
    
    # Loads the DOS
    energy_centres, dos = LoadDOS(dos_filepath)
    bin_width = energy_centres[1] - energy_centres[0]
    E_min = energy_centres[0] - bin_width/2
    E_max = energy_centres[-1] + bin_width/2
    
    # Defines bin edges
    energy_bins = np.linspace(E_min, E_max, len(energy_centres)+1)
    
    # Sets of up the frequency bins
    spectrum_bins = np.linspace(0, 3500, 35001)
    
    # Sets troubleshooting filepath
    troubleshoot_file = "/rds/general/user/ogd21/home/Outputs/energy_dependent_spectra/"+date+"/"+args.title+"_"+args.visits+"av_visits.txt"
    
    # Simulates the energy-dependent spectrum
    spectrum = GetEnergyDependentSpectrum(frequencies, anharmonic_coupling, cross_sections, dos, energy_bins, spectrum_bins, visits, ground_state_energy, troubleshoot_file, args.type)
    
    # Saves the energy-dependent spectrum
    filepath = "/rds/general/user/ogd21/home/Outputs/energy_dependent_spectra/"+date+"/"+args.title+"_"+args.visits+"av_visits.csv"
    SaveEnergyDependentSpectrum(filepath, energy_centres, spectrum_bins, spectrum)
    
    # Calculate elapsed time
    end_time = time.time()
    elapsed_seconds = int(end_time - start_time)

    # Convert seconds into days, hours, minutes, and seconds
    days, remainder = divmod(elapsed_seconds, 86400)  # 86400 seconds in a day
    hours, remainder = divmod(remainder, 3600)        # 3600 seconds in an hour
    minutes, seconds = divmod(remainder, 60)          # 60 seconds in a minute
    
    # Prints the final calculation information
    with open(troubleshoot_file, 'w') as f:
        f.write("Calculation complete\n")
        f.write("Data source : " + "/rds/general/user/ogd21/home/Outputs/log_files/"+args.data+".log" + "\n")
        f.write("DOS source: " + "/rds/general/user/ogd21/home/Outputs/dos/"+args.dos+".txt" + "\n")
        f.write("Calculation started: " + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))) + "\n")
        f.write("Calculation complete: " + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))) + "\n")
        f.write("Elapsed time: {} days, {} hours, {} minutes, {} seconds".format(days, hours, minutes, seconds) + "\n")
        f.flush()

