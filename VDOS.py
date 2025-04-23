#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[44]:


import numpy as np
import random
import math
import argparse
import time


# # Data Extraction

# In[45]:


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


# # Functions

# In[46]:


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


# In[47]:


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


# In[48]:


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


# In[89]:


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


# In[83]:


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


# In[50]:


def TroubleshootFile(troubleshoot_file, progress, iteration):
    """
    Updates the troubleshooting file, giving information on calculation progress.
    
    Parameters:
    - troubleshoot_file: str, the filepath to the troubleshooting file.
    - progress: integer, a percentage value to indicate the progress of the current iteration.
    - iteration: integer, the current iteration of the simulation.
    
    """ 
    with open(troubleshoot_file, 'w') as f:
        f.write("Iteration: " + str(iteration+1) + " of 20\n")
        f.write("Iteration Progress: " + str(progress) + "%\n")
        f.flush()


# In[51]:


def CheckFlatness(histogram):
    """
    Checks if the histogram has achieved the flatness criterion.
    
    Parameters:
    - histogram: array, a histogram to be checked for flatness.

    Returns:
    - bool, whether or not the flatness criterion has been achieved.
    
    """
    nonzero_bins = histogram[histogram > 0]  # Exclude empty bins

    #if len(nonzero_bins) < 0.9*len(histogram):  # Prevent division by zero if all bins are empty
    #    return False

    if np.min(nonzero_bins)/np.mean(nonzero_bins) >= 0.3:
        return True  # Flatness reached if all bins are within 70% of the mean
    else:
        return False


# In[91]:


def GetDOS(frequencies, anharmonic_coupling, energy_bins, steps, troubleshoot_file, filepath, f_init=math.e, iterations=20, prob_up_down=0.01):
    """
    Performs a Wang-Landau Monte Carlo simulation to calculate the vibrational density of states (VDOS).
    
    Parameters:
    - frequencies: array, the harmonic frequencies, as calculated by Gaussian.
    - anharmonic_coupling: array, the anharmonic coupling matrix, as calculated by Gaussian.
    - energy_bins: array, the edges of the energy bins.
    - steps: integer, number of steps in each iteration.
    - troubleshoot_file: str, the filepath to the troubleshooting file.
    - filepath: str, the output filepath for intermediate VDOS and histograms.
    - f_init: float, initial modification factor.
    - iterations: integer, number of Wang-Landau iterations.
    - prob_up_down: float, probability of quantum numbers changing.
    
    Returns:
    - dos: array, logarithmic density values corresponding to each energy bin.
    
    """
    E_max = energy_bins[-1] # Sets upper energy limit
    E_min = energy_bins[0] # Sets lower energy limit
    ln_dos = np.zeros(len(energy_bins)-1)  # Stores the logarithm of DOS to avoid overflow
    log_f = np.log(f_init)  # Initial modification factor
    max_n = GetUpperBounds(frequencies, anharmonic_coupling)

    for iteration in range(iterations):
        hist = np.array(np.zeros(len(energy_bins)-1))
        current_state = GenerateRandomState(frequencies, anharmonic_coupling, E_max, steps=1000) # Begins the iteration at a random state with a random energy
        current_energy = ComputeStateEnergycm(current_state, frequencies, anharmonic_coupling)
        current_bin = AssignBin(current_energy, energy_bins)
    
        last_logged_progress = -1    
        for step in range(steps): # Repeats the walk for a given number of steps per iteration
            
            # Propose a new state by modifying the vibrational quantum numbers
            new_state = ModifyState(current_state, prob_up_down, max_n)

            # Calculate the energy of the proposed state
            new_energy = ComputeStateEnergycm(new_state, frequencies, anharmonic_coupling)
            
            if E_min <= new_energy <= E_max: # Only considers new states within the energy range    
                new_bin = AssignBin(new_energy, energy_bins) # Calculates the bin index of the new energy
                P_accept = min(1, np.exp(ln_dos[current_bin] - ln_dos[new_bin])) # Calculates the acceptance probability

                # Accept the new state with probability P_accept
                if random.random() < P_accept:    
                    current_state = list(new_state)
                    current_bin = new_bin

            ln_dos[current_bin] = ln_dos[current_bin] + log_f  # Update the logarithm of DOS
            hist[current_bin] += 1
            
            progress = int(100 * step / steps)  # Converts progress to a percentage
            if progress > last_logged_progress:  # Checks flatness and updates troubleshooting file only when a new percentage step is reached
                last_logged_progress = progress
                TroubleshootFile(troubleshoot_file, progress, iteration) # Prints additional troubleshooting data
                if CheckFlatness(hist) == True and progress > 0: # Flatness criterion
                    break

        # Update the modification factor
        current_filepath = filepath + "_iteration" + str(iteration + 1) + ".txt"
        SaveDOS(ln_dos, energy_bins, current_filepath) # Saves intermediate VDOS file
        current_filepath = filepath + "_hist" + str(iteration + 1) + ".txt"
        SaveDOS(hist, energy_bins, current_filepath) # Saves intermediate histogram
        log_f *= 0.5

    return ln_dos


# In[53]:


def SaveDOS(dos, energy_bins, filepath):
    """
    Saves density of states (DOS) data to a text file.

    Parameters:
    - dos: array, normalised density values corresponding to each energy bin.
    - energy_bins: array, the edges of the energy bins.
    - filepath: string, the filepath of the data to be saved.

    """
    
    # Compute bin centres
    bin_centres = (energy_bins[:-1] + energy_bins[1:]) / 2
    
    with open(filepath, "w") as file:
        # Write header
        file.write("# Energy (cm^-1)   Density\n")
        # Write data
        for energy, density in zip(bin_centres, dos):
            file.write("{}   {}\n".format(energy, density))


# # Main Function

# In[10]:


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Calculation of an energy-dependent absorption spectrum")
    parser.add_argument("--title", type=str, required=True, help="A title for the calculation.")
    parser.add_argument("--data", type=str, required=True, help="A path to the required input file.")
    parser.add_argument("--date", type=str, required=True, help="Date of the calculation.")
    parser.add_argument("--dos_steps", type=str, required=True, help="How many max steps should be considered per cycle?")
    
    # Parse arguments
    args = parser.parse_args()
    filename = args.data
    date = args.date
    steps = 10**int(args.dos_steps)
    filepath = "/rds/general/user/ogd21/home/Outputs/log_files/"+filename+".log"

    # Extracts Gaussian data
    vibrational_modes, coupling_matrix = ExtractGaussianData(filepath)
    
    # Processes Gaussian data
    anharmonic_coupling = coupling_matrix
    frequencies = [item[1] for item in vibrational_modes]
    frequencies = frequencies[:len(anharmonic_coupling)] # This code extracts only the fundamental frequencies only
    
    # Calculates ground_state energy
    ground_state = np.zeros(len(frequencies))
    ground_state_energy = ComputeStateEnergycm(ground_state, frequencies, anharmonic_coupling)
    
    # Defines simulation energy bounds
    E_min= ground_state_energy
    E_max= ground_state_energy + 8065.541154 * 9
    
    # Defines bin edges
    energy_bins = np.linspace(E_min, E_max, 3001)
    
    # Sets troubleshooting filepath
    troubleshoot_file = "/rds/general/user/ogd21/home/Outputs/dos/"+date+"/"+args.title+"_"+args.dos_steps+"steps_info.txt"
    
    # Calculates the DOS
    filepath = "/rds/general/user/ogd21/home/Outputs/dos/"+date+"/"+args.title+"_"+args.dos_steps+"steps"
    dos = GetDOS(frequencies, anharmonic_coupling, energy_bins, steps, troubleshoot_file, filepath) #calculates the density of states
    
    # Saves the final DOS
    filepath += ".txt"
    SaveDOS(ln_dos, energy_bins, filepath)
    
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
        f.write("Data source : " + "/rds/general/user/ogd21/home/Outputs/log_files/"+filename+".log" + "\n")
        f.write("Calculation started: " + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))) + "\n")
        f.write("Calculation complete: " + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))) + "\n")
        f.write("Elapsed time: {} days, {} hours, {} minutes, {} seconds".format(days, hours, minutes, seconds) + "\n")
        f.flush()

