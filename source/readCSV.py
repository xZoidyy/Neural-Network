import pandas as pd

# Load data from CSV file
filename = '../datas/nudat_data.csv'
data = pd.read_csv(filename)

# Read nudat data into lists 
xZ = data['z']
xN = data['n']
xT1_2 = data['halflife']
xT1_2_units = data['halflifeUnit']
xJpi = data['spinAndParity']
xdecay_mode_complex = data['decayModes']
xmass_excess = data['massExcess(keV)']
xQ_beta_minus = data['betaMinus(keV)']
xQ_alpha = data['alpha(keV)']
xS_2n = data['twoNeutronSeparationEnergy(keV)']
xS_2p = data['twoProtonSeparationEnergy(keV)']
xBE = data['bindingEnergy(keV)']

# List of time units that I allowed
time_units = ['ps', 'ns', 'us', 'ms', 's', 'm', 'h', 'd', 'y']
time_units_equivalent = [pow(10, -12), pow(10, -9), pow(10, -6), pow(10, -3), 1, 60, 3600, 86400, 31536000]
# Decay channels of interest
decay = ['B-', 'EC', 'A'] # in numbers, B- = -1, EC = +1, A = 0

# Truncation of data
Z = []
N = []
T1_2 = []
Jpi = []
mass_excess = []
Q_beta_minus = []
Q_alpha = []
S_2n = []
S_2p = []
BE = []

# take care of decay modes in form: EC = 100.00
decay_mode = []
decay_prob = []
position = []

count = 0 # count of how many nuclei I throw away

# MAKE NEW DATA SET THAT WILL WORKS WITH NEURAL NETWORK AND CREATE INPUT DATA
for i in range(0, len(xN)):
    mode_complex = xdecay_mode_complex[i]
    if isinstance(mode_complex, str):  
        mode_split = mode_complex.split('=')
        if len(mode_split) == 2:
            mode_str = mode_split[0].strip()
            prob_str = mode_split[1].strip()
            if mode_str[:2] in decay and mode_str[2:].strip() == '' and prob_str.replace('.', '', 1).isdigit():
                if isinstance(xT1_2_units[i], str) and any((unit != '' and pd.notna(unit) and str(unit).lower() != 'nan' and unit in time_units) for unit in xT1_2_units): # control only allowed units
                    if float(prob_str) == 100:

                        # Handle NaN values for Q_beta_minus, Q_alpha, S_2n, S_2p
                        if pd.notna(xQ_beta_minus[i]):
                            Q_beta_minus.append(float(xQ_beta_minus[i]))
                        else:
                            continue  # Skip this index i and proceed to the next iteration
                        
                        if pd.notna(xQ_alpha[i]):
                            Q_alpha.append(float(xQ_alpha[i]))
                        else:
                            continue  # Skip this index i and proceed to the next iteration
                        
                        if pd.notna(xS_2n[i]):
                            S_2n.append(float(xS_2n[i]))
                        else:
                            continue  # Skip this index i and proceed to the next iteration
                        
                        if pd.notna(xS_2p[i]):
                            S_2p.append(float(xS_2p[i]))
                        else:
                            continue  # Skip this index i and proceed to the next iteration

                        #print(mode_complex)
                        if (mode_str == 'B-'):
                            decay_mode.append(int(0))
                        elif (mode_str == 'EC'):
                            decay_mode.append(int(0))
                        elif (mode_str == 'A'):
                            decay_mode.append(int(1))
                        decay_prob.append(float(prob_str))
                        position.append(i)

                        # Fill others values
                        Z.append(int(xZ[i]))
                        N.append(int(xN[i]))
                        Jpi.append(xJpi[i])
                        mass_excess.append(float(xmass_excess[i]))
                        #Q_beta_minus.append(float(xQ_beta_minus[i]))
                        #Q_alpha.append(float(xQ_alpha[i]))
                        #S_2n.append(float(xS_2n[i]))
                        #S_2p.append(float(xS_2p[i]))
                        BE.append(float(xBE[i]))

                        for it in range(0, len(time_units)):
                            if time_units[it] == xT1_2_units[i]:
                                T1_2.append(float(xT1_2[i])*float(time_units_equivalent[it]))
                    else:
                        count += 1
                else:
                    count += 1
            else:
                count += 1
        else:
            count += 1
    else:
        count += 1

print(len(Z), "  ", len(N), "  ", len(T1_2), "  ", len(decay_prob), "  ", len(mass_excess), "  ", len(Q_beta_minus), "  ", len(Q_alpha), "  ", len(S_2n), "  ", len(S_2p), "  ", len(BE), "  ", len(decay_mode))

with open('../datas/training_data.txt', 'w') as file:
    # Write the values of the variables separated by tabs
    for i in range(0, len(T1_2), 3):
        file.write(f"1\t{Z[i]}\t{N[i]}\t{T1_2[i]}\t{mass_excess[i]}\t{Q_beta_minus[i]}\t{Q_alpha[i]}\t{S_2n[i]}\t{S_2p[i]}\t{BE[i]}\t{decay_mode[i]}\n")
    for i in range(1, len(T1_2), 3):
        file.write(f"1\t{Z[i]}\t{N[i]}\t{T1_2[i]}\t{mass_excess[i]}\t{Q_beta_minus[i]}\t{Q_alpha[i]}\t{S_2n[i]}\t{S_2p[i]}\t{BE[i]}\t{decay_mode[i]}\n")

with open('../datas/new_data.txt', 'w') as file:
    # Write the values of the variables separated by tabs
    for i in range(2, len(T1_2), 3):
        file.write(f"1\t{Z[i]}\t{N[i]}\t{T1_2[i]}\t{mass_excess[i]}\t{Q_beta_minus[i]}\t{Q_alpha[i]}\t{S_2n[i]}\t{S_2p[i]}\t{BE[i]}\t{decay_mode[i]}\n")

