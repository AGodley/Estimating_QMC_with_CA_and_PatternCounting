############
# Patterns #
############

# Standard imports
import numpy as np
from qutip import *
from numpy.linalg import pinv

# For generating all sequences of 0s and 1s
import itertools


# Generates all binary combinations up to order+2
def possible_patterns(order):
    result = []

    # Finds all combinations of 0s and 1s up to order
    for r in range(1, order+1):
        combinations = itertools.product([0, 1], repeat=r)
        # Result is a list of tuples containing the patterns
        result.extend(combinations)

    # Initializes the dictionary that will contain the patterns
    dictionary = {}
    # Adds first two patterns
    dictionary['1'] = 0
    dictionary['11'] = 0

    # Converts each tuple into a string pattern, adds a 1 either side and adds it to a dictionary
    for tuple in result:
        pat = '1'
        for i in tuple:
            pat += str(i)
        pat += '1'
        dictionary[pat] = 0
    return dictionary


# Checks for patterns in the output list
def pattern_check(ones_list):
    # Creates the dictionary that stores the analysis
    order_patterns = 6
    analysis_x = possible_patterns(order_patterns)

    # # Finds the indices of all ones
    # ones_loc = []
    # for i in range(len(ones_list)):
    #     if ones_list[i] == 1:
    #         ones_loc.append(i)

    # Counts the 1s in all observed patterns
    weighted_sum_patterns = 0

    # Loop runs over each pattern in the dictionary above
    for key in analysis_x.keys():
        # This converts the string in the dict to the actual pattern found in the trajectory
        pattern = [int(num) for num in key]

        # Then adds a number of padding zeroes either side of the pattern
        pad = 10
        for i in range(pad):
            pattern.reverse()
            pattern.append(0)
            pattern.reverse()
            pattern.append(0)

        # Loops over the trajectory and identifies how many times the pattern occurs; this misses the tails of the list
        for i in range(len(ones_list) - len(pattern)):
            if ones_list[i:i+len(pattern)] == pattern:
                analysis_x[key] += 1
                weighted_sum_patterns += np.sum(pattern)/len(ones_list)

    return analysis_x, weighted_sum_patterns


def expected(stationary_state, ks, local_u, n_final):
    # Creates the dictionary that stores the analysis
    order_patterns = 6
    expected_x = possible_patterns(order_patterns)

    # Stores Fisher information calculation
    FI_patterns = 0

    # Calculates the mpn from a sum over expected patters
    expected_mpn = 0

    # Sum of mu terms squared
    sum_mus = 0

    # Loop runs over each pattern in the dictionary above
    for key in expected_x.keys():
        # This converts the string in the dict to the actual pattern found in the trajectory
        pattern = [int(num) for num in key]

        # Sets the mu to the stationary state for the formula
        mu_pattern = stationary_state

        # Creates a superoperator for the transition operator
        T = sprepost(ks[0], ks[0].dag()) + sprepost(ks[1], ks[1].dag())
        # Creates a superoperator for the other term in the formula
        J = sprepost(ks[0], ks[1].dag())

        # Runs through the pattern applying either T or J depending on the outcome 0 or 1 respectively
        for i in pattern:
            if i == 0:
                mu_pattern = T(mu_pattern)
            elif i == 1:
                mu_pattern = J(mu_pattern)
            else:
                raise "How'd you get here?"

        expected_x[key] = np.abs(mu_pattern.tr()) ** 2 * n_final
        # print(expected_x[key], mu_pattern.tr(), len(ones_list))

        # Cumulative sum of mus^2 updated
        sum_mus += 4 * np.abs(mu_pattern.tr()) ** 2 * np.sqrt(n_final)

        # Adds the number of photons to the total sum of photons in detected patterns
        expected_mpn += expected_x[key] / n_final * np.sum(pattern)

        # FI
        FI_patterns += 4 * expected_x[key] / n_final / (abs(local_u)) ** 2
    # Print statements
    # print(f'Analytical result for the m.p.n: {expected_mpn}')
    # print(f'Fisher information from patterns: {FI_patterns}')
    return expected_x, expected_mpn


def alternative(stationary_state_at_theta_rough, ks, ks_dot, local_u, n_final):
    # Alternative formula for Poisson rates |mu|^2
    # Creates the dictionary that stores |mu|^2
    order_patterns = 6
    alt_mu = possible_patterns(order_patterns)

    # Fisher information calculation
    alt_FI = 0

    # Dictionary for actual expected counts
    alt_expected = alt_mu.copy()

    # Loop runs over each pattern in the dictionary above
    for key in alt_expected.keys():
        # This converts the string in the dict to the actual pattern found in the trajectory
        pattern = [int(num) for num in key]

        # Sets the mu to the stationary state for the formula
        ss = stationary_state_at_theta_rough

        # Finds parts of each term that doesn't change with pattern
        inverse = Qobj(np.linalg.pinv(qeye([2, 2]) - ks[0]), dims=[[2, 2], [2, 2]])
        term_1 = ks[1] * inverse * ks_dot[0]
        term_2 = ks_dot[1]

        # Handles 1 pattern
        if key == '1':
            # Updates mus
            alt_mu[key] = np.abs((ss * term_1.dag()).tr() + (ss * term_2.dag()).tr()) ** 2
        # Handles other patterns
        else:
            # Adds product of Kraus' to this
            for i in pattern[1:]:
                term_1 = ks[i] * term_1
                term_2 = ks[i] * term_2

            # Updates mus
            alt_mu[key] = np.abs((ss * term_1.dag()).tr() + (ss * term_2.dag()).tr()) ** 2

        # Updates expected counts
        alt_expected[key] = alt_mu[key] * local_u ** 2 * n_final

        # FI
        alt_FI += 4 * alt_mu[key]
    # Print statements
    # print(f'Analytical result for the m.p.n: {expected_mpn}')
    # print(f'Fisher information from patterns: {FI_patterns}')
    return alt_mu, alt_FI, alt_expected


if __name__ == '__main__':
    x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pattern_check(x)
    # Generating all possible patterns up to length j+2
    j = 6
    result = possible_patterns(j)
    # print(result)
    output = ','
    print(output.join([f'{int(i)}' for i in result.keys()]))
