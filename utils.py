
import numpy  as np
from scipy.interpolate import interp1d

def convert_to_array(strain):
    """
    :param strain: can be a vector or an matrix
    :return:
    """
    data_structure_shape = strain.shape
    if len(data_structure_shape) == 1:
        return strain  # strain is already an array
    if len(data_structure_shape) == 2:
        if data_structure_shape[0] > 1 and data_structure_shape[1] > 1:  # if matrix
            return None  # this is the return value for a matrix, cannot be converted to an array
        if data_structure_shape[0] == 1 and data_structure_shape[1] > 1:  # row vector, one row multiple columns
            return strain[0, :]  # reduce the dimensions by 1 and return the first row
        if data_structure_shape[1] == 1 and data_structure_shape[0] > 1:  # column vector, multiple rows and one column
            return strain[:, 0]
    raise ValueError("Unexpected input strain shape")

    
def MGWTinner_product(aa, bb, freq, PSD, s2):
    '''
    Calculate the inner product defined in the matched filter statistic

    arguments:
    aai, bb: single-sided Fourier transform, created, e.g., by the nfft function above
    freq: an array of frequencies associated with aa, bb, also returned by nfft
    PSD: an Nx2 array describing the noise power spectral density
    s2 : is the variance of extra noise, needs to have the same length as freq

    Returns:
    The matched filter inner product for aa and bb
    '''
    # interpolate the PSD to the freq grid
    PSD_interp_func = interp1d(PSD[:, 0], PSD[:, 1], bounds_error=False, fill_value=np.inf)
    PSD_interp = PSD_interp_func(freq) + s2

    # caluclate the inner product
    integrand = np.conj(aa) * bb / PSD_interp
    #print('integrand',integrand)
    df = freq[1] - freq[0]
    integral = np.sum(integrand) * df

    product = 4. * np.real(integral)
    #input('press enter')
    return product

def inner_product(strain1, strain2, frequency, power_spectral_density, s2):
    """
    inner_product = <strain1,strain2>
    strain1 can be a single array, a row or column vector, or a matrix
    strain2 can be a single array, a row or column vector, or a matrix
    If strain1 and strain2 are both be matrices then they must have the exact same size
    If strain1 is effectively single dimensioned then output is:
    <strain1,strain2>=
        <strain1,strain2[0]>
        <strain1,strain2[1]>
           .
           .
           .
        <strain1,strain2[m]>
    The size of strain1 and strain2[i] must be the same (must be arrays)
    Vice versa for strain2 being single dimensioned
    Otherwise
    <strain1,strain2>=
        <strain1[0],strain2[0]>
        <strain1[1],strain2[1]>
           .
           .
           .
        <strain1[m],strain2[m]>
    """
    # check dimensions of strain1 and strain2
    strain1shape = strain1.shape
    strain2shape = strain2.shape
    strain1array = convert_to_array(strain1)
    strain2array = convert_to_array(strain2)
    # deal with one or both as matrices first
    if strain1array is None:  # is strain1 a matrix
        if strain2array is None:  # is strain2 a matrix as well
            if (strain1shape[0] == strain2shape[0]) and (strain1shape[1] == strain2shape[1]):  # are they the same shape
                inner_product_result = np.zeros(strain1shape[0])
                # prepare to return an array of inner products len=rows(strain1)=rows(strain2)
                for idx, strain1vector in enumerate(strain1):
                    inner_product_result[idx] = MGWTinner_product(convert_to_array(strain1[idx]),
                                                                        convert_to_array(strain2[idx]), frequency,
                                                                        power_spectral_density,
                                                                         s2)
                # print(1)
                return inner_product_result
            else:
                raise ValueError('Wrong shape')
        else:  # strain2array != None, so strain2 is not a matrix and strain2array is an array and strain1 is a matrix
            if len(strain2array) != strain1shape[1]:
                raise ValueError('len(strain2)!= strain1.shape[1]')
            inner_product_result = np.zeros(
                strain1shape[0])  # prepare to return an array of inner products len=rows(strain1)
            for idx, strain1vector in enumerate(strain1):
                inner_product_result[idx] = MGWTinner_product(convert_to_array(strain1vector), strain2array,
                                                                    frequency, power_spectral_density,
                                                                         s2)
            # print(2)
            return inner_product_result
    else:  # strain1 is a vector, strain1array an array
        if strain2array is None:  # strain2 is a matrix and strain1 is a vector
            if len(strain1array) != strain2shape[1]:
                raise ValueError('len(strain1)!=strain2.shape[1]')
            inner_product_result = np.zeros(
                strain2shape[0])  # prepare to return an array of inner products len=rows(strain2)
            for idx, strain2vector in enumerate(strain2):
                #                 print(idx,' ',strain2shape)
                inner_product_result[idx] = MGWTinner_product(strain1array, convert_to_array(strain2vector),
                                                                    frequency, power_spectral_density,
                                                                         s2)
            # print(3)
            return inner_product_result
        else:  # strain1 is a vector and strain2 is a vector
            inner_product_result = MGWTinner_product(strain1array, strain2array, frequency,
                                                           power_spectral_density,
                                                                         s2)
            #             print(4)
            return inner_product_result


def strain_fitting_function_old(strain1, strain2, frequency, power_spectral_density):
    """
    strain_fitting_function = <strain1,strain2>/sqrt(<strain1,strain1><strain2,strain2>)
    """
    return (inner_product(strain1, strain2, frequency, power_spectral_density) /
            np.sqrt(inner_product(strain1, strain1, frequency, power_spectral_density) *
                    inner_product(strain2, strain2, frequency, power_spectral_density)))




def strain_fitting_function_corrected(strain1, strain2, frequency, power_spectral_density,s2):
    """
    strain_fitting_function = <strain1,strain2>/sqrt(<strain1,strain1><strain2,strain2>)
    the input is the CHARACTERISTIC STRAIN = strain*sqrt(freq)
    """
    return (inner_product(strain1/np.sqrt(frequency), strain2/np.sqrt(frequency), 
                          frequency, power_spectral_density, s2) /
            np.sqrt(inner_product(strain1/np.sqrt(frequency), strain1/np.sqrt(frequency), 
                                  frequency, power_spectral_density, s2) *
                    inner_product(strain2/np.sqrt(frequency), strain2/np.sqrt(frequency), 
                                  frequency, power_spectral_density, s2)))



def bns_inner_product(bns_parameter_values1, bns_parameter_values2):
    """
    Implements the cartesian inner products of two vectors bns_parameter_values1 and bns_parameter_values2
    """
    return np.sum(bns_parameter_values1 * bns_parameter_values2)


def bns_parameter_values_fitting_function(bns_parameter_values1, bns_parameter_values2):
    """
    bns_parameter_values_fitting_function = <bns_parameter_values1,bns_parameter_values2>/sqrt(<bns_parameter_values1,
    bns_parameter_values1><bns_parameter_values2,bns_parameter_values2>)
    """
    return (bns_inner_product(bns_parameter_values1, bns_parameter_values2) /
            np.sqrt(bns_inner_product(bns_parameter_values1, bns_parameter_values1) *
                    bns_inner_product(bns_parameter_values2, bns_parameter_values2)))


