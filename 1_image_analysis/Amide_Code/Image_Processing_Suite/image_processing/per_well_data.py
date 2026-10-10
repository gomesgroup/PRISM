import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class PerWellData(object):
  '''
    A general class for storing and plotting the per-well data created by 
    transforming the normalized intensity data.
  '''

  def __init__(self, normalized_intensity_data, data, data_display_name, 
               data_units, sample_or_ref, against_independent_variable, 
               valid_data = None):
    '''
      Store the normalized intensity data object and the per-well data created 
      by transforming the intensities (or a derivative of the intensities).

      Inputs:
        normalized_intensity_data (NormalizedIntensityData): the normalized
          intensity data object, which will be referenced to determine the shape
          of the sample/reference wells.

        data (np.float64 np.ndarray): per-well data, possibly over many 
          independent variable/time points.

        data_display_name (str): a string that identifies the content of the 
          data.

        data_units (str): a string that specifies the units of the data. 
          None or "" can be used to specify no units.

        sample_or_ref (str | None): a string (either "sample" or "ref") that 
          specifies whether the data was calculated from sample or reference 
          intensities respectively. Can also be None.

        against_independent_variable (bool): if True, this indicates that the 
          data has data points for each independent variable point. Note, the 
          shape of the data will still be checked.

        valid_data (np.bool_ np.ndarray): either None or a boolean array of the
          same shape as data. If None, all data is assumed to be valid. If it is 
          an array, entries denoted with "True" are considered valid, entries 
          denoted with "False" are considered invalid. This will not affect the 
          processing or plotting of the data unless the user implements that 
          functionality in the subclass of data transform. One use case for this 
          is for the quadratic hydrogen fit, specifically for calculating the 
          rate (since all normalized intensities below a specific value are 
          considered invalid).

      Output:
        PerWellData
    '''
    self.normalized_intensity_data = normalized_intensity_data
    self.data = data
    self.data_display_name = data_display_name

    # Standardize the "no-units" option to be None.
    if data_units == "":
      data_units = None
    
    self.data_units = data_units

    # Check that the sample_or_ref string is valid.
    if sample_or_ref not in ["sample", "ref", None]:
      raise Exception("The intensity data must come from 'sample' or 'ref' wells.")
    self.sample_or_ref = sample_or_ref

    self.against_independent_variable = against_independent_variable

    # Check that the valid data array is the proper shape.
    if valid_data is not None:
      if valid_data.shape != self.data.shape:
        raise Exception("Valid data boolean array does not match shape of data.")
    
    self.valid_data = valid_data

  def display(self, figsize = (15, 15)):
    '''
      Display the data. For per-well data over independent variable/time
      points, plot a grid of line graphs. For per-well data that is a single
      value per well, plot an array plot.

      Inputs:
        self (PerWellData): a per-well data object with information about the 
          intensity data stored in self.normalized_intensity_data, the data to 
          plot stored in self.data, and the display name/units for the data 
          stored in self.data_display_name and self.data_units.
        
        figsize (int, int): width and height of output graphs in inches.
      
      Errors:
        1) If the last two dimensions of the data do not match the last two 
          dimensions of the sample or reference intensities, raise an exception.

        2) If the data is not two or three dimensional, raise an exception.

      Plots:
        1) A grid of line graphs for per well data over many independent 
          variable/time points or an array plot if there is only a single value
          per well.

      Output:
        plt.Figure: returns the figure object of the plot.
    '''
    # Unpack some values for readability.
    input_directory_name = self.normalized_intensity_data.input_directory_name
    analysis_channel = self.normalized_intensity_data.analysis_channel

    sample_columns_per_row = \
      self.normalized_intensity_data.sample_columns_per_row
    reference_columns_per_row = \
      self.normalized_intensity_data.reference_columns_per_row
    
    independent_variables = \
      self.normalized_intensity_data.independent_variables
    independent_variable_name = \
      self.normalized_intensity_data.independent_variable_name
    independent_variable_units = \
      self.normalized_intensity_data.independent_variable_units

    # Determine the appropriate columns_per_row to use.
    if self.sample_or_ref == "sample":
      # Check the dimensions of the data.
      num_sample_rows = len(sample_columns_per_row)
      max_sample_cols = max(sample_columns_per_row)

      if (self.data.shape[-2:] != (num_sample_rows, max_sample_cols)):
        raise Exception("The dimensions of the data do not match the dimensions of the sample grid.")
      
      columns_per_row = sample_columns_per_row
    elif self.sample_or_ref == "ref":
      # Check the dimensions of the data.
      num_reference_rows = len(reference_columns_per_row)
      max_reference_cols = max(reference_columns_per_row)

      if (self.data.shape[-2:] != (num_reference_rows, max_reference_cols)):
        raise Exception("The dimensions of the data do not match the dimensions of the reference grid.")

      columns_per_row = reference_columns_per_row
    
    # Add the units to the data display name for the figure.
    if self.data_units == None:
      data_name_with_units = self.data_display_name
    else:
      data_name_with_units = self.data_display_name + " (" + \
                             self.data_units + ")"
      
    # Add the units to the independent variable display name.
    if independent_variable_units == None:
      independent_variable_display_name_with_units = independent_variable_name
    else:
      independent_variable_display_name_with_units = \
        independent_variable_name + " (" + independent_variable_units + ")"
    
    # Compute the title prefix for the figures.
    title_prefix = "(" + input_directory_name + ", " + analysis_channel + ") "

    # Grid of line plots.
    if self.against_independent_variable and len(self.data.shape) == 3:
      num_rows = len(columns_per_row)
      max_cols = max(columns_per_row)

      fig = plt.figure(figsize = figsize)
    
      fig.suptitle(title_prefix + data_name_with_units + " vs. " + \
                   independent_variable_display_name_with_units, y = 1)
    
      # Used to pad the axis bounds of the plots.
      data_max = np.nanmax(self.data)
      data_min = np.nanmin(self.data)

      data_range = data_max - data_min

      padding = data_range * 0.05

      for row, num_columns in enumerate(columns_per_row):
        for col in range(num_columns):
          # Matplotlib is 1-indexed for this purpose.
          sub_plot_index = 1 + max_cols * row + col

          # Create the subplot axes.
          ax = fig.add_subplot(num_rows, max_cols, sub_plot_index)
          
          if data_min == data_max == 0:
            ax.set_ylim(ymin = 0.01, ymax = data_max + 0.5)
          elif data_min != data_max:
            ax.set_ylim(ymin = data_min - padding, ymax = data_max + padding)
          
          ax.plot(independent_variables, self.data[:, row, col])
    
      fig.tight_layout()
    # Array plot.
    elif len(self.data.shape) == 2:
      fig, ax = plt.subplots(figsize = figsize)

      fig.suptitle(title_prefix + data_name_with_units)

      im = ax.matshow(self.data, cmap = "gray")

      # Add a color bar to the image.
      fig.colorbar(im, ax = ax)
    else:
      raise Exception("Shape of data not conducive to displaying.")
    
    return fig

class HydrogenMoleFractionData(PerWellData):
  '''
    A class used to store per-well hydrogen mole fraction data.
  '''

  def __init__(self, normalized_intensity_data, norm_intensities,
               reactor_calibration_coefficients, reactor_name, sample_or_ref):
    '''
      Given a reactor name, use that reactor's calibration constants (determined
      with calibration.ipynb or other methods) to convert normalized
      intensity data to hydrogen mole fraction data.

      Inputs:
        normalized_intensity_data (NormalizedIntensityData): the normalized
          intensity data object, which will be referenced to determine the shape
          of the sample/reference wells.

        norm_intensities (np.float64 np.ndarray): an array containing the
          per-well average intensity data across the independent variable/
          time series.
        
        reactor_calibration_coefficients ((str, (str, int) dict) dict): a
          dictionary that maps reactor names to dictionaries containing the
          calibration coefficients. If the calibration coefficient dictionary
          contains two entries ("a" and "b"), a linear equation (ax + b) will
          be used. If the dictionary contains three entries ("a", "b", and "c"),
          a quadratic equation (ax^2 + bx + c) will be used.
        
        reactor_name (str): a string denoting the reactor name, one of the keys
          in reactor_calibration_coefficients.

        sample_or_ref (str): a string (either "sample" or "ref") that specifies
          whether the data was calculated from sample or reference intensities
          respectively.
      
      Errors:
        1) If the reactor_name is not one of the keys in the 
          reactor_calibration_coefficients dictionary, raise an exception.
        
        2) If the calibration coefficient dictionary has more than three entries
          or less than two entries, raise an exception (only quadratic and
          linear calibrations are supported).

      Output:
        HydrogenMoleFractionData
    '''
    if reactor_name in reactor_calibration_coefficients:
      calibration_coefficients = reactor_calibration_coefficients[reactor_name]
    else:
      raise Exception("Reactor name ( " + reactor_name + \
                      " ) does not have an intensity to hydrogen mole fraction calibration.")
    
    # Quadratic calibration curve.
    if len(calibration_coefficients) == 3:
      a = np.float64(calibration_coefficients["a"])
      b = np.float64(calibration_coefficients["b"])
      c = np.float64(calibration_coefficients["c"])

      # Since the normalized intensity vs. mole fraction calibration curve
      # is a quadratic fit, there is a minimum valid normalized intensity value,
      # corresponding to the minimum of the quadratic curve.
      # NOTE: a small value is added to ensure that the data is not exactly
      # at the minimum of the quadratic curve (without this, due to the way
      # floating point arithmetic works, it may crash).
      epsilon = np.float64(1e-10)
      minimum_valid_norm_intensity = c - ((b ** 2) / (4 * a)) + epsilon

      # Create an array where the valid intensities (those originally greater 
      # than minimum_valid_norm_intensity) correspond to a True value.
      valid_data = norm_intensities > minimum_valid_norm_intensity

      # Set any intensity values below minimum_valid_norm_intensity to
      # minimum_valid_norm_intensity.
      truncated_data = np.maximum(norm_intensities, 
                                  minimum_valid_norm_intensity)
      
      hydrogen_mole_fractions = \
        (-b - np.sqrt((b ** 2) - (4 * a * (c - truncated_data)))) / (2 * a)
    # Linear calibration curve.
    elif len(calibration_coefficients) == 2:
      a = np.float64(calibration_coefficients["a"])
      b = np.float64(calibration_coefficients["b"])

      # The issue with the quadratic curve is not an issue here.
      valid_data = None

      hydrogen_mole_fractions = (norm_intensities - b) / a
    else:
      raise Exception("Only quadratic and linear calibration curves are supported.")
    
    super().__init__(normalized_intensity_data, hydrogen_mole_fractions, 
                     "Hydrogen Mole Fraction", None, sample_or_ref, True,
                     valid_data = valid_data)

class CorrectedHydrogenVolumeData(PerWellData):
  '''
    A class used to store per-well corrected hydrogen volume data. "Corrected" 
    refers to the fact that the the partial volume of hydrogen is divided by the
    mole fraction of air (1 - mole fraction of hydrogen). This accounts for
    the increase in pressure, allowing the corrected volume-to-amount
    conversion to assume that the pressure is 1 atm.
  '''

  def __init__(self, normalized_intensity_data, hydrogen_mole_fractions, 
               headspace_volumes, volume_units, sample_or_ref, 
               valid_data = None):
    '''
      Given the hydrogen mole fraction data and the headspace volumes for
      each vial (along with the volume units), compute the corrected volume of 
      hydrogen (corrected means that the partial volume of hydrogen is divided
      by the mole fraction of air; as mentioned above, this allows the corrected 
      volume-to-amount conversion to assume a pressure of 1 atm).

      Inputs:
        normalized_intensity_data (NormalizedIntensityData): the normalized
          intensity data object, which will be referenced to determine the shape
          of the sample/reference wells.

        hydrogen_mole_fractions (np.float64 np.ndarray): an array containing the 
          per-well hydrogen mole fraction across the independent variable/time 
          series.
        
        headspace_volumes (np.float64 np.ndarray): an array containing the
          per-well headspace volumes.
        
        volume_units (str): a string that represents the volume units for
          hydrogen (must be one of "L", "mL", "uL").

        sample_or_ref (str): a string (either "sample" or "ref") that specifies
          whether the data was calculated from sample or reference intensities
          respectively.
        
        valid_data (np.bool_ np.ndarray): either None or a boolean array of the
          same shape as data. If None, all data is assumed to be valid. If it is 
          an array, entries denoted with "True" are considered valid, entries 
          denoted with "False" are considered invalid. This will not affect the 
          processing or plotting of the data unless the user implements that 
          functionality in the subclass of data transform. One use case for this 
          is for the quadratic hydrogen fit, specifically for calculating the 
          rate (since all normalized intensities below a specific value are 
          considered invalid).
      
      Errors:
        1) If the volume_units are not one of "L", "mL", "uL", raise an
          exception.
        
        2) If the headspace_volumes array is not the same shape as the
          hydrogen_mole_fraction data from a single image, raise an
          exception.
      
      Output:
        CorrectedHydrogenVolumeData
    '''
    # Check that the volume units are valid.
    if volume_units not in ["L", "mL", "uL"]:
      raise Exception("Volume units (" + volume_units + \
                      ") not one of 'L', 'mL', or 'uL'.")

    # Check that the headspace volumes array is the proper shape.
    if headspace_volumes.shape != hydrogen_mole_fractions[0].shape:
      raise Exception("Headspace volume array does not match shape of hydrogen mole fraction data from a single image.")

    # NOTE: Division by (1 - hydrogen_mole_fractions) is the correction 
    # discussed above.
    corrected_hydrogen_volumes = (hydrogen_mole_fractions / \
      (1 - hydrogen_mole_fractions)) * headspace_volumes
    
    super().__init__(normalized_intensity_data, corrected_hydrogen_volumes, 
                     "Corrected Hydrogen Volume", volume_units, sample_or_ref, 
                     True, valid_data = valid_data)

class HydrogenAmountData(PerWellData):
  '''
    A class used to store per-well hydrogen amount data.
  '''

  def __init__(self, normalized_intensity_data, corrected_hydrogen_volumes, 
               volume_units, amount_units, sample_or_ref, valid_data = None):
    '''
      Given the corrected hydrogen volume (see CorrectedHydrogenVolumeData class 
      for definition of "corrected" in this context), the volume units, and the 
      amount units, compute the amount of hydrogen using the ideal gas law.

      Inputs:
        normalized_intensity_data (NormalizedIntensityData): the normalized
          intensity data object, which will be referenced to determine the shape
          of the sample/reference wells.
        
        corrected_hydrogen_volumes (np.float64 np.ndarray): an array containing 
          the per-well corrected hydrogen volume across the independent 
          variable/time series (see CorrectedHydrogenVolumeData class for 
          definition of "corrected" in this context)

        volume_units (str): a string that represents the volume units for
          hydrogen (must be one of "L", "mL", "uL").

        amount_units (str): a string that represents the amount units for
          hydrogen (must be one of "mol", "mmol", or "umol").

        sample_or_ref (str): a string (either "sample" or "ref") that specifies
          whether the data was calculated from sample or reference intensities
          respectively.

        valid_data (np.bool_ np.ndarray): either None or a boolean array of the
          same shape as data. If None, all data is assumed to be valid. If it is 
          an array, entries denoted with "True" are considered valid, entries 
          denoted with "False" are considered invalid. This will not affect the 
          processing or plotting of the data unless the user implements that 
          functionality in the subclass of data transform. One use case for this 
          is for the quadratic hydrogen fit, specifically for calculating the 
          rate (since all normalized intensities below a specific value are 
          considered invalid).

      Errors:
        1) If the volume_units are not one of "L", "mL", "uL", raise an
          exception.

        2) If the amount_units are not one of "mol", "mmol", or "umol", raise
          an exception.
      
      Output:
        HydrogenAmountData
    '''
    # Check that the volume units are valid.
    if volume_units not in ["L", "mL", "uL"]:
      raise Exception("Volume units (" + volume_units + \
                      ") not one of 'L', 'mL', or 'uL'.")
    
    # Check that the amount units are valid.
    if amount_units not in ["mol", "mmol", "umol"]:
      raise Exception("Amount units (" + amount_units + \
                      ") not one of 'mol', 'mmol', or 'umol'")

    # Get the conversion constant based on volume units.
    if volume_units == "L":
      volume_conversion_constant = 1
    elif volume_units == "mL":
      volume_conversion_constant = 1 / 1000
    elif volume_units == "uL":
      volume_conversion_constant = 1 / 1000000

    # Get the conversion constant based on amount units.
    if amount_units == "mol":
      amount_conversion_constant = 1
    elif amount_units == "mmol":
      amount_conversion_constant = 1000
    elif amount_units == "umol":
      amount_conversion_constant = 1000000

    # Use the ideal gas law to compute the hydrogen amount in moles.
    # n = PV / (RT)
    # Assume that T = 298 K.
    # NOTE: We can assume that P = 1 atm due to the fact that we are using
    # corrected hydrogen partial volumes (see CorrectedHydrogenVolumeData for
    # more detail).
    R = 0.08206
    T = 298.0
    hydrogen_amounts = corrected_hydrogen_volumes * \
      ((volume_conversion_constant * amount_conversion_constant) / (R * T))
    
    super().__init__(normalized_intensity_data, hydrogen_amounts, 
                     "Hydrogen Amount", amount_units, sample_or_ref, True, 
                     valid_data = valid_data)

class HydrogenRateData(PerWellData):
  '''
    A class used to store per-well rate of hydrogen evolution data.
  '''
  
  def __init__(self, normalized_intensity_data, hydrogen_amounts, amount_units, 
               time_units, savitzky_golay_window, savitzky_golay_degree, 
               sample_or_ref, valid_data = None):
    '''
      Given the hydrogen amounts, amount units, time units, and an array
      denoting the valid data, fit SciPy Univariate Splines to the hydrogen
      amounts data and extract the rates (invalid data has a rate of 0 by
      default).

      #NOTE: a Savitzky-Golay filter is used to find a smoothed derivative.
      # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
      # https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter

      Inputs:
        normalized_intensity_data (NormalizedIntensityData): the normalized
          intensity data object, which will be referenced to determine the shape
          of the sample/reference wells.

        hydrogen_amounts (np.float64 np.ndarray): an array containing the
          per-well hydrogen amounts across the independent variable/time series.

        amount_units (str): a string that represents the amount units for
          hydrogen (must be one of "mol", "mmol", or "umol").
        
        time_units (str): a string that represents the time units for the
          hydrogen data (must be one of "second", "minute", or "hour").

        savitzky_golay_window (int): the length of the window used for the
          savitzky-golay filter. Must be less than or equal to 
          hydrogen_amounts.shape[0].
        
        savitzky_golay_degree (int): the degree of the polynomial used for
          fitting. Must be less than savitzky_golay_window.

        sample_or_ref (str): a string (either "sample" or "ref") that specifies
          whether the data was calculated from sample or reference intensities
          respectively.
        
        valid_data (np.bool_ np.ndarray): either None or a boolean array of the
          same shape as data. If None, all data is assumed to be valid. If it is 
          an array, entries denoted with "True" are considered valid, entries 
          denoted with "False" are considered invalid. This will not affect the 
          processing or plotting of the data unless the user implements that 
          functionality in the subclass of data transform. One use case for this 
          is for the quadratic hydrogen fit, specifically for calculating the 
          rate (since all normalized intensities below a specific value are 
          considered invalid).

      Errors:
        1) If the amount_units are not one of "mol", "mmol", or "umol", raise
          an exception.
        
        2) If the time_units are not one of "second", "minute", or "hour", raise
          an exception.

        3) If the savitzky_golay_window is greater than the number of images,
          raise an exception.

        4) If the savitzky_golay_degree is greater than or equal to the 
          savitzky_golay_window, raise an exception.
        
        5) If the picture times are not exactly evenly spaced, raise an
          exception.
        
      Output:
        HydrogenRateData
    '''
    # Check that the amount units are valid.
    if amount_units not in ["mol", "mmol", "umol"]:
      raise Exception("Amount units (" + amount_units + \
                      ") not one of 'mol', 'mmol', or 'umol'")
    
    # Check that the time units are valid.
    if time_units not in ["second", "minute", "hour"]:
      raise Exception("Time units (" + time_units + \
                      ") not one of 'second', 'minute', or 'hour'.")
    
    # Check that the savitsky-golay window length is valid.
    if savitzky_golay_window > hydrogen_amounts.shape[0]:
      raise Exception("Savitzky-Golay window must be less than or equal to the number of images.")

    # Check that the savitsky-golay degree is valid.
    if savitzky_golay_degree >= savitzky_golay_window:
      raise Exception("Savitzky-Golay degree must be less than the Savitzky-Golay window.")

    time = normalized_intensity_data.independent_variables

    # Check that all time points are equally spaced (the Savitsky-Golay filter
    # only works in this case).
    time_evenly_spaced = True
    for i in range(len(time) - 2):
      curr_difference = time[i + 1] - time[i]
      next_difference = time[i + 2] - time[i + 1]
      time_evenly_spaced = time_evenly_spaced and np.isclose(curr_difference,
                                                             next_difference)
    if not time_evenly_spaced:
      raise Exception("Savitzky-Golay filter only works if time points are evenly spaced.")

    # Create variables to store the computed fits and rates.
    hydrogen_rates = np.full(hydrogen_amounts.shape, np.nan, dtype = np.float64)
    
    num_rows = hydrogen_amounts.shape[1]
    num_cols = hydrogen_amounts.shape[2]
    # Compute the fits and rates.
    for row in range(num_rows):
      for col in range(num_cols):
        well_amounts = hydrogen_amounts[:, row, col]
        # If a well's amounts contain any np.nan values, do not create a fit,
        # and set all rate values to np.nan.
        if not np.any(np.isnan(well_amounts)):
          # If the all the data is valid, fit all of the data.
          if valid_data is None or np.all(valid_data[:, row, col]):
            hydrogen_rates[:, row, col] = \
              signal.savgol_filter(well_amounts, savitzky_golay_window, 
                                   savitzky_golay_degree, deriv = 1, 
                                   delta = time[1] - time[0])
          # Otherwise, set the rates for the invalid data to 0, and fit the
          # valid data.
          else:
            # Set the invalid data points to have a rate of zero.
            first_invalid_index = \
              np.where(np.logical_not(valid_data[:, row, col]))[0][0]
            hydrogen_rates[first_invalid_index :, row, col] = 0.0

            # Fit the valid data points.
            trimmed_well_amounts = well_amounts[: first_invalid_index]
            trimmed_time = time[: first_invalid_index]

            # Compute the derivative
            hydrogen_rates[: first_invalid_index, row, col] = \
              signal.savgol_filter(trimmed_well_amounts, savitzky_golay_window, 
                                   savitzky_golay_degree, deriv = 1, 
                                   delta = trimmed_time[1] - trimmed_time[0])
    
    super().__init__(normalized_intensity_data, hydrogen_rates, "Hydrogen Rate", 
                     amount_units + "/" + time_units, sample_or_ref, True, 
                     valid_data = valid_data)
  
class MaxHydrogenAmountData(PerWellData):
  '''
    A class used to store the per-well maximum hydrogen amount data.
  '''
  
  def __init__(self, normalized_intensity_data, hydrogen_amounts, amount_units, 
               sample_or_ref, valid_data = None):
    '''
      Given the hydrogen amounts over the independent variable, compute the
      the max amount for each well.

      Inputs:
        normalized_intensity_data (NormalizedIntensityData): the normalized
          intensity data object, which will be referenced to determine the shape
          of the sample/reference wells.

        hydrogen_amounts (np.float64 np.ndarray): an array containing the
          per-well hydrogen amounts across the independent variable/time series.

        amount_units (str): a string that represents the amount units for
          hydrogen (must be one of "mol", "mmol", or "umol").

        sample_or_ref (str): a string (either "sample" or "ref") that specifies
          whether the data was calculated from sample or reference intensities
          respectively.
        
        valid_data (np.bool_ np.ndarray): either None or a boolean array of the
          same shape as data. If None, all data is assumed to be valid. If it is 
          an array, entries denoted with "True" are considered valid, entries 
          denoted with "False" are considered invalid. This will not affect the 
          processing or plotting of the data unless the user implements that 
          functionality in the subclass of data transform. One use case for this 
          is for the quadratic hydrogen fit, specifically for calculating the 
          rate (since all normalized intensities below a specific value are 
          considered invalid).

      Errors:
        1) If the amount_units are not one of "mol", "mmol", or "umol", raise
          an exception.

      Output:
        MaxHydrogenAmountData
    '''
    # Check that the amount units are valid.
    if amount_units not in ["mol", "mmol", "umol"]:
      raise Exception("Amount units (" + amount_units + \
                      ") not one of 'mol', 'mmol', or 'umol'")

    max_hydrogen_amounts = np.amax(hydrogen_amounts, axis = 0)

    super().__init__(normalized_intensity_data, max_hydrogen_amounts, 
                     "Max Hydrogen Amount", amount_units, sample_or_ref, False, 
                     valid_data = valid_data)

class MaxHydrogenRateData(PerWellData):
  '''
    A class used to store the per-well maximum rate of hydrogen evolution data.
  '''

  def __init__(self, normalized_intensity_data, hydrogen_rates, amount_units, 
               time_units, sample_or_ref, valid_data = None):
    '''
      Given the hydrogen rates over the independent variable, compute the
      the max rate for each well.

      Inputs:
        normalized_intensity_data (NormalizedIntensityData): the normalized
          intensity data object, which will be referenced to determine the shape
          of the sample/reference wells.

        hydrogen_rates (np.float64 np.ndarray): an array containing the
          per-well hydrogen rates across the independent variable/time series.

        amount_units (str): a string that represents the amount units for
          hydrogen (must be one of "mol", "mmol", or "umol").

        time_units (str): a string that represents the time units for the
          hydrogen data (must be one of "second", "minute", or "hour").

        sample_or_ref (str): a string (either "sample" or "ref") that specifies
          whether the data was calculated from sample or reference intensities
          respectively.
        
        valid_data (np.bool_ np.ndarray): either None or a boolean array of the
          same shape as data. If None, all data is assumed to be valid. If it is 
          an array, entries denoted with "True" are considered valid, entries 
          denoted with "False" are considered invalid. This will not affect the 
          processing or plotting of the data unless the user implements that 
          functionality in the subclass of data transform. One use case for this 
          is for the quadratic hydrogen fit, specifically for calculating the 
          rate (since all normalized intensities below a specific value are 
          considered invalid).
      
      Errors:
        1) If the amount_units are not one of "mol", "mmol", or "umol", raise
          an exception.
        
        2) If the time_units are not one of "second", "minute", or "hour", raise
          an exception.
      
      Output:
        MaxHydrogenRateData
    '''
    # Check that the amount units are valid.
    if amount_units not in ["mol", "mmol", "umol"]:
      raise Exception("Amount units (" + amount_units + \
                      ") not one of 'mol', 'mmol', or 'umol'")
    
    # Check that the time units are valid.
    if time_units not in ["second", "minute", "hour"]:
      raise Exception("Time units (" + time_units + \
                      ") not one of 'second', 'minute', or 'hour'.")

    max_hydrogen_rates = np.amax(hydrogen_rates, axis = 0)

    super().__init__(normalized_intensity_data, max_hydrogen_rates, 
                     "Max Hydrogen Rate", amount_units + "/" + time_units, 
                     sample_or_ref, False, valid_data = valid_data)

class HydrogenIncubationTimeData(PerWellData):
  '''
    A class used to store the per-well incubation time data for hydrogen 
    evolution. Incubation time refers to the time that it takes for hydrogen to 
    start evolving.
  '''

  def __init__(self, normalized_intensity_data, hydrogen_amounts, 
               max_hydrogen_amounts, incubation_threshold_fraction, time_units, 
               sample_or_ref, valid_data = None):
    '''
      Given the max hydrogen amounts and the fraction of the maximum which
      denotes the end of incubation, return the incubation time for each
      well.

      Inputs:
        normalized_intensity_data (NormalizedIntensityData): the normalized
          intensity data object, which will be referenced to determine the shape
          of the sample/reference wells.
        
        hydrogen_amounts (np.float64 np.ndarray): an array containing the
          per-well hydrogen amounts across the independent variable/time series.

        max_hydrogen_amounts (np.float64 np.ndarray): an array containing the
          per-well maximum hydrogen amount.

        incubation_threshold_fraction (float): a float that denotes the fraction 
          of the maximum hydrogen amount that, once reached, indicates the
          incubation time.

        time_units (str): a string that represents the time units for the
          hydrogen data (must be one of "second", "minute", or "hour").

        sample_or_ref (str): a string (either "sample" or "ref") that specifies
          whether the data was calculated from sample or reference intensities
          respectively.

        valid_data (np.bool_ np.ndarray): either None or a boolean array of the
          same shape as data. If None, all data is assumed to be valid. If it is 
          an array, entries denoted with "True" are considered valid, entries 
          denoted with "False" are considered invalid. This will not affect the 
          processing or plotting of the data unless the user implements that 
          functionality in the subclass of data transform. One use case for this 
          is for the quadratic hydrogen fit, specifically for calculating the 
          rate (since all normalized intensities below a specific value are 
          considered invalid).
      
      Errors:
        1) If the time_units are not one of "second", "minute", or "hour", raise
          an exception.

        2) If the incubation_threshold_fraction is less than 0.0 or greater than
          1.0, raise an exception.

      Output:
        HydrogenIncubationTimeData
    '''

    # Check that the time units are valid.
    if time_units not in ["second", "minute", "hour"]:
      raise Exception("Time units (" + time_units + \
                      ") not one of 'second', 'minute', or 'hour'.")

    # Check the incubation threshold fraction.
    if incubation_threshold_fraction < 0.0 or \
      incubation_threshold_fraction > 1.0:
      raise Exception("Incubation threshold fraction is less than 0.0 or greater than 1.0.")

    incubation_times = np.full(max_hydrogen_amounts.shape, np.nan, 
                               dtype = np.float64)
    
    # Calculate the incubation times.
    num_rows = max_hydrogen_amounts.shape[0]
    num_cols = max_hydrogen_amounts.shape[1]

    time = normalized_intensity_data.independent_variables
    incubation_threshold_fraction = np.float64(incubation_threshold_fraction)

    for row in range(num_rows):
      for col in range(num_cols):
        # If there are any np.nan values in the hydrogen amount data for that
        # well, the incubation time is np.nan.
        if not np.any(np.isnan(hydrogen_amounts[:, row, col])):
          incubation_time_threshold = \
            incubation_threshold_fraction * max_hydrogen_amounts[row, col]

          # Find the first index where the hydrogen amount exceeds the
          # incubation threshold hydrogen amount.
          incubation_time_index = np.where(hydrogen_amounts[:, row, col] > \
                                           incubation_time_threshold)[0][0]

          incubation_times[row, col] = time[incubation_time_index]

    super().__init__(normalized_intensity_data, incubation_times, 
                     "Hydrogen Incubation Time", time_units, sample_or_ref, 
                     False, valid_data = valid_data)

class HydrogenPlateauTimeData(PerWellData):
  '''
    A class used to store the per-well plateau time data for hydrogen evolution. 
    Plateau time refers to the time at which hydrogen evolution stops.
  '''

  def __init__(self, normalized_intensity_data, hydrogen_amounts, 
               max_hydrogen_amounts, plateau_threshold_fraction, time_units, 
               sample_or_ref, valid_data = None):
    '''
      Given the max hydrogen amounts and the fraction of the maximum which
      denotes the beginning of the plateau, return the plateau time for each
      well.

      Inputs:
        normalized_intensity_data (NormalizedIntensityData): the normalized
          intensity data object, which will be referenced to determine the shape
          of the sample/reference wells.
        
        hydrogen_amounts (np.float64 np.ndarray): an array containing the
          per-well hydrogen amounts across the independent variable/time series.

        max_hydrogen_amounts (np.float64 np.ndarray): an array containing the
          per-well maximum hydrogen amount.

        plateau_threshold_fraction (float): a float that denotes the fraction of
          the maximum hydrogen amount that, once reached, indicates the
          plateau time.

        time_units (str): a string that represents the time units for the
          hydrogen data (must be one of "second", "minute", or "hour").

        sample_or_ref (str): a string (either "sample" or "ref") that specifies
          whether the data was calculated from sample or reference intensities
          respectively.

        valid_data (np.bool_ np.ndarray): either None or a boolean array of the
          same shape as data. If None, all data is assumed to be valid. If it is 
          an array, entries denoted with "True" are considered valid, entries 
          denoted with "False" are considered invalid. This will not affect the 
          processing or plotting of the data unless the user implements that 
          functionality in the subclass of data transform. One use case for this 
          is for the quadratic hydrogen fit, specifically for calculating the 
          rate (since all normalized intensities below a specific value are 
          considered invalid).

      Errors:
        1) If the time_units are not one of "second", "minute", or "hour", raise
          an exception.

        2) If the plateau_threshold_fraction is less than 0.0 or greater than
          1.0, raise an exception.

      Output:
        HydrogenPlateauTimeData
    '''

    # Check that the time units are valid.
    if time_units not in ["second", "minute", "hour"]:
      raise Exception("Time units (" + time_units + \
                      ") not one of 'second', 'minute', or 'hour'.")

    # Check the plateau threshold fraction.
    if plateau_threshold_fraction < 0.0 or \
      plateau_threshold_fraction > 1.0:
      raise Exception("Plateau threshold fraction is less than 0.0 or greater than 1.0.")

    plateau_times = np.full(max_hydrogen_amounts.shape, np.nan, 
                            dtype = np.float64)
    
    # Calculate the plateau times.
    num_rows = max_hydrogen_amounts.shape[0]
    num_cols = max_hydrogen_amounts.shape[1]

    time = normalized_intensity_data.independent_variables
    plateau_threshold_fraction = np.float64(plateau_threshold_fraction)

    for row in range(num_rows):
      for col in range(num_cols):
        # If there are any np.nan values in the hydrogen amount data for that
        # well, the plateau time is np.nan.
        if not np.any(np.isnan(hydrogen_amounts[:, row, col])):
          plateau_time_threshold = \
            plateau_threshold_fraction * max_hydrogen_amounts[row, col]
          
          # Find the first index where the hydrogen amount exceeds the plateau
          # threshold hydrogen amount.
          plateau_time_index = np.where(hydrogen_amounts[:, row, col] > \
                                           plateau_time_threshold)[0][0]

          plateau_times[row, col] = time[plateau_time_index]

    super().__init__(normalized_intensity_data, plateau_times, 
                     "Hydrogen Plateau Time", time_units, sample_or_ref, False, 
                     valid_data = valid_data)