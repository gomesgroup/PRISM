import numpy as np
import matplotlib.pyplot as plt

class PerPlateData(object):
  '''
    A general class for storing and plotting the per-plate data (one data point
    for each image or one data point across all images) created by transforming 
    the intensity data.
  '''

  def __init__(self, normalized_intensity_data, data, data_display_name, 
               data_units, sample_or_ref, against_independent_variable):
    '''
      Store the normalized intensity data object and the per-plate data.

      Inputs:
        normalized_intensity_data (NormalizedIntensityData): the normalized
          intensity data object, which will be referenced to determine the
          number of images.

        data (np.float64 np.ndarray or np.float64): per-plate data, 
          possibly over many independent variable/time points.

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

      Output:
        PerPlateData
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

  def display(self, figsize = (15, 15)):
    '''
      Display the data. For per-image data over independent variable/time
      points, plot a line graph. For data that is a single value, print the
      value.

      Inputs:
        self (PerPlateData): a per-plate data object with information about the 
          intensity data stored in self.normalized_intensity_data, the data to 
          plot stored in self.data, and the display name/units for the data 
          stored in self.data_display_name and self.data_units.

        figsize (int, int): width and height of output graphs in inches.
      
      Errors:
        1) If the data is not one or zero dimensional, raise an exception.

        2) If the data is one dimensional, it should have the same number of
          data points as there are images.

      Plots:
        1) A line plot if there is one value per-image, and a printed value
          if there is only a single value.

      Output:
        plt.Figure or None: returns the figure object of the plot, or None if
        the data is a single value.
    '''
    # Unpack some values for readability.
    input_directory_name = self.normalized_intensity_data.input_directory_name
    analysis_channel = self.normalized_intensity_data.analysis_channel
    
    independent_variables = \
      self.normalized_intensity_data.independent_variables
    independent_variable_name = \
      self.normalized_intensity_data.independent_variable_name
    independent_variable_units = \
      self.normalized_intensity_data.independent_variable_units
    
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

    # Line plot.
    if self.against_independent_variable and len(self.data.shape) == 1:
      # Check that the number of data points is equivalent to the number of 
      # images.
      if self.data.shape[0] != independent_variables.shape[0]:
        raise Exception("If an array of per-plate data points is provided, the length should be equivalent to the number of images.")

      fig, ax = plt.subplots(figsize = figsize)
    
      fig.suptitle(title_prefix + data_name_with_units + " vs. " + \
                   independent_variable_display_name_with_units, y = 1)
    
      # Used to pad the axis bounds of the plot.
      data_max = np.nanmax(self.data)
      data_min = np.nanmin(self.data)

      data_range = data_max - data_min

      padding = data_range * 0.05
      
      if data_min != data_max:
        ax.set_ylim(ymin = data_min - padding, ymax = data_max + padding)

      ax.plot(independent_variables, self.data)
    
      fig.tight_layout()
    # Print result.
    elif len(self.data.shape) == 0:
      fig = None
      print(title_prefix + data_name_with_units + ": " + str(self.data))
    else:
      raise Exception("Shape of data not conducive to displaying.")
    
    return fig

class HydrogenMoleFracIndepVar(PerPlateData):
  '''
    A class used to store the per-plate hydrogen mole fraction, serving as the
    independent variable for calibration.
    #NOTE: this class is not designed to be used for analysis (only for 
    calibration).
  '''

  def __init__(self, normalized_intensity_data, injected_hydrogen_volumes, 
               initial_air_volume, volume_units, sample_or_ref):
    '''
      Given the volumes of hydrogen injected, and the initial air volume in
      each vial, compute the corresponding hydrogen mole fractions.
      NOTE: This class/function is designed to be used for calibration, not
      for analysis.

      Inputs:
        normalized_intensity_data (NormalizedIntensityData): the normalized
          intensity data object, which will be referenced to determine the
          number of images.
        
        injected_hydrogen_volumes (np.float64 np.ndarray): the volumes of
          hydrogen injected into the vials.

        initial_air_volume (np.float64): the initial volume of air in each of
          the vials.
        
        volume_units (str): a string that represents the volume units (must be 
          one of "L", "mL", "uL").

        sample_or_ref (str): a string (either "sample" or "ref") that specifies
          whether the data was calculated from sample or reference intensities
          respectively.
      
      Errors:
        1) If the volume_units are not one of "L", "mL", "uL", raise an
          exception.
      
      Output:
        HydrogenMoleFracIndepVar
    '''
    # Check that the volume units are valid.
    if volume_units not in ["L", "mL", "uL"]:
      raise Exception("Volume units (" + volume_units + \
                      ") not one of 'L', 'mL', or 'uL'.")

    # Since we know the volume (at 1 atm) of hydrogen that was injected, and the
    # initial volume of air (also at 1 atm), we can calculate the mole fraction 
    # of hydrogen by considering the system at a pressure of 1 atm, absent of
    # the vial. Since we make ideal gas assumptions, the mole fraction is equal
    # to the volume fraction.
    hydrogen_mole_fractions = injected_hydrogen_volumes / \
      (injected_hydrogen_volumes + initial_air_volume)
    
    super().__init__(normalized_intensity_data, hydrogen_mole_fractions, 
                     "Hydrogen Mole Fraction", None, sample_or_ref, False)

class MeanOfIntensitiesData(PerPlateData):
  '''
    A class used to compute and store the mean of the intensities of a subset of 
    wells.
  '''

  def __init__(self, normalized_intensity_data, norm_intensities, 
               calibration_wells, sample_or_ref):
    '''
      Given the normalized intensities, compute the mean of the intensities of 
      the wells specified by the list of (well row, well column) 0-indexed pairs 
      in the provided well coordinates for every image.

      Inputs:
        normalized_intensity_data (NormalizedIntensityData): the normalized
          intensity data object, which will be referenced to determine the
          number of images.

        norm_intensities (np.float64 np.ndarray): an array containing the
          per-well average intensity data across the independent variable/
          time series.

        calibration_wells ((int, int) List): a list of 0-indexed (well row, well 
          column), specifying the wells to be used for the computation.

        sample_or_ref (str): a string (either "sample" or "ref") that specifies
          whether the data was calculated from sample or reference intensities
          respectively.

      Output:
        MeanOfIntensitiesData
    '''
    # Note, the "*" is used to unpack the list of tuples such that each tuple is 
    # an argument.
    well_rows, well_cols = tuple(zip(*calibration_wells))

    subset_norm_intensities = norm_intensities[:, well_rows, well_cols]

    means_of_intensities = np.mean(subset_norm_intensities, axis = 1)

    super().__init__(normalized_intensity_data, means_of_intensities, 
                     "Mean of Intensities", None, sample_or_ref, True)

class SampleStandardDeviationOfIntensitiesData(PerPlateData):
  '''
    A class used to compute and store the sample standard deviation of the 
    intensities of a subset of wells.
  '''

  def __init__(self, normalized_intensity_data, norm_intensities, 
               calibration_wells, sample_or_ref):
    '''
      Given the normalized intensities, compute the sample standard deviation of 
      the intensities of the wells specified by the list of (well row, 
      well column) 0-indexed pairs in the provided well coordinates for every 
      image.

      Inputs:
        normalized_intensity_data (NormalizedIntensityData): the normalized
          intensity data object, which will be referenced to determine the
          number of images.

        norm_intensities (np.float64 np.ndarray): an array containing the
          per-well average intensity data across the independent variable/
          time series.

        calibration_wells ((int, int) List): a list of 0-indexed (well row, well 
          column), specifying the wells to be used for the computation.

        sample_or_ref (str): a string (either "sample" or "ref") that specifies
          whether the data was calculated from sample or reference intensities
          respectively.

      Output:
        SampleStandardDeviationOfIntensitiesData
    '''
    # Note, the "*" is used to unpack the list of tuples such that each tuple is 
    # an argument.
    well_rows, well_cols = tuple(zip(*calibration_wells))

    subset_norm_intensities = norm_intensities[:, well_rows, well_cols]

    # Use ddof = 1 to compute sample standard deviation for each image.
    # https://numpy.org/doc/stable/reference/generated/numpy.std.html
    sample_standard_deviations_of_intensities = np.std(subset_norm_intensities, 
                                                       ddof = 1, axis = 1)

    super().__init__(normalized_intensity_data, 
                     sample_standard_deviations_of_intensities, 
                     "Sample Standard Deviation of Intensities", None, 
                     sample_or_ref, True)

class CalibrationResults(PerPlateData):
  '''
    A class used to compute and store the calibration results.
  '''
  
  @staticmethod
  def from_mean_intensity_data(normalized_intensity_data, means_of_intensities, 
                               degree, sample_or_ref):
    '''
      Given the means of the intensity data for each image, compute a 
      polynomial fit of the specified degree, returning the coefficients and the
      r^2 value.

      NOTE: This calibration produces a calibration curve that has mole fraction
      of hydrogen as the "x" value (independent variable), and the normalized
      intensity as the "y" value (dependent variable). The fit must be inverted
      to make normalized intensity to hydrogen mole fraction predictions.

      NOTE: the following documentation was referenced for this function:
      https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyfit.html
      https://en.wikipedia.org/wiki/Coefficient_of_determination

      Inputs:
        normalized_intensity_data (NormalizedIntensityData): the normalized
          intensity data object, which will be referenced to determine the
          number of images.

        means_of_intensities (np.float64 np.ndarray): an array of the mean
          intensity of a subset of wells, with one mean per image in the
          dataset.

        degree (int): the degree of the polynomial to fit to the data. The
          degree should be either 1 (for linear fitting) or 2 (for quadratic 
          fitting). 

        sample_or_ref (str): a string (either "sample" or "ref") that specifies
          whether the data was calculated from sample or reference intensities
          respectively.

      Errors:
        1) The length of means_of_intensities should be the same as the number 
          of images. If not, raise an exception.

        2) The degree should be either 1 or 2. If not, raise an exception.

      Output:
        (str, CalibrationResults dict): a dictionary that contains per-plate 
          data objects of the desired calibration quantities. For a linear fit 
          (degree of 1), the fit is ax + b. For a quadratic fit (degree of 2), 
          the fit is ax^2 + bx + c. This dictionary will contain either a and b 
          (for a linear fit) or a, b, and c (for a quadratic fit). Additionally, 
          the r squared will be calculated and saved.
    '''
    independent_variables = normalized_intensity_data.independent_variables

    # Check that the length of means_of_intensities is the same as the number
    # of images.
    if means_of_intensities.shape[0] != independent_variables.shape[0]:
      raise Exception("The length of the means of the intensities should be equivalent to the number of images.")
    
    calibration_output = dict()
    
    # The degree must be 1 or 2.
    if degree in [1, 2]:
      # NOTE: np.polynomial.polynomial.polyfit returns coefficients ordered
      # from low degree terms to high.
      coefficients = np.polynomial.polynomial.polyfit(independent_variables, 
                                                      means_of_intensities, 
                                                      degree)
      # Linear fit.
      if degree == 1:
        b, a = coefficients
        calibration_output["a"] = CalibrationResults(normalized_intensity_data, 
                                                     a, "a", None, 
                                                     sample_or_ref, False)
        calibration_output["b"] = CalibrationResults(normalized_intensity_data, 
                                                     b, "b", None, 
                                                     sample_or_ref, False)
      # Quadratic fit.
      elif degree == 2:
        c, b, a = coefficients
        calibration_output["a"] = CalibrationResults(normalized_intensity_data, 
                                                     a, "a", None, 
                                                     sample_or_ref, False)
        calibration_output["b"] = CalibrationResults(normalized_intensity_data, 
                                                     b, "b", None, 
                                                     sample_or_ref, False)
        calibration_output["c"] = CalibrationResults(normalized_intensity_data, 
                                                     c, "c", None, 
                                                     sample_or_ref, False)
    else:
      raise Exception("The degree should be either 1 or 2.")
    
    # Calculate the r squared.
    # Recall, the fit has hydrogen mole fraction as the independent variable.
    # See the following Wikipedia page for details:
    # https://en.wikipedia.org/wiki/Coefficient_of_determination
    predicted_means_of_intensities = \
      np.polynomial.polynomial.polyval(independent_variables, coefficients)
    sum_of_squares_residuals = \
      np.sum((predicted_means_of_intensities - means_of_intensities) ** 2)
    sum_of_squares_total = \
      np.sum((means_of_intensities - np.mean(means_of_intensities)) ** 2)

    r_squared = 1 - sum_of_squares_residuals / sum_of_squares_total
    calibration_output["r_squared"] = \
      CalibrationResults(normalized_intensity_data, r_squared, "R Squared", 
                         None, sample_or_ref, False)

    return calibration_output