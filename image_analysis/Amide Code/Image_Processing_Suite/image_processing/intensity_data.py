import matplotlib.pyplot as plt

from image_processing.per_well_data import *
from image_processing.per_plate_data import *

class NormalizedIntensityData(object):
  '''
    This class is used to store normalized intensity data (normalized by the
    first time point) and the corresponding independent variable (time, 
    injection volume of hydrogen, etc.).
  '''

  def __init__(self, input_directory_name, analysis_channel, 
               reference_columns_per_row, sample_columns_per_row, 
               norm_raw_reference_intensities, norm_reference_intensities, 
               norm_sample_intensities, independent_variable_name, 
               independent_variable_units, independent_variables):
    '''
      Store the input image directory name. Store the columns per row in the 
      grid of reference wells and sample wells. Store the raw reference, 
      reference, and sample intensities, all normalized by the data from the 
      first image. Store the independent variable name, units, and list of 
      variables.

      Inputs:
        input_directory_name (str): the name of the image directory (not the
          full path) from which this intensity data was calculated.

        analysis_channel (str): the analysis channel that was used to compute
          this data.
        
        reference_columns_per_row (int list): the number of columns in each
          row of the reference well grid.

        sample_columns_per_row (int list): the number of columns in each
          row of the sample well grid.

        norm_raw_reference_intensities (np.float64 np.ndarray): the raw
          intensities (not normalized by the reference wells) of the reference
          wells over time, normalized by the data from the first image. Can
          be indexed by (image number, reference well row, 
          reference well column).

        norm_reference_intensities (np.float64 np.ndarray): the intensities 
          (normalized by the reference wells) of the reference wells over time, 
          normalized by the data from the first image. Can be indexed by 
          (image number, reference well row, reference well column).

        norm_sample_intensities (np.float64 np.ndarray): the intensities 
          (normalized by the reference wells) of the sample wells over time, 
          normalized by the data from the first image. Can be indexed by 
          (image number, sample well row, sample well column).

        independent_variable_name (str): a string denoting the name of the
          independent variable.

        independent_variable_units (str): a string denoting the units for
          the independent variable.

        independent_variables (_ list): a list of independent variables (one
          for each image).
      
      Output:
        NormalizedIntensityData
    '''
    self.input_directory_name = input_directory_name
    self.analysis_channel = analysis_channel

    self.reference_columns_per_row = reference_columns_per_row
    self.sample_columns_per_row = sample_columns_per_row

    # Store the intensities as well data.
    self.norm_raw_reference_intensities = \
      PerWellData(self, norm_raw_reference_intensities, 
                  "Normalized Raw Reference Intensity", None, "ref", True)
    self.norm_reference_intensities = \
      PerWellData(self, norm_reference_intensities, 
                  "Normalized Reference Intensity", None, "ref", True)
    self.norm_sample_intensities = \
      PerWellData(self, norm_sample_intensities, "Normalized Sample Intensity", 
                  None, "sample", True)

    self.independent_variable_name = independent_variable_name
    self.independent_variable_units = independent_variable_units
    self.independent_variables = independent_variables

  def display_raw_reference_intensities(self, figsize = (3, 3)):
    '''
      Plot the normalized raw reference intensities vs. the independent 
      variable.
    '''
    self.norm_raw_reference_intensities.display(figsize = figsize)

  def display_reference_intensities(self, figsize = (3, 3)):
    '''
      Plot the normalized reference intensities vs. the independent variable.
    '''
    self.norm_reference_intensities.display(figsize = figsize)

  def display_sample_intensities(self, figsize = (20, 20)):
    '''
      Plot the normalized sample intensities vs. the independent variable.
    '''
    self.norm_sample_intensities.display(figsize = figsize)
  
  def display_calibration_figure(self, independent_variable_data, 
    means_of_intensities, sample_stds_of_intensities, r_squared, a, b, c = None, 
    figsize = (5, 5)):
    '''
      Given the results of the calibration, plot a figure that captures the
      calibration data.

      Inputs:
        independent_variable_data (PerPlateData): an object that contains the
          display name, units, and data for the independent variable.
        
        means_of_intensities (PerPlateData): an object that contains the
          mean intensity of the calibration wells for each image.

        sample_stds_of_intensities (PerPlateData): an object that contains the 
          sample standard deviation of the calibration wells for each image.
        
        r_squared (PerPlateData): the r-squared value for the calibration.

        a (PerPlateData): the coefficient of the highest degree term.

        b (PerPlateData): the coefficient of the second highest degree term.

        c (PerPlateData): the coefficient of the lowest degree term. If None, 
          a linear fit is assumed (ax + b). If not None, a quadratic fit is 
          assumed (ax^2 + bx + c).
        
        figsize (int, int): width and height of output graphs in inches.

      Plots:
        1) A line plot of the means_of_intensities with error bars of one
          standard deviation (determined from sample_stds_of_intensities). 
          Overlaid on this plot is the calibration curve.

      Output:
        plt.Figure: returns the figure object of the plot.
    '''
    fig, ax = plt.subplots(figsize = figsize)

    title_prefix = "(" + self.input_directory_name + ", " + \
      self.analysis_channel + ") "
    
    if independent_variable_data.data_units == None:
      independent_variable_display_name_with_units = \
        independent_variable_data.data_display_name
    else:
      independent_variable_display_name_with_units = \
        independent_variable_data.data_display_name + " (" + \
        independent_variable_data.data_units + ")"

    fig.suptitle(title_prefix + means_of_intensities.data_display_name + \
                 " vs. " + independent_variable_display_name_with_units + \
                  ": actual vs. fit", y = 1)

    ax.errorbar(independent_variable_data.data, means_of_intensities.data, 
                yerr = sample_stds_of_intensities.data)
    
    domain = np.linspace(np.amin(independent_variable_data.data), 
                         np.amax(independent_variable_data.data), num = 1000)
    # Linear fit.
    if c == None:
      predicted_means = a.data * domain + b.data
    # Quadratic fit.
    else:
      predicted_means = a.data * (domain ** 2) + b.data * domain + c.data

    ax.plot(domain, predicted_means)

    return fig

  def post_process(self, user_parameters, figsize = (5, 5)):
    '''
      Given the intensity data and the user parameters from the config file,
      post process the intensity data, returning the independent variable data,
      the per-well data, and the per-plate data.

      Inputs:
        self (NormalizedIntensityData): a valid intensity data object.

        user_parameters ((str, _) dict): a dictionary that maps the variable
          name as a string to the user-defined values from the config file.

        figsize (int, int): width and height of output graphs in inches.

      Errors:
        1) If any of the requested outputs dictionary are invalid, raise an 
          exception.

      Output:
        independent_variable_data (PerPlateData): a per-plate data object that
          stores the details of the independent variable.

        per_well_data (PerWellData List): a list of the computed per-well data
          that is to be displayed/exported. This list is based off the user-
          requested outputs in the config file.

        per_plate_data (PerPlateData List): a list of the computed per-plate 
          data that is to be displayed/exported. This list is based off the 
          user-requested outputs in the config file.
    '''
    image_processing_type = user_parameters.get("image_processing_type")
    is_calibration = user_parameters.get("is_calibration")
    display_figures = user_parameters.get("display_figures")
    # Get the specific requested outputs for the corresponding image dataset.
    specific_requested_outputs = \
      user_parameters.get("requested_outputs")[self.input_directory_name]


    # Independent variable data.
    independent_variable_data = PerPlateData(None, self.independent_variables, 
                                             self.independent_variable_name, 
                                             self.independent_variable_units,
                                             None, False)

    per_well_data = []
    per_plate_data = []
    
    # Output the intensity data if requested.
    if specific_requested_outputs.get("Normalized Raw Reference Intensity", 
                                      False):
      per_well_data.append(self.norm_raw_reference_intensities)
    
    if specific_requested_outputs.get("Normalized Reference Intensity", False):
      per_well_data.append(self.norm_reference_intensities)
    
    if specific_requested_outputs.get("Normalized Sample Intensity", False):
      per_well_data.append(self.norm_sample_intensities)

    # Post-processing for calibration.
    if is_calibration:
      calibration_wells = user_parameters.get("calibration_wells")
      sample_or_ref = "sample"
      calibration_type = user_parameters.get("calibration_type")
      if calibration_type == "linear":
        degree = 1
      elif calibration_type == "quadratic":
        degree = 2
      else:
        raise Exception("Calibration type not implemented: " + \
                        calibration_type + ".")

      if image_processing_type == "hydrogen":
        # Perform the calibration.
        means_of_intensities = MeanOfIntensitiesData(self, 
          self.norm_sample_intensities.data, calibration_wells, sample_or_ref)
        sample_stds_of_intensities = \
          SampleStandardDeviationOfIntensitiesData(self, 
            self.norm_sample_intensities.data, calibration_wells, sample_or_ref)
        calibration_results = CalibrationResults.from_mean_intensity_data(self,
          means_of_intensities.data, degree, sample_or_ref)
        
        # Save the data.
        if specific_requested_outputs.get("Means of Intensities", False):
          per_plate_data.append(means_of_intensities)
        
        if specific_requested_outputs.get("Sample Standard Deviations of Intensities", 
                                          False):
          per_plate_data.append(sample_stds_of_intensities)
        
        if specific_requested_outputs.get("a", False):
          per_plate_data.append(calibration_results["a"])
        
        if specific_requested_outputs.get("b", False):
          per_plate_data.append(calibration_results["b"])
        
        if specific_requested_outputs.get("c", False):
          per_plate_data.append(calibration_results["c"])
        
        if specific_requested_outputs.get("R Squared", False):
          per_plate_data.append(calibration_results["r_squared"])

      else:
        raise Exception("Calibration is not implemented for experiment type: " + \
                        image_processing_type + ".")

      if display_figures:
        self.display_calibration_figure(independent_variable_data, 
          means_of_intensities, sample_stds_of_intensities, 
          calibration_results["r_squared"], calibration_results["a"], 
          calibration_results["b"], c = calibration_results.get("c", None), 
          figsize = figsize)
    
    # Post-processing for analysis.
    else:
      # Intensity processing has no additional post-processing.
      if image_processing_type == "intensity":
        pass
      elif image_processing_type == "hydrogen":
        # Load variables for convenience.
        reactor_calibration_coefficients = \
          user_parameters.get("reactor_calibration_coefficients")
        reactor_name = user_parameters.get("reactor_name")
        vial_volume = np.float64(user_parameters.get("vial_volume"))
        solution_volumes = np.array(user_parameters.get("solution_volumes"), 
                                    dtype = np.float64)
        time_units = user_parameters.get("time_units")
        volume_units = user_parameters.get("volume_units")
        amount_units = user_parameters.get("amount_units")
        savitzky_golay_window = user_parameters.get("savitzky_golay_window")
        savitzky_golay_degree = user_parameters.get("savitzky_golay_degree")
        incubation_threshold_fraction = \
          user_parameters.get("incubation_threshold_fraction")
        plateau_threshold_fraction = user_parameters.get("plateau_threshold_fraction")
        
        # Check the plateau and incubation threshold fractions.
        if plateau_threshold_fraction <= incubation_threshold_fraction:
          raise Exception("Plateau threshold fraction should be greater than incubation threshold fraction.")
        
        sample_or_ref = "sample"
        
        # Compute all post-processing transforms.
        hydrogen_mole_fractions = HydrogenMoleFractionData(self, 
          self.norm_sample_intensities.data, reactor_calibration_coefficients, 
          reactor_name, sample_or_ref)
        
        # Compute the headspace volumes.
        headspace_volumes = vial_volume - solution_volumes
        corrected_hydrogen_volumes = CorrectedHydrogenVolumeData(self, 
          hydrogen_mole_fractions.data, headspace_volumes, volume_units, 
          sample_or_ref, valid_data = hydrogen_mole_fractions.valid_data)
        
        hydrogen_amounts = HydrogenAmountData(self, 
          corrected_hydrogen_volumes.data, volume_units, amount_units, 
          sample_or_ref, valid_data = corrected_hydrogen_volumes.valid_data)
        
        # Valid data becomes None because we no longer have to track whether the
        # intensities are too low.
        max_hydrogen_amounts = MaxHydrogenAmountData(self, 
          hydrogen_amounts.data, amount_units, sample_or_ref, valid_data = None)
        
        hydrogen_rates = HydrogenRateData(self, hydrogen_amounts.data, 
          amount_units, time_units, savitzky_golay_window, 
          savitzky_golay_degree, sample_or_ref, 
          valid_data = hydrogen_amounts.valid_data)
        
        max_hydrogen_rates = MaxHydrogenRateData(self, hydrogen_rates.data, 
          amount_units, time_units, sample_or_ref, valid_data = None)
        
        hydrogen_incubation_times = HydrogenIncubationTimeData(self, 
          hydrogen_amounts.data, max_hydrogen_amounts.data, 
          incubation_threshold_fraction, time_units, sample_or_ref, 
          valid_data = None)
        
        hydrogen_plateau_times = HydrogenPlateauTimeData(self, 
          hydrogen_amounts.data, max_hydrogen_amounts.data, 
          plateau_threshold_fraction, time_units, sample_or_ref, 
          valid_data = None)

        # Save the requested outputs.
        if specific_requested_outputs.get("Hydrogen Mole Fraction", False):
          per_well_data.append(hydrogen_mole_fractions)

        if specific_requested_outputs.get("Corrected Hydrogen Volume", False):
          per_well_data.append(corrected_hydrogen_volumes)
        
        if specific_requested_outputs.get("Hydrogen Amount", False):
          per_well_data.append(hydrogen_amounts)
        
        if specific_requested_outputs.get("Max Hydrogen Amount", False):
          per_well_data.append(max_hydrogen_amounts)

        if specific_requested_outputs.get("Hydrogen Rate", False):
          per_well_data.append(hydrogen_rates)
        
        if specific_requested_outputs.get("Max Hydrogen Rate", False):
          per_well_data.append(max_hydrogen_rates)
        
        if specific_requested_outputs.get("Hydrogen Incubation Time", False):
          per_well_data.append(hydrogen_incubation_times)
        
        if specific_requested_outputs.get("Hydrogen Plateau Time", False):
          per_well_data.append(hydrogen_plateau_times)

      else:
        raise Exception("Analysis is not implemented for experiment type: " + \
                        image_processing_type + ".")

    return (independent_variable_data, per_well_data, per_plate_data)