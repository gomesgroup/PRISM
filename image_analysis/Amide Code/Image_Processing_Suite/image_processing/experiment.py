import numpy as np
import yaml
import os
import shutil
import csv

from image_processing.image import Image, Mask
from image_processing.image_dataset import ImageDataset, MaskDataset
from image_processing.intensity_data import NormalizedIntensityData
from image_processing.per_plate_data import HydrogenMoleFracIndepVar

class UserParameters(object):
  '''
    A class for handling user parameters from the user config file.
  '''

  #Given a file path, path, ending in .txt, read the contents of the file and 
  #return the contents as a string.
  def read_text_file(self, path):
    '''
      Given a file path to a text file, read the contents of the file and
      output the contents as a string.

      NOTE: This function was modified from 15-112:
        http://www.krivers.net/15112-s19/notes/notes-strings.html

      Inputs:
        self (UserParameters): a user parameters object.

        path (str): the absolute or relative file path of a .txt file.

      Output:
        str: a string of the contents of the file specified by path.
    '''
    with open(path, "rt") as f:
        return f.read()
  
  def read_parameters_from_yaml(self, yaml_file_path):
    '''
      Given a path to a yaml file, read the file into python as a dictionary.

      NOTE: Documentation can be found at the following sources:
        https://pyyaml.org/wiki/PyYAMLDocumentation
        https://www.cloudbees.com/blog/yaml-tutorial-everything-you-need-get-started
      
      Inputs:
        yaml_file_path (str): a string that specifies the yaml file path.
      
      Output:
        (str, _) dict: a dictionary where the variable names (as strings) from
          the yaml file are mapped to their corresponding data.
    '''
    return yaml.safe_load(self.read_text_file(yaml_file_path))
  
  def __init__(self, config_file_path):
    '''
      Given a user config file path, load the user parameter dictionary from the
      specified user config yaml file.

      Inputs:
        config_file_path (str): a string that specifies the user config file 
          path.

      Errors:
        1) If the config_file_path is not a yaml file, raise an exception.

      Output:
        UserParameters
    '''
    # Check that the user config file is a yaml file.
    if config_file_path.split(".")[-1] != "yaml":
      raise Exception("User config file must be a yaml file.")
    
    self.config_file_path = config_file_path
    
    self.user_parameters = self.read_parameters_from_yaml(config_file_path)
  
  def get(self, variable_name):
    '''
      Given a variable name string, attempt to fetch the associated value
      from the parameters loaded from the

      Inputs:
        variable_name (str): the name of the variable in the user config file.
      
      Errors:
        1) If the requested variable name is not in the user config file, raise
          a KeyError.

      Output:
        _: the value of the variable.
    '''
    try:
      variable_value = self.user_parameters[variable_name]
      return variable_value
    except KeyError:
      raise KeyError("Variable '" + variable_name + \
                     "' not found in config file '" + \
                     self.config_file_path + "'.")

class Experiment(object):
  '''
    A class for managing the data from an experiment.
  '''

  def read_input_directory_structure(self):
    '''
      Check that the input directory structure is valid.

      Inputs:
        self (Experiment): a valid experiment object, with 
          "parent_data_directory", "project_name", and "experiment_name"
          variables defined in the user config file.

      Side Effects:
        1) self.experiment_directory (str): a string that represents the
          absolute path to the experiment directory.

        2) self.input_directory (str): a string that represents the absolute
          path to the input directory (stored within the experiment directory).

        3) self.image_directories ((str, str) dict): a dictionary mapping the
          name of the image directory to its absolute path.

        4) self.mask_directories_and_file_paths ((str, (str, str List)) dict): 
          a dictionary mapping the name of the image directory to a tuple of the 
          corresponding mask directory and a list of the absolute file path
          of each mask.
      
      Errors:
        1) If the experiment directory does not exist, raise an exception.

        2) If an "input" directory does not exist within the experiment
          directory, raise an exception.

        3) If the "images" or "masks" directories do not exist within the
          "input" directory, raise an exception.

        4) Within the "masks" directory, each mask (or mask dataset) should
          be placed in a folder with a name that matches the name of the
          corresponding image directory. If not, raise an exception.

        5) If "is_calibration" is False, there should only be one mask file in
          each folder. If not, raise an exception.

      Output:
        None
    '''
    # Check the experiment directory.
    self.experiment_directory = \
      os.path.join(self.user_parameters.get("parent_data_directory"), 
                   self.user_parameters.get("project_name"), 
                   self.user_parameters.get("experiment_name"))
    if not os.path.isdir(self.experiment_directory):
      raise Exception("The experiment directory does not exist. Please check the 'parent_data_directory', 'project_name', and 'experiment_name' variables in the config file.")
    
    # Check the input directory.
    self.input_directory = os.path.join(self.experiment_directory, "input")
    if not os.path.isdir(self.input_directory):
      raise Exception("The 'input' sub-directory in the experiment directory does not exist.")
    
    # Check that the images and masks directories exist.
    all_images_directory = os.path.join(self.input_directory, "images")
    if not os.path.isdir(all_images_directory):
      raise Exception("The 'images' sub-directory in the 'input' directory does not exist.")

    all_masks_directory = os.path.join(self.input_directory, "masks")
    if not os.path.isdir(all_masks_directory):
      raise Exception("The 'masks' sub-directory in the 'input' directory does not exist.")
    
    # Find the image directories.
    self.image_directories = dict()
    for image_directory_name in os.listdir(all_images_directory):
      self.image_directories[image_directory_name] = \
        os.path.join(all_images_directory, image_directory_name)

    # Find the mask directories.
    self.mask_directories_and_file_paths = dict()
    for mask_directory_name in os.listdir(all_masks_directory):
      if mask_directory_name not in self.image_directories:
        raise Exception("The directory for each mask must have a name that matches the corresponding image directory.")
      
      mask_directory = os.path.join(all_masks_directory, mask_directory_name)

      mask_file_paths = []
      for mask_file_name in os.listdir(mask_directory):
        mask_file_paths.append(os.path.join(mask_directory, mask_file_name))
        
      # Mask directories for analysis should only have one mask.
      if not self.user_parameters.get("is_calibration") and \
        len(mask_file_paths) > 1:
          raise Exception("For analysis, each mask directory should only have one mask.")
      
      self.mask_directories_and_file_paths[mask_directory_name] = \
        (mask_directory, mask_file_paths)

  def check_user_inputs(self):
    '''
      Check that the user inputs are valid.
      NOTE: this function will not check if all the necessary variables exist;
      it will only check the validity of the exist variables.

      Inputs:
        self (Experiment): a valid experiment object.

      Errors:
        1) If any mask directory does not have an associated requested
          outputs, raise an exception.

        2) If any of the elements of "movies_to_be_generated" does not
          correspond to one of the image dataset's directory name, raise
          an exception.
      
      Output:
        None
    '''
    # Check that each mask has a corresponding entry in the requested
    # outputs dictionary.
    for mask_directory_name in self.mask_directories_and_file_paths:
      requested_outputs = self.user_parameters.get("requested_outputs")
      if mask_directory_name not in requested_outputs:
        raise Exception("Mask was provided but no outputs were requested for that mask.")

    # movies are for folders that exist.
    movie_names = self.user_parameters.get("movies_to_be_generated")
    for movie_name in movie_names:
      if movie_name not in self.image_directories:
        raise Exception("Every movie name must correspond to the directory name for an image dataset.")

  def make_output_directory(self):
    '''
      Make the output directories.

      Inputs:
        self (Experiment): a valid experiment object.

      Side Effects:
        1) self.output_directory (str): the output directory for the experiment.

        2) Create a "data", a "masks", and a "movies" sub-directory in the
          "output" directory.

        3) For each input mask directory, generate a sub-directory in the "data"
          sub-directory. For each analysis channel, create a sub-directory in
          this new directory.
        
        4) self.data_output_directories ((str, (str, str) dict) dict): a nested
          dictionary storing the absolute paths generated as discussed in
          "3)". The top level is indexed by the name of the mask directory, and
          the inner level is indexed by the analysis channel.

        5) For each input mask directory, generate a sub-directory in the 
          "masks" sub-directory.

        6) self.mask_output_directories ((str, str) dict): a dictionary storing
          the absolute paths generated in "5)".

        7) For each input image directory specified in "movies_to_be_generated",
          generate a sub-directory in the "movies" sub-directory.

        8) self.movie_output_directories ((str, str) dict): a dictionary storing
          the absolute paths generated in "7)".
      
      Output:
        None
    '''
    # Make the output directory.
    self.output_directory = os.path.join(self.experiment_directory, "output")
    os.makedirs(self.output_directory, exist_ok = True)

    # Make the data output directories.
    analysis_channels = self.user_parameters.get("analysis_channels")
    overall_data_output_directory = os.path.join(self.output_directory, "data")
    self.data_output_directories = dict()

    for mask_directory_name in self.mask_directories_and_file_paths:
      # Create the data output directory for the mask/corresponding image
      # directory.
      data_output_directory = os.path.join(overall_data_output_directory, 
                                           mask_directory_name)
      os.makedirs(data_output_directory, exist_ok = True)

      # Create the output directories for each channel
      channel_output_directories = dict()
      for analysis_channel in analysis_channels:
        # Create the channel output directory.
        channel_output_directory = os.path.join(data_output_directory, 
                                                analysis_channel)
        os.makedirs(channel_output_directory, exist_ok = True)

        channel_output_directories[analysis_channel] = channel_output_directory

      self.data_output_directories[mask_directory_name] = \
        channel_output_directories

    # Make the mask output directories.
    overall_mask_output_directory = os.path.join(self.output_directory, "masks")
    self.mask_output_directories = dict()

    for mask_directory_name in self.mask_directories_and_file_paths:
      mask_output_directory = os.path.join(overall_mask_output_directory, 
                                           mask_directory_name)
      
      os.makedirs(mask_output_directory, exist_ok = True)
    
      self.mask_output_directories[mask_directory_name] = mask_output_directory

    # Make the movie output directories.
    overall_movie_output_directory = os.path.join(self.output_directory, 
                                                  "movies")
    self.movie_output_directories = dict()

    movie_names = self.user_parameters.get("movies_to_be_generated")
    for movie_name in movie_names:
      movie_output_directory = os.path.join(overall_movie_output_directory, 
                                            movie_name)
      os.makedirs(movie_output_directory, exist_ok = True)

      self.movie_output_directories[movie_name] = movie_output_directory

  def __init__(self, config_file_path):
    '''
      Load the data from the input directories and create the output 
      directories.

      Inputs:
        config_file_path (str): a string that specifies the user config file 
          path.

      Side Effects:
        See sub-functions.
      
      Errors:
        See sub-functions.

      Output:
        Experiment
    '''
    self.user_parameters = UserParameters(config_file_path)

    # Read input directory structure and check that it is valid.
    self.read_input_directory_structure()

    # Check that the config file inputs are valid.
    self.check_user_inputs()

    # Create the output directory (and sub-directories).
    self.make_output_directory()
  
  def get_user_parameters(self):
    '''
      Return the UserParameters object created from reading the user config 
      file.

      Inputs:
        self (Experiment): a valid experiment object.
      
      Output:
        UserParameters: the user parameters object created by reading the user
          config file.
    '''
    return self.user_parameters
  
  def process_masks_or_mask_datasets(self):
    '''
      Process the mask or mask dataset file paths into mask or masker dataset 
      objects.
      
      Inputs:
        self (Experiment): a valid experiment object.

    Plots:
        1) The mask validation figures for each mask, if "display_figures" is
          True.
      
      Side Effects:
        1) self.masks_or_mask_datasets ((str, Mask or MaskDataset) dict): a 
          dictionary that stores the masks for analysis or mask datasets for
          calibration. The keys for the masks or mask_datasets are the names
          of the directories for the mask images.

        2) The mask validation figures are saved.
      
      Output:
        None
    '''
    self.masks_or_mask_datasets = dict()
    columns_per_row = self.user_parameters.get("columns_per_row")
    ref_or_empty = self.user_parameters.get("ref_or_empty")
    
    for mask_directory_name in self.mask_directories_and_file_paths:
      mask_directory, mask_file_paths = \
        self.mask_directories_and_file_paths[mask_directory_name]
      mask_output_directory = self.mask_output_directories[mask_directory_name]

      if self.user_parameters.get("is_calibration"):
        mask_dataset = MaskDataset(mask_directory, columns_per_row, 
                                   ref_or_empty, image_stride = 1)
        
        self.masks_or_mask_datasets[mask_directory_name] = mask_dataset

        mask_dataset.display_mask_validations(mask_output_directory, 
          display_figures = self.user_parameters.get("display_figures"), 
          only_display_first = True)
      
      else:
        mask = Mask(mask_file_paths[0], columns_per_row, ref_or_empty, 
                    image_stride = 1)
        
        self.masks_or_mask_datasets[mask_directory_name] = mask

        mask.display_mask_validation(mask_output_directory, 
          title_prefix = mask_directory_name, 
          display_figures = self.user_parameters.get("display_figures"))

  def load_image_datasets(self):
    '''
      Using the masks and mask datasets generated from 
      process_masks_or_mask_datasets, process the corresponding image
      directories.
      
      Inputs:
        self (Experiment): a valid experiment object.

      Side Effects:
        self.image_datasets ((str, ImageDataset) dict): a dictionary that stores
          the loaded image datasets. The keys for the image datasets correspond
          to the names of the image directories.
      
      Plots:
        1) The first image of each dataset, if "display_figures" is True.
          
      Output:
        None
    '''
    self.image_datasets = dict()
    
    # Image datasets are only processed if they have a corresponding mask.
    for image_directory_name in self.image_directories:
      image_dataset_directory = self.image_directories[image_directory_name]
      
      image_dataset = ImageDataset(image_dataset_directory, image_stride = 1)

      self.image_datasets[image_directory_name] = image_dataset

      if self.user_parameters.get("display_figures"):
        image_dataset.display_first_image()
  
  def generate_movies(self):
    '''
      For each image directory in "movies_to_be_generated", generate a movie 
      from the images in the corresponding dataset, and save the movie in the
      corresponding output directory.
      
      Inputs:
        self (Experiment): a valid experiment object.
      
      Side Effects:
        1) For each image directory in "movies_to_be_generated", generate a 
          movie from the images, and save the movie in the corresponding output 
          directory.
      
      Output:
        None
    '''
    movie_names = self.user_parameters.get("movies_to_be_generated")
    for movie_name in movie_names:
      image_dataset = self.image_datasets[movie_name]
      image_dataset.generate_movie(self.movie_output_directories[movie_name], 
        self.user_parameters.get("frames_per_second"))

  def process_image_datasets_intensity(self):
    '''
      For each mask or mask dataset, for each analysis channel, process the 
      corresponding image dataset with the specified analysis channel. Store
      the resulting intensity datasets in a nested dictionary, where the first
      level is indexed by the image data set name, and the second level is
      indexed by the analysis channel. Display the raw reference, reference, and
      sample intensities for each intensity dataset if "display_figures" is 
      True.
      
      Inputs:
        self (Experiment): A valid experiment object.
      
      Side Effects:
        self.intensity_datas ((str, (str, NormalizedIntensityData) dict) dict): 
          a nested dictionary that stores the resultant normalized intensity 
          data, where the first level is indexed by the image data set name, and 
          the second level is indexed by the analysis channel.
          NOTE: only image datasets with corresponding masks will be analyzed.

      Plots (for each intensity dataset, if "display_figures" is True):
        1) The raw reference intensity (normalized by the first data point)
          against the independent variable for each reference well.

        2) The reference intensity (normalized by the first data point) against 
          the independent variable for each reference well.

        3) The sample intensity (normalized by the first data point) against the
          independent variable for each sample well.

      Errors:
        1) If calibration is not implemented for the specified implementation
          type, raise an exception.

      Output:
        None
    '''
    image_processing_type = self.user_parameters.get("image_processing_type")
    is_calibration = self.user_parameters.get("is_calibration")
    display_figures = self.user_parameters.get("display_figures")
    analysis_channels = self.user_parameters.get("analysis_channels")
    normalize_by_reference = self.user_parameters.get("normalize_by_reference")

    # Set up the independent variable for calibration.
    if is_calibration:
      if image_processing_type == "hydrogen":
        injected_hydrogen_volumes = \
          np.array(self.user_parameters.get("injected_hydrogen_volumes"), 
                   dtype = np.float64)
        initial_air_volume = np.float64(self.user_parameters.get("vial_volume"))
        volume_units = self.user_parameters.get("volume_units")

        independent_variable_name = "Hydrogen Mole Fraction"
        independent_variable_units = None
        hydrogen_mole_fraction = HydrogenMoleFracIndepVar(None, 
          injected_hydrogen_volumes, initial_air_volume, volume_units, None)
        independent_variables = hydrogen_mole_fraction.data
      else:
        raise Exception("Calibration is not implemented for image processing type: " + \
                        image_processing_type + ".")
    
    self.intensity_datas = dict()

    # For each mask, process the corresponding image datasets.
    for mask_directory_name in self.mask_directories_and_file_paths:
      image_dataset = self.image_datasets[mask_directory_name]
      mask_or_mask_dataset = self.masks_or_mask_datasets[mask_directory_name]

      channel_intensity_datas = dict()
      global intensity_data

      # Process the image dataset for each specified analysis channel.
      for analysis_channel in analysis_channels:
        if is_calibration:
          intensity_data = image_dataset.analyze_images(mask_or_mask_dataset, 
            analysis_channel, normalize_by_reference, 
            independent_variable_name = independent_variable_name, 
            independent_variable_units = independent_variable_units, 
            independent_variables = independent_variables)
        else:
          time_units = self.user_parameters.get("time_units")

          intensity_data = image_dataset.analyze_images(mask_or_mask_dataset, 
            analysis_channel, normalize_by_reference, 
            independent_variable_units = time_units)
        
        # Store in the specific channel's intensity dataset.
        channel_intensity_datas[analysis_channel] = intensity_data

        # Display the intensity data.
        if display_figures:
          intensity_data.display_raw_reference_intensities()
          intensity_data.display_reference_intensities()
          intensity_data.display_sample_intensities()

      # Store all the channel intensity datasets for the image dataset.
      self.intensity_datas[mask_directory_name] = channel_intensity_datas

  def check_requested_output(self, image_processing_type, is_calibration, 
                             requested_output):
    '''
      Check that a specific requested output is valid. Raise an exception if
      not.

      Inputs:
        self (Experiment): a valid experiment object.

        image_processing_type (str): a string that describes the type of image
          processing. Currently, "intensity" and "hydrogen" are supported.
        
        is_calibration (bool): a boolean that specifies if calibration is
          occuring.

        requested_output (str): a string that represents the requested output.
    
      Errors:
        1) If the requested output is invalid, raise an exception.
      
      Output:
        None
    '''
    if is_calibration:
      if image_processing_type == "hydrogen":
        if requested_output not in ["Normalized Raw Reference Intensity", 
                                    "Normalized Reference Intensity", 
                                    "Normalized Sample Intensity", 
                                    "Means of Intensities", 
                                    "Sample Standard Deviations of Intensities", 
                                    "a", 
                                    "b", 
                                    "c", 
                                    "R Squared"]:
          raise Exception("Calibration requested output (" + \
                          requested_output + ") is invalid.")
      
      else:
        raise Exception("Calibration not implemented for image processing type: " + \
                        image_processing_type + ".")
    
    else:
      if image_processing_type == "intensity":
        if requested_output not in ["Normalized Raw Reference Intensity", 
                                    "Normalized Reference Intensity", 
                                    "Normalized Sample Intensity"]:
          raise Exception("Analysis requested output (" + requested_output + \
                          ") is invalid.")
      
      elif image_processing_type == "hydrogen":
        if requested_output not in ["Normalized Raw Reference Intensity", 
                                    "Normalized Reference Intensity", 
                                    "Normalized Sample Intensity", 
                                    "Hydrogen Mole Fraction", 
                                    "Corrected Hydrogen Volume", 
                                    "Hydrogen Amount", 
                                    "Max Hydrogen Amount", 
                                    "Hydrogen Rate", 
                                    "Max Hydrogen Rate", 
                                    "Hydrogen Incubation Time", 
                                    "Hydrogen Plateau Time"]:
          raise Exception("Analysis requested output (" + requested_output + \
                          ") is invalid.")
      
      else:
        raise Exception("Analysis not implemented for image processing type: " + \
                        image_processing_type + ".")

  def check_requested_outputs(self):
    '''
      Check that the requested outputs are valid. Raise an exception if they are 
      not.

      Inputs:
        self (Experiment): a valid experiment object.

      Errors:
        1) If any of the requested outputs dictionary are invalid, raise an 
          exception.
      
      Output:
        None
    '''
    image_processing_type = self.user_parameters.get("image_processing_type")
    is_calibration = self.user_parameters.get("is_calibration")
    requested_outputs = self.user_parameters.get("requested_outputs")

    for image_dataset_name in requested_outputs:
      image_dataset_requested_outputs = requested_outputs[image_dataset_name]

      for requested_output in image_dataset_requested_outputs:
        self.check_requested_output(image_processing_type, is_calibration, 
                                    requested_output)
      
  def post_process_intensity_datasets(self):
    '''
      For each intensity data created by processing the image datasets, 
      post-process the intensity data according to the user-specified
      transformations in the config file. Store the resulting data in a
      two-level dictionary, with the same shape and indexing as 
      self.intensity_datas.

      Inputs:
        self (Experiment): a valid experiment object.

      Side Effects:
        self.post_processed_datas ((str, (str, (PerPlateData, PerWellData List, 
          PerPlateData List)) dict) dict): a two-level, nested dictionary, with 
          the first level being indexed by the image dataset name and the second 
          level being indexed by the analysis channel. For each entry, store a 
          3-tuple, with the first entry being the independent variable 
          data, the second entry being a list of the per-well properties, and 
          the third entry being a list of the per-plate properties.
      
      Output:
        None
    '''
    self.check_requested_outputs()

    self.post_processed_data = dict()

    # For each image dataset represented in the intensity datas.
    for image_dataset_name in self.intensity_datas:
      channel_post_processed_data = dict()
      
      # For each analysis channel.
      for analysis_channel in self.intensity_datas[image_dataset_name]:
        intensity_data = \
          self.intensity_datas[image_dataset_name][analysis_channel]
        
        channel_post_processed_data[analysis_channel] = \
          intensity_data.post_process(self.user_parameters)
      
      self.post_processed_data[image_dataset_name] = channel_post_processed_data

  def display_post_processed_data(self):
    '''
      If "display_figures" is True, display all the figures specified by the
      requested outputs.

      Inputs:
        self (Experiment): a valid experiment object.

      Plots:
        1) For every per-well and per-plate property specified by the
          user in "requested_outputs", display the associated figure/print
          the associated data.

      Output:
        None
    '''
    if self.user_parameters.get("display_figures"):
      for image_dataset_name in self.post_processed_data:
        for analysis_channel in self.post_processed_data[image_dataset_name]:
          independent_variable_data, per_well_data, per_plate_data = \
            self.post_processed_data[image_dataset_name][analysis_channel]
          
          for per_well_property in per_well_data:
            per_well_property.display()

          for per_plate_property in per_plate_data:
            per_plate_property.display()

  def replace_none_with_string(self, lst):
    '''
      Given a list, generate a new list where all of the None are replaced with 
      the string "NONE".

      Inputs:
        self (Experiment): A valid experiment object.
        
        lst (_ List): A list, possibly with None.
      
      Output:
        _ List: The same as lst, but all of the None are replaced with "NONE".
    '''
    new_lst = []
    for elem in lst:
      if elem == None:
        new_lst.append("NONE")
      else:
        new_lst.append(elem)
    return new_lst

  def export_post_processed_data(self):
    '''
      Export the per-well and per-plate data specified by the user in 
      "requested_outputs".

      NOTE: the following documentation was used for csv writing:
      https://docs.python.org/3/library/csv.html

      Inputs:
        self (Experiment): a valid experiment object.

      Side Effects:
        1) For every per-well and per-plate property specified by the
          user in "requested_outputs", save the data to the output csv.

        2) Copy the config file into the experiment directory.

      Output:
        None
    '''
    # The column titles for all but the data columns.
    first_column_titles = \
      ["Type", "Computed from Sample or Reference Wells", 
       "Against Independent Variable", "Property Name", 
       "Property Units", "Well Row (0-indexed)", "Well Column (0-indexed)"]
    
    for image_dataset_name in self.post_processed_data:
      for analysis_channel in self.post_processed_data[image_dataset_name]:
        independent_variable_data, per_well_data, per_plate_data = \
          self.post_processed_data[image_dataset_name][analysis_channel]
        
        # Add "Data" columns for each image.
        num_images = self.image_datasets[image_dataset_name].num_images
        column_titles_csv_row = first_column_titles + ["Data"] * num_images
        
        data_output_directory = \
          self.data_output_directories[image_dataset_name][analysis_channel]
        
        csv_file_path = os.path.join(data_output_directory, "data.csv")

        with open(csv_file_path, "w", newline = "") as csv_file:
          csv_writer = csv.writer(csv_file, delimiter = ",", 
            quotechar = "'", quoting = csv.QUOTE_MINIMAL)
          
          # Write the column titles.
          csv_writer.writerow(column_titles_csv_row)

          # Write the independent variable data.
          independent_variable_csv_row = \
            ["independent_variable", 
             independent_variable_data.sample_or_ref, 
             independent_variable_data.against_independent_variable,
             independent_variable_data.data_display_name,
             independent_variable_data.data_units,
             None,
             None] + \
            list(independent_variable_data.data)
          
          independent_variable_csv_row = \
            self.replace_none_with_string(independent_variable_csv_row)  
          csv_writer.writerow(independent_variable_csv_row)

          # Save the per-well sample properties.
          for per_well_property in per_well_data:
            normalized_intensity_data = \
              per_well_property.normalized_intensity_data
            
            # Determine which rows/columns to output.
            if per_well_property.sample_or_ref == "sample":
              columns_per_row = normalized_intensity_data.sample_columns_per_row
            elif per_well_property.sample_or_ref == "ref":
              columns_per_row = \
                normalized_intensity_data.reference_columns_per_row
            
            # Output the data for all rows/columns specified by columns per row.
            for row, num_columns in enumerate(columns_per_row):
              for col in range(num_columns):
                per_well_csv_row = \
                  ["per_well",
                  per_well_property.sample_or_ref,
                  per_well_property.against_independent_variable,
                  per_well_property.data_display_name,
                  per_well_property.data_units,
                  row,
                  col]
                
                if per_well_property.against_independent_variable:
                  data_to_output = per_well_property.data[:, row, col]
                  per_well_csv_row.extend(list(data_to_output))
                else:
                  data_to_output = per_well_property.data[row, col]
                  per_well_csv_row.append(data_to_output)

                per_well_csv_row = \
                  self.replace_none_with_string(per_well_csv_row)
                csv_writer.writerow(per_well_csv_row)

          # Save the per-plate data.
          for per_plate_property in per_plate_data:
            per_plate_csv_row = \
              ["per_plate", 
               per_plate_property.sample_or_ref,
               per_plate_property.against_independent_variable,
               per_plate_property.data_display_name,
               per_plate_property.data_units,
               None,
               None]
            
            if per_plate_property.against_independent_variable:
              per_plate_csv_row.extend(list(per_plate_property.data))
            else:
              per_plate_csv_row.append(per_plate_property.data)

            per_plate_csv_row = \
              self.replace_none_with_string(per_plate_csv_row)
            csv_writer.writerow(per_plate_csv_row)

    # Copy the config file.
    final_config_path = os.path.join(self.experiment_directory, 
      os.path.basename(self.user_parameters.config_file_path))
    shutil.copy2(self.user_parameters.config_file_path, final_config_path)
    