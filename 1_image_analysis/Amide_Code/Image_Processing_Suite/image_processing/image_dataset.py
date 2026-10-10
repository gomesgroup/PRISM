import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from image_processing.image import Image, Mask
from image_processing.intensity_data import NormalizedIntensityData

def compare_int_lists(int_lists):
  '''
    Compare all of the int lists stored in lists, returning True if all
    the int lists have the same entries.

    Inputs:
      lists (int list list): a list containing the int lists to compare.
    
    Output:
      all_lists_equal (bool): True if each int list has the same entries. False
        otherwise.
  '''
  # If there are no lists to compare, return true by default.
  if len(int_lists) == 0:
    return True
  
  # Check that the lengths are the same.
  first_len = len(int_lists[0])
  for int_list in int_lists:
    if len(int_list) != first_len:
      return False
  
  # Check that the elements are the same.
  for list_elements in zip(*int_lists):
    first_list_element = list_elements[0]
    for element in list_elements:
      if element != first_list_element:
        return False
  
  return True

class ImageDataset(object):
  '''
    This class is used to load, store, and process an entire directory of 
    images.
  '''

  def load_dataset(self):
    '''
      Loads all of the images in the directory, applying the specified image
      stride.
    
      Inputs:
        self (ImageDataset): an image dataset object with the directory 
          stored in self.directory and the image stride stored in
          self.image_stride.

      Side Effects:
        self.images (Image list): a list of the Image objects created from the
          images stored in the specified directory.

        self.num_images (int): the number of images in the dataset.

      Output:
        None
    '''
    self.images = []

    for file_name in os.listdir(self.directory):
      file_path = os.path.join(self.directory, file_name)
      image = Image(file_path, image_stride = self.image_stride)
      self.images.append(image)
    
    self.num_images = len(self.images)
  
  def extract_times(self):
    '''
      Using the loaded images stored in self.images, extract the raw times
      from the filenames, and convert the raw times to relative times, with
      the first chronological image being set to time 0.

      Inputs:
        self (ImageDataset): an image dataset object with the images loaded
          into self.images.
      
      Side Effects:
        self.raw_image_times (float list): a list containing the date (in 
          seconds since the epoch) that each image was taken.

        self.relative_image_times (float list): a list containing the time 
          elapsed in seconds for each image relative to the first chronological 
          image.
      
      Output:
        None
    '''
    self.raw_image_times = []
    self.relative_image_times = []

    for image in self.images:
      image_name_no_extension = image.image_name.split(".")[0]
      
      date_string = image_name_no_extension.split("_")[-1]

      # This reads the date from the file name, using the same format as the
      # Reactor Code Framework uses to save the images.
      date_object = time.strptime(date_string, "y%ym%md%dH%HM%MS%S")

      # Convert to seconds since the epoch.
      date_in_seconds = time.mktime(date_object)

      self.raw_image_times.append(date_in_seconds)
    
    start_time = min(self.raw_image_times)
    for raw_time in self.raw_image_times:
      self.relative_image_times.append(raw_time - start_time)

  def sort_images_by_times(self):
    '''
      Sort self.images and self.raw_image_times using self.relative_image_times.
      This step is most likely unnecessary (as the images in the file system
      are likely to be ordered by the time they were taken, given the way the
      image names are saved.); however, it is included for the case where the
      images are not loaded from the file system in the order that they were
      taken.

      Inputs:
        self (ImageDataset): an image dataset object with the images stored in
          self.image, the raw times extracted into self.raw_image_times, and
          the relative times computed and stored in self.relative_image_times.
      
      Side Effects:
        self.relative_image_times (float list): sort from smallest to largest.

        self.raw_image_times (float list): sort using the order defined by 
          sorting self.relative_image_times.

        self.images (Image list): sort using the order defined by sorting 
          self.relative_image_times.
      
      Output:
        None
    '''
    # Sort by the relative times.
    times_and_images = list(zip(self.relative_image_times, self.raw_image_times, 
                                self.images))
    times_and_images.sort(key = lambda x: x[0])

    # Unpack the sorted values into separate lists.
    self.relative_image_times = []
    self.raw_image_times = []
    self.images = []
    for relative_time, raw_time, image in times_and_images:
      self.relative_image_times.append(relative_time)
      self.raw_image_times.append(raw_time)
      self.images.append(image)

  def __init__(self, directory, image_stride = 1):
    '''
      Loads and stores all of the images in the directory, applying the 
      specified image stride. Extracts (from the file name) and stores the raw 
      and relative times for each image.

      Inputs:
        directory (str): a string that specifies the directory containing the
          images.
        
        image_stride (int): the stride over the rows and columns 
          (image_stride - 1 rows and columns will be skipped after each 
          row/column). This can be used to reduce the image size for faster 
          processing, but at the cost of throwing away pixels.
      
      Output:
        ImageDataset
    '''
    self.directory = directory
    self.image_stride = image_stride

    # Compute the name of the directory.
    self.dataset_name = os.path.basename(directory)

    # Load the images.
    self.load_dataset()

    # Extract the times from the images.
    self.extract_times()

    # Sort images by relative times.
    self.sort_images_by_times()
  
  def generate_movie(self, movie_output_directory, frame_rate):
    '''
      Turn the images stored in this dataset into a video with the specified
      frame rate, saving the result video in the specified output directory.

      NOTE: the following stack exchange post was helpful for implementing this
      function:
      https://stackoverflow.com/questions/29317262/opencv-video-saving-in-python

      Inputs:
        movie_output_directory (str): the absolute path to the directory
          in which the result video will be saved.

        frame_rate (int): the frame rate of the result video.

      Side Effects:
        1) Saves the result video in the specified output directory.
      
      Output:
        None
    '''
    image_shape = self.images[0].image.shape
    movie_output_file_path = os.path.join(movie_output_directory, 
                                          self.dataset_name + ".avi")

    # Image shape must be reversed when passed to this function.
    video_writer = cv2.VideoWriter(movie_output_file_path, 
                                   cv2.VideoWriter_fourcc("M", "J", "P", "G"), 
                                   frame_rate, (image_shape[1], image_shape[0]))

    for image in self.images:
      video_writer.write(image.image.astype(np.uint8))
    
    # Release the video writer.
    video_writer.release()

  def display_images(self, indices, figsize = (10, 10)):
    '''
      Display the images specified by the given indicies.
      #NOTE: 0-indexing is used.

      Inputs:
        self (ImageDataset): a valid ImageDataset.

        indicies (int list): a list of the indices of the images in
          self.images to display.

        figsize (int, int): width and height of output graphs in inches.

      Plots:
        1) Each specified image, titled by the file name.
      
      Output:
        None
    '''
    for i in indices:
      image = self.images[i]
      image.display_image(title_prefix = self.dataset_name, figsize = figsize)

  def display_first_image(self, figsize = (10, 10)):
    '''
      Equivalent to display_images([0]).
    '''
    self.display_images([0], figsize = (10, 10))
  
  def check_analysis_inputs(self, mask_or_mask_dataset, independent_variables, 
                            independent_variable_name, 
                            independent_variable_units):
    """
      Given a mask or a mask dataset, if it is a dataset, check that the 
      number of masks is equivalent to the number of images. Also, check that 
      all of the masks have the same grid (rows/columns) for the reference wells
      and for the sample wells. Check that the number of independent variables 
      is equivalent to the number of images (one per).

      Inputs:
        mask_or_mask_dataset (Mask or MaskDataset): a single mask (to be applied 
          to all images in the dataset) or a mask dataset, with the masks stored 
          in mask_or_mask_dataset.masks (to be applied to each image in the 
          dataset).

        independent_variables (_ list): a list of independent variables (one
          for each image).

        independent_variable_name (str): the name of the independent variable. 
          If "Time", the relative times extracted from the image names are used.
          Otherwise, use the specified independent variables.
        
        independent_variable_units (str): a string indicating the units for
          the independent. The default (to go with the "Time" default for the
          indpendent variable) is "second". For time, "second", "minute", and
          "hour" are acceptable.

      Errors:
        1) If neither a mask or a mask dataset is passed into the function,
          raise an exception.

        2) If a mask dataset is passed into the function, if the number
          of masks is not equivalent to the number of images, raise an 
          exception.

        3) If a mask dataset is passed into the function, the number
          rows or the columns per row for the reference or sample wells are
          not all the same, raise an exception.

        4) If the number of independent variables is not the same as the
          the number of images, raise an exception.

        5) If independent_variable_name is "Time", if independent_variable_units 
          is not one of "second", "minute", or "hour", raise an exception.
        
      Outputs:
        None
    """
    if (not isinstance(mask_or_mask_dataset, Mask)) and \
       (not isinstance(mask_or_mask_dataset, MaskDataset)):
      raise Exception("Must pass Mask or MaskDataset for image processing.")

    # Only perform these checks for mask datasets.
    if isinstance(mask_or_mask_dataset, MaskDataset):
      # Check the number of masks.
      if (len(mask_or_mask_dataset.masks) != self.num_images):
        raise Exception("MaskDataset does not have exactly one mask for each image.")
      
      # Gather the reference and sample columns per row from the masks.
      all_reference_columns_per_row = []
      all_sample_columns_per_row = []
      for mask in mask_or_mask_dataset.masks:
        all_reference_columns_per_row.append(mask.reference_columns_per_row)
        all_sample_columns_per_row.append(mask.sample_columns_per_row)
      
      # Check that all masks have the same number of reference rows/columns per 
      # row.
      if not compare_int_lists(all_reference_columns_per_row):
        raise Exception("MaskDataset does not have consistent number of reference rows/columns.")

      # Check that all masks have the same number of sample rows/columns per 
      # row.
      if not compare_int_lists(all_sample_columns_per_row):
        raise Exception("MaskDataset does not have consistent number of sample rows/columns.")
    
    # Check the number of independent variables.
    if len(independent_variables) != self.num_images:
      raise Exception("Number of independent variables does not match number of images.")
    
    # For "Time" independent variables, check that the units are valid.
    if independent_variable_name == "Time":
      if independent_variable_units not in ["second", "minute", "hour"]:
        raise Exception("Independent variable units for time is not one of 'second', 'minute', or 'hour'.")
  
  def analyze_images(self, mask_or_mask_dataset, analysis_channel, 
                     normalize_by_reference, independent_variable_name = "Time", 
                     independent_variable_units = "second",
                     independent_variables = None):
    '''
      Given a mask or a mask dataset, compute the average intensity of the
      specified channel for each image in the dataset (if a mask is provided,
      apply the same mask to all images; if a mask dataset is provided, the
      masks in order to each image). If normalize_by_reference is True,
      normalize each image's intensities by the mean intensity of its
      reference wells. If normalize_by_reference is False, the raw reference
      and reference intensities will be the same. Gather the raw reference, 
      reference, and sample intensities into a numpy array, and normalize
      each well by the mean intensity of the first time point.

      Inputs:
        self (ImageDataset): a valid image dataset.

        mask_or_mask_dataset (Mask or MaskDataset): a single mask (to be applied 
          to all images in the dataset) or a mask dataset, with the masks stored 
          in mask_or_mask_dataset.masks (to be applied to each image in the 
          dataset).

        analysis_channel (str): "blue", "green", "red", or "average". This 
          dictates the channel that is used for analysis ("blue", "green", and 
          "red" cause the blue, green, or red channel to be analyzed 
          respectively). "average" causes the three channels to be averaged 
          together, then analyzed.

        normalize_by_reference (bool): if True, for each image, compute the mean 
          intensity of the reference wells, and normalize the reference and 
          sample wells by this mean intensity. If false, the raw reference
          and reference intensities will be the same.
        
        independent_variable_name (str): the name of the independent variable. 
          If "Time", the relative times extracted from the image names are used.
          Otherwise, use the specified independent variables.
        
        independent_variable_units (str): a string indicating the units for
          the independent. The default (to go with the "Time" default for the
          indpendent variable) is "second". For time, "second", "minute", and
          "hour" are acceptable.

        independent_variables (_ list): a list of independent variables (one
          for each image).
      
      Errors:
        1) If neither a mask or a mask dataset is passed into the function,
          raise an exception.

        2) If a mask dataset is passed into the function, if the number
          of masks is not equivalent to the number of images, raise an 
          exception.

        3) If a mask dataset is passed into the function, the number
          rows or the columns per row for the reference or sample wells are
          not all the same, raise an exception.

        4) If the number of independent variables is not the same as the
          the number of images, raise an exception.

        5) If independent_variable_name is "Time", if independent_variable_units 
          is not one of "second", "minute", or "hour", raise an exception.

        6) If a MaskDataset is passed in, each mask's name (without file
          extension) must match the corresponding image's name (without file
          extension). If not, raise an exception.

      Outputs:
        intensity_data (IntensityData): an object that stores the reference
          columns per row, the sample columns per row, the raw reference
          intensities, the reference intensities, the sample intensities, the 
          name of the independent variable, and the independent variables.
    '''
    # If the independent variable is "Time", use the relative times extracted
    # from the image names as the independent variable.
    if independent_variable_name == "Time":
      independent_variables = self.relative_image_times
    
    # Convert the independent variables to a numpy array.
    independent_variables = np.array(independent_variables, dtype = np.float64)

    # If "Time" is the independent variable, and the user requests units other
    # than seconds, convert the time data.
    if independent_variable_name == "Time":
      if independent_variable_units == "minute":
        independent_variables = independent_variables / np.float64(60)
      elif independent_variable_units == "hour":
        independent_variables = independent_variables / np.float64(3600)
    
    # Check that the mask(s)/independent variables are valid.
    self.check_analysis_inputs(mask_or_mask_dataset, independent_variables, 
                               independent_variable_name, 
                               independent_variable_units)

    # Gather the intensities.
    reference_columns_per_row = None
    sample_columns_per_row = None
    raw_reference_intensities = []
    reference_intensities = []
    sample_intensities = []

    # Mask case.
    if isinstance(mask_or_mask_dataset, Mask):
      mask = mask_or_mask_dataset

      # Extract the columns per row for reference and sample wells.
      reference_columns_per_row = mask.reference_columns_per_row
      sample_columns_per_row = mask.sample_columns_per_row

      # Process the images and gather the intensities.
      for image in self.images:
        (image_raw_reference_intensities, image_reference_intensities, 
         image_raw_sample_intensities, image_sample_intensities) = \
          image.analyze_image(mask, analysis_channel, normalize_by_reference)
        
        raw_reference_intensities.append(image_raw_reference_intensities)
        reference_intensities.append(image_reference_intensities)
        sample_intensities.append(image_sample_intensities)

    # MaskDataset case.
    elif isinstance(mask_or_mask_dataset, MaskDataset):
      mask_dataset = mask_or_mask_dataset

      # Extract the columns per row for reference and sample wells.
      # It should be the same for every mask.
      first_mask = mask_dataset.masks[0]
      reference_columns_per_row = first_mask.reference_columns_per_row
      sample_columns_per_row = first_mask.sample_columns_per_row

      # Process the images and gather the intensities.
      for image, mask in zip(self.images, mask_dataset.masks):
        # Check that the mask corresponds to the image.
        if mask.get_name_no_extension() != image.get_name_no_extension():
          raise Exception("For calibration, corresponding masks and images must have identical file names (disregarding the file extension).")

        (image_raw_reference_intensities, image_reference_intensities, 
         image_raw_sample_intensities, image_sample_intensities) = \
          image.analyze_image(mask, analysis_channel, normalize_by_reference)
        
        raw_reference_intensities.append(image_raw_reference_intensities)
        reference_intensities.append(image_reference_intensities)
        sample_intensities.append(image_sample_intensities)

    # Convert to numpy arrays.
    raw_reference_intensities = np.array(raw_reference_intensities)
    reference_intensities = np.array(reference_intensities)
    sample_intensities = np.array(sample_intensities)

    # Normalize by the first time point.
    norm_raw_reference_intensities = \
      raw_reference_intensities / raw_reference_intensities[0]
    norm_reference_intensities = \
      reference_intensities / reference_intensities[0]
    global norm_sample_intensities
    norm_sample_intensities = sample_intensities
    
    # Create the intensity data.
    intensity_data = NormalizedIntensityData(self.dataset_name, 
      analysis_channel, reference_columns_per_row, sample_columns_per_row, 
      norm_raw_reference_intensities, norm_reference_intensities, 
      norm_sample_intensities, independent_variable_name, 
      independent_variable_units, independent_variables)
    
    return intensity_data
  
  def get_norm_sample_intensities(self):
    np.savetxt("intensity_data.csv",
        norm_sample_intensities,
        delimiter =", ",
        fmt ='% s')


class MaskDataset(ImageDataset):
  '''
    This sub-class of ImageDataset load, store, and display masks. It is meant 
    to be used when there is one mask per image.
  '''

  def load_dataset(self):
    '''
      Override load_dataset from the parent class ImageDataset. The one
      difference being that the images are loaded with the Mask class instead
      of the Image class.
    '''
    self.images = []

    for file_name in os.listdir(self.directory):
      file_path = os.path.join(self.directory, file_name)
      image = Mask(file_path, self.overall_columns_per_row, self.ref_or_empty, 
                   image_stride = self.image_stride)
      self.images.append(image)
    
    self.num_images = len(self.images)

  def __init__(self, directory, overall_columns_per_row, ref_or_empty, 
               image_stride = 1):
    '''
      Loads and stores all of the masks in the directory, applying the 
      specified image stride. The masks will be sorted by the relative
      times extracted from the file names.

      Inputs:
        directory (str): a string that specifies the directory containing the
          masks.
        
        overall_columns_per_row (int list): a list of the number of columns in 
          each row.

        ref_or_empty ((str, str) dict): a dictionary which specifies
          the wells that are references or empty (keys are strings 
          with the format "(well_row,well_col)", and entries are "ref" or 
          "empty").
        
        image_stride (int): the stride over the rows and columns 
          (image_stride - 1 rows and columns will be skipped after each 
          row/column). This can be used to reduce the image size for faster 
          processing, but at the cost of throwing away pixels.
      
      Output:
        MaskDataset
    '''
    self.overall_columns_per_row = overall_columns_per_row
    self.ref_or_empty = ref_or_empty

    super().__init__(directory, image_stride)

    # Copy the data into more intuitive variable names for masks.
    self.masks = self.images
    self.num_masks = self.num_images

  def display_mask_validations(self, mask_output_directory, 
                               display_figures = True, 
                               only_display_first = True, figsize = (10, 10), 
                               fontsize = 8):
    '''
      For each Mask in the dataset, display an image of the mask. Display the 
      overall (well row, well column) pairs at the centroid of that well. Also,
      display the reference (well row, well column) pairs and the sample 
      (well_row, well_column) pairs. The name of each mask (extracted from the 
      file name of the mask) will be included in each plot. Finally, save each
      generated figure.
      NOTE: rows and columns are 0-indexed.

      Inputs:
        self (MaskDataset): a valid mask dataset object.

        mask_output_directory (str): the directory for saving the mask figures.

        display_figures (bool): if False, the figures are closed, preventing
          in-line display in the python notebook.
        
        only_display_first (bool): If True, pass display_figures to the first 
          mask only, and pass False to the rest. Otherwise, pass display_figures
          to every mask.

        figsize (int, int): width and height of output graphs in inches.

        fontsize (int): size of the font for the well annotations.
      
      Plots (for each mask):
        1) An image of the mask.

        2) A graph of the overall (well row, well column) pairs at the centroid
          of the well.  

        3) A graph of the reference (well row, well column) pairs at the
          centroid of the well.
        
        4) A graph of the sample (well row, well column) pairs at the centroid
          of the well.
      
      Side Effects (for each mask):
        1) Saves each of the plots generated in a sub-directory of 
          overall_mask_output_directory.

      Output:
        None
    '''
    for i, mask in enumerate(self.masks):
      if only_display_first:
        mask_display_figures = (i == 0) and display_figures
      else:
        mask_display_figures = display_figures
      mask.display_mask_validation(mask_output_directory, 
                                   title_prefix = self.dataset_name, 
                                   display_figures = mask_display_figures, 
                                   figsize = figsize, 
                                   fontsize = fontsize)