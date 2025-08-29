import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def remove_whitespace(s):
  '''
    Given a string s, remove all whitespace (tabs, new lines, and spaces) from
    the string.

    Inputs:
      s (str): a string
    
    Output:
      str: s with the whitespace (tabs, new lines, and spaces) removed.
  '''
  s = s.replace("\t", "")
  s = s.replace("\n", "")
  while "  " in s:
    s = s.replace("  ", "")
  s = s.replace(" ", "")
  return s

class Image(object):
  '''
  This class is used to load, store, and manipulate individual images.
  '''

  def __init__(self, file_path, image_stride = 1):
    '''
      Loads the image, possibly disregarding parts of the image based on the
      provided stride.

      Inputs:
        file_path (str): a string that specifies the file path of the image.
    
        image_stride (int): the stride over the rows and columns 
          (image_stride - 1 rows and columns will be skipped after each 
          row/column). This can be used to reduce the image size for faster 
          processing, but at the cost of throwing away pixels.
  
      Output:
        Image
    '''
    self.file_path = file_path
    self.image_name = os.path.basename(file_path)
    self.image_stride = image_stride

    # Load the image from the file path.
    #NOTE: OpenCV reads color images as BGR.
    self.image = cv2.imread(self.file_path, cv2.IMREAD_COLOR)

    # Apply the image stride to the image. For example, if 
    # image_stride is 2 this removes every other row and every other column.
    self.image = self.image[::self.image_stride, ::self.image_stride]
  
  def get_name_no_extension(self):
    '''
      Returns the file name of the image without the extension.

      Inputs:
        self (Image): a valid image object.
      
      Output:
        str: self.image_name with the extension removed.
    '''
    return self.image_name.split(".")[0]

  def display_image(self, title_prefix = None, figsize = (10, 10)):
    '''
      Display the image.

      Inputs:
        self (Image): a valid image object.

        title_prefix (str): a prefix to the title of the plot. If None, no
          prefix will be added.

        figsize (int, int): width and height of output graphs in inches.
      
      Plots:
        1) The image, titled by the file name.
      
      Output:
        None
    '''

    image_in_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize = figsize)

    if title_prefix == None:
      title = "Image: " + self.image_name
    else:
      title = "(" + title_prefix +")" + " Image: " + self.image_name

    ax.set_title(title)

    ax.imshow(image_in_rgb)
  
  def check_analysis_inputs(self, mask, analysis_channel, 
                            normalize_by_reference):
    '''
      Given a mask, channel to analyze, and boolean signifying normalization,
      raise an exception if the parameters are invalid.

      Inputs:
        self (Image): a valid image.

        mask (Mask): a valid mask of the same size as the image.

        analysis_channel (str): "blue", "green", "red", or "average". This 
          dictates the channel that is used for analysis ("blue", "green", and 
          "red" cause the blue, green, or red channel to be analyzed 
          respectively). "average" causes the three channels to be averaged 
          together, then analyzed.
        
        normalize_by_reference (bool): if True, compute the mean intensity of
          the reference wells, and normalize the reference and sample wells
          by this mean intensity.
      
      Errors:
        1) if the mask is not the same size as the image, an exception is 
          raised.
        2) if the provided analysis_channel is not one of "blue", "green", 
          "red", or "average", an exception is raised.
        3) if normalize_by_reference is True, but the mask contains no
          reference wells, an exception is raised.
    '''
    # Check that the image/mask shape match.
    if self.image.shape != mask.image.shape:
      raise Exception("Image shape (" + str(self.image.shape) + \
                      ") does not match mask shape (" + \
                      str(mask.image.shape) + ").")
    
    # Check that the analysis channel is valid. 
    if analysis_channel not in ["blue", "green", "red", "average"]:
      raise Exception("Channel specified for image analysis (" + \
                      analysis_channel + ") is invalid.")
    
    # Check that the reference wells exist if normalization is requested.
    if normalize_by_reference and len(mask.reference_columns_per_row) == 0:
      raise Exception("Normalization requested but no reference well specified.")

  def analyze_image(self, mask, analysis_channel, normalize_by_reference):
    '''
      Given a mask, apply the mask to the image, and create a numpy array 
      (can be indexed by the sample well row and sample well column) that 
      contains the mean intensity of the specified channel in each sample well. 
      Also, create a numpy array that contains the average intensity of the 
      specified channel in each reference well. If normalize_by_reference is
      True, use the mean intensity of the reference wells to normalize the
      sample and reference well intensities.

      Inputs:
        self (Image): a valid image.

        mask (Mask): a valid mask of the same size as the image.

        analysis_channel (str): "blue", "green", "red", or "average". This 
          dictates the channel that is used for analysis ("blue", "green", and 
          "red" cause the blue, green, or red channel to be analyzed 
          respectively). "average" causes the three channels to be averaged 
          together, then analyzed.
        
        normalize_by_reference (bool): if True, compute the mean intensity of
          the reference wells, and normalize the reference and sample wells
          by this mean intensity.

      Errors:
        1) if the mask is not the same size as the image, an exception is 
          raised.
        
        2) if the provided channel is not one of "blue", "green", "red", 
          or "average", an exception is raised.
        
        3) if normalize_by_reference is True, but the mask contains no
          reference wells, an exception is raised.
      
      Outputs:
        raw_reference_intensities (np.float64 np.ndarray): a numpy array 
          that can be indexed with the reference well row and reference well 
          column to obtain the average intensity of the specified channel in 
          that reference well.

        reference_intensities (np.float64 np.ndarray): 
          if normalize_by_reference is True:
            raw_reference_intensities normalized by the mean of 
            raw_reference_intensities.
          if normalize_by_reference is False:
            equivalent to raw_reference_intensities.

        raw_sample_intensities (np.float64 np.ndarray): a numpy array that 
          can be indexed with the sample well row and sample well column to 
          obtain the average intensity of the specified channel in that sample 
          well.

        sample_intensities (np.float64 np.ndarray): 
          if normalize_by_reference is True:
            raw_sample_intensities normalized by the mean of 
            raw_reference_intensities.
          if normalize_by_reference is False:
            equivalent to raw_sample_intensities.
    '''
    # Check the validity of the inputs.
    self.check_analysis_inputs(mask, analysis_channel, normalize_by_reference)

    # Get the data from the requested analysis channel.
    if analysis_channel == "blue":
      image_data = self.image[:, :, 0].astype(np.float64)
    elif analysis_channel == "green":
      image_data = self.image[:, :, 1].astype(np.float64)
    elif analysis_channel == "red":
      image_data = self.image[:, :, 2].astype(np.float64)
    elif analysis_channel == "average":
      image_data = np.mean(self.image, axis = 2, dtype = np.float64)

    # Create arrays to hold the reference and sample well average intensities.
    # The default value is np.nan.
    num_reference_rows = len(mask.reference_columns_per_row)
    max_reference_cols = max(mask.reference_columns_per_row)
    raw_reference_intensities = np.full((num_reference_rows, 
                                         max_reference_cols),
                                        np.nan, dtype = np.float64)

    num_sample_rows = len(mask.sample_columns_per_row)
    max_sample_cols = max(mask.sample_columns_per_row)
    raw_sample_intensities = np.full((num_sample_rows, max_sample_cols), 
                                     np.nan, dtype = np.float64)

    # Compute the reference and sample well intensities.
    well_types = ["ref", "sample"]
    columns_per_row_to_use = [mask.reference_columns_per_row, 
                              mask.sample_columns_per_row]
    intensity_arrays = [raw_reference_intensities, raw_sample_intensities]

    for well_type, columns_per_row, well_intensities in zip(well_types, 
      columns_per_row_to_use, intensity_arrays):
      for row, num_columns in enumerate(columns_per_row):
        for col in range(num_columns):
          well_mask_array, bb_top, bb_left, bb_height, bb_width = \
            mask.get_well_mask(well_type, row, col)
          pixel_area = mask.get_well_pixel_area(well_type, row, col)

          # Get the sub-image specified by the well mask bounding box.
          sub_image = image_data[bb_top : bb_top + bb_height, 
                                bb_left : bb_left + bb_width]
          
          # Compute the average intensity.
          average_intensity = np.sum(well_mask_array * sub_image) / pixel_area

          well_intensities[row, col] = average_intensity
    
    # If specified, compute the mean of the reference wells, and then normalize
    # the results by this mean.
    if normalize_by_reference:
      ref_mean = np.mean(raw_reference_intensities)

      reference_intensities = raw_reference_intensities / ref_mean
      sample_intensities = raw_sample_intensities / ref_mean
    else:
      reference_intensities = np.copy(raw_reference_intensities)
      sample_intensities = np.copy(raw_sample_intensities)
    
    return (raw_reference_intensities, reference_intensities, 
            raw_sample_intensities, sample_intensities)

class Mask(Image):
  '''
  This sub-class of Image is used to store and process the provided mask.
  '''

  def threshold_mask(self):
    '''
      Using the grayscale image stored in self.gray_image, compute a thresholded
      image (any pixel with an intensity greater than 127 is set to white), and 
      store the thresholded image in self.threshed_image.

      Inputs:
        self (Mask): a Mask object with a grayscale image stored in 
          self.gray_image.
      
      Side Effects:
        self.threshed_image (np.uint8 np.ndarray): stores the thresholded image.
      
      Output:
        None
    '''
    thresh, self.threshed_image = cv2.threshold(self.gray_image, 127, 255, 
                                                cv2.THRESH_BINARY)

  def detect_connected_components(self):
    '''
      Using the thresholded image stored in self.threshed_image, detect the
      connected components, storing the number of connected components, 
      an array of the labels of each pixel, an array of connected components
      stats, and an array of the centroids of each connected component.

      Inputs:
        self (Mask): a Mask object with a thresholded image stored in 
                     self.threshed_image

      Side Effects:
        self.background_label (int): stores the label of the background 
          connected component
        
        self.num_ccs (int): stores the total number of connected components.

        self.labels (int32 np.ndarray): stores the connected component label of
          each pixel.
        
        self.centroids (np.float64 np.ndarray, self.num_ccs by 2): stores the
          (x, y) coordinate of the centroid of each connected component.
        
        self.bb_lefts, self.bb_tops, self.bb_widths, self.bb_heights
          (np.int32 np.ndarray): stores the left, top, width, and height of the
          bounding box for each connected component.
        
        self.pixel_areas (np.int32 np.ndarray): stores the number of pixels in
          each connected component.

      Output:
        None
    '''
    self.background_label = 0
    self.num_ccs, self.labels, stats, self.centroids = \
      cv2.connectedComponentsWithStats(self.threshed_image, 4)

    self.bb_lefts = stats[:, cv2.CC_STAT_LEFT]
    self.bb_tops = stats[:, cv2.CC_STAT_TOP]
    self.bb_widths = stats[:, cv2.CC_STAT_WIDTH]
    self.bb_heights = stats[:, cv2.CC_STAT_HEIGHT]
    self.pixel_areas = stats[:, cv2.CC_STAT_AREA]

  def determine_well_ordering(self):
    '''
      Using the centroids of the connected components, as well as the number
      of columns in each row (self.overall_columns_per_row), infer the connected
      component associated with each well. Then, split the well orderings based
      on sample wells and reference wells. Additionally, generate
      columns_per_row int lists for the sample and reference well grids.

      Inputs:
        self (Mask): a Mask object with the columns per row list stored in
          self.overall_columns_per_row and the centroids of the conncected 
          components stored in self.centroids.
      
      Side Effects:
        self.sample_well_to_cc (int list list): a list of lists,
          there is one inner list per row of sample wells, where each inner
          list stores the connected component label of each sample well in the
          row.
        
        self.reference_well_to_cc (int list list): a list of lists,
          there is one inner list per row of reference wells, where each inner
          list stores the connected component label of each reference well in 
          the row.
          
      Outputs:
        None
    '''
    self.overall_well_to_cc = []
    # Start at label 1 to avoid background label. Sort connected component 
    # centroids by y coordinate. Add 1 to account for background label.
    cc_sorted_by_y = np.argsort(self.centroids[1:, 1]) + 1

    # Split the connected components into rows.
    cc_rows = []
    start_index = 0
    for num_columns in self.overall_columns_per_row:
      cc_rows.append(cc_sorted_by_y[start_index : start_index + num_columns])
      start_index += num_columns
    
    # Sort the connected components within each row (left to right).
    for cc_row in cc_rows:
      column_ordering = np.argsort(self.centroids[cc_row, 0])
      cc_row_sorted_by_x = cc_row[column_ordering]
      self.overall_well_to_cc.append(list(cc_row_sorted_by_x))
    
    # Split into sample and reference wells.
    self.sample_well_to_cc = []
    self.reference_well_to_cc = []
    
    # Split the well connected components into sample and reference
    # grids.
    for well_row, num_columns in enumerate(self.overall_columns_per_row):
      sample_row = []
      reference_row = []

      for well_col in range(num_columns):
        well_type = self.ref_or_empty.get((well_row, well_col), "sample")
        if well_type == "sample":
          sample_row.append(self.overall_well_to_cc[well_row][well_col])
        elif well_type == "ref":
          reference_row.append(self.overall_well_to_cc[well_row][well_col])

      if len(sample_row) > 0:
        self.sample_well_to_cc.append(sample_row)
      
      if len(reference_row) > 0:
        self.reference_well_to_cc.append(reference_row)
    
    # Create the columns_per_row int lists for the sample and reference wells.
    self.sample_columns_per_row = []
    self.reference_columns_per_row = []

    for cc_row in self.sample_well_to_cc:
      self.sample_columns_per_row.append(len(cc_row))
    
    for cc_row in self.reference_well_to_cc:
      self.reference_columns_per_row.append(len(cc_row))
  
  def check_input_image(self):
    '''
      Check that the input image has only black and white pixels, and raise
      an exception if this is not the case.

      Inputs:
        self (Mask): a Mask object.
      
      Side Effects:
        1) If self.image contains pixels that are anything but completely
          black or completely white, print a warning.

      Output:
        None
    '''
    if (np.size(self.image) != np.count_nonzero(self.image == 0) + \
      np.count_nonzero(self.image == 255)):
      print("WARNING: Mask input image contains pixels that are not completely white or black. PNG masks are preferable to JPG masks.")
  
  def check_well_row_and_col(self, well_row, well_col):
    '''
      Check that the well row and well column are within the bounds specified
      by self.overall_columns_per_row.

      Inputs:
        self (Mask): a Mask object.

        well_row (int): the 0-indexed row of the target well.

        well_col (int): the 0-indexed column of the target well.
      
      Output:
        (bool): return False if the well row/column is invalid, and return
          True otherwise.
    '''
    if well_row >= len(self.overall_columns_per_row) or well_row < 0:
      return False

    if well_col >= self.overall_columns_per_row[well_row] or well_col < 0:
      return False
    
    return True
  
  def check_mask(self):
    '''
      Check that a Mask is valid, raising an exception if it is not.

      Inputs:
        self (Mask): a Mask object.

      Errors:
        1) If the total number of columns in self.overall_columns_per_row is not 
          equal to the number of connected components detected in the mask, 
          raise an exception.
        
        2) Check the keys in self.ref_or_empty. If any of the
           (well row, well col) pairs are invalid, raise an exception.
        
        3) Check the values in self.ref_or_empty. If any of the values are
           not "ref" or "empty", raise an exception.

      Output:
        None
    '''
    # The total number of wells specified should be equal to the number of
    # connected components minus 1 (for the background connected component).
    if sum(self.overall_columns_per_row) != (self.num_ccs - 1):
      raise Exception("Mask Invalid: Specified Number of Wells (" + \
                      str(sum(self.overall_columns_per_row)) + \
                      ") is Not Equal to the Number of Connected Components Detected in the Mask (" + \
                      str(self.num_ccs - 1) + ").")

    # Check that the reference well and column pairs are valid.
    for (well_row, well_col) in list(self.ref_or_empty.keys()):
      if not self.check_well_row_and_col(well_row, well_col):
        raise Exception("Mask Invalid: Reference/blank well row/column pair (" + \
                        str(well_row) + "," + str(well_col) + ") is invalid.")
    
    # Check that each entry in self.ref_or_empty is either "ref" or "empty".
    for value in list(self.ref_or_empty.values()):
      if value != "ref" and value != "empty":
        raise Exception("Mask Invalid: Label (" + value + \
                        ") for reference/blank well row/column pair (" + \
                        str(well_row) + "," + str(well_col) + \
                        ") is not one of 'ref' or 'empty'.")
  
  def generate_cc_masks(self):
    '''
      Using the number of connected components stored in self.num_ccs, the
      label array stored in self.labels, and the bounding box details stored
      in self.bb_lefts, self.bb_tops, self.bb_widths, and self.bb_heights, 
      generate a mask for each connected component (cropped to exactly contain 
      the connected component).

      Inputs:
        self (Mask): a Mask object with the number of connected components
          in self.num_ccs, the label array in self.labels, and the connected
          component bounding box details in self.bb_lefts, self.bb_tops,
          self.bb_widths, and self.bb_heights.
        
      Side Effects:
        self.cc_masks ((int32 np.ndarray) list): a list of the masks for
          each connected component.

      Output:
        None
    '''
    self.cc_masks = []
    for label in range(self.num_ccs):
      bb_top = self.bb_tops[label]
      bb_left = self.bb_lefts[label]
      bb_height = self.bb_heights[label]
      bb_width = self.bb_widths[label]

      # Crop the mask using the bounding box.
      cropped_cc_mask = (self.labels[bb_top : bb_top + bb_height, 
                                     bb_left : bb_left + bb_width] == label) * 1

      self.cc_masks.append(cropped_cc_mask)

  def preprocess_ref_or_empty(self):
    '''
      Convert the keys from the ref_or_empty dictionary read from the config 
      file from strings into tuples (the keys should be string representations
      of tuples).

      Inputs:
        self (Mask): a mask object that has the ref_or_empty dictionary
          stored in self.ref_or_empty.

      Side Effects:
        self.ref_or_empty (((int, int), str) dict): a dictionary that maps
          (well_row, well_col) pairs to "ref" (for reference wells) or "empty"
          (for empty wells).

      Errors:
        1) If the keys in ref_or_empty do not match the format "(a,b)", where
          a is the 0-indexed well row and b is the 0-indexed well column, raise
          and exception.

      Output:
        None
    '''
    # Catch the case where ref_or_empty is None.
    if self.ref_or_empty == None:
      self.ref_or_empty = dict()
    else:
      temp_ref_or_empty = self.ref_or_empty
      self.ref_or_empty = dict()
      
      # For each key, convert to a tuple and create an entry in the new
      # dictionary.
      for key_string in temp_ref_or_empty.keys():
        original_key_string = key_string

        # Remove whitespace from the key string.
        key_string = remove_whitespace(key_string)

        # Remove the opening parenthesis.
        if len(key_string) == 0 or key_string[0] != "(":
          raise Exception("ref_or_empty keys must begin with '('.")
        key_string = key_string[1 :]
        
        # Remove the closing parenthesis
        if len(key_string) == 0 or key_string[-1] != ")":
          raise Exception("ref_or_empty keys must end with ')'.")
        key_string = key_string[: -1]

        # Split on the comma.
        if key_string.count(",") != 1:
          raise Exception("ref_or_empty keys must contain exactly one comma.")
        well_row_string, well_col_string = key_string.split(",")

        # Check that the well row and well col strings are not empty.
        if well_row_string == "" or well_col_string == "":
          raise Exception("ref_or_empty keys must have a row and column pair.")
        
        # Convert the well_row_string to the well_row number.
        for c in well_row_string:
          if c not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            raise Exception("ref_or_empty keys must have a numerical well row.")
        well_row = int(well_row_string)

        # Convert the well_col_string to the well_col number.
        for c in well_col_string:
          if c not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            raise Exception("ref_or_empty keys must have a numerical well column.")
        well_col = int(well_col_string)

        # Copy the data into the new ref_or_empty dictionary, using the
        # well_row, well_col tuple as the key.
        self.ref_or_empty[(well_row, well_col)] = \
          temp_ref_or_empty[original_key_string]


  def __init__(self, file_path, overall_columns_per_row, ref_or_empty, 
               image_stride = 1):
    '''
      Loads the mask, possibly disregarding parts of the mask based on the
      provided stride. Applies a binary threshold to the mask, detects the
      connected components, creates the an ordering of the connected
      components (wells), based on the specified number of well rows 
      (num_well_rows) and well columns (num_well_columns) and which wells are
      references or empty (ref_or_empty), and generates the masks for each 
      connected component.

      Inputs:
        file_path (str): a string that specifies the complete file path of the
          image.
        
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
        Mask
    '''
    super().__init__(file_path, image_stride)
    self.overall_columns_per_row = overall_columns_per_row
    self.ref_or_empty = ref_or_empty

    # Replace the keys in ref or empty with their tuple equivalents.
    self.preprocess_ref_or_empty()

    # Check that the input image only contains white or black pixels.
    self.check_input_image()

    # Convert the image to grayscale.
    self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    # Compute the binary threshold.
    self.threshold_mask()

    # Detect the connected components.
    self.detect_connected_components()

    # Determine the corresponding connected component for each well.
    self.determine_well_ordering()

    # Generate the masks for each connected component.
    self.generate_cc_masks()

    # Check that the mask is valid.
    self.check_mask()

  def get_well_cc_label(self, well_type, well_row, well_col):
    '''
      Given a Mask, a well type, a well row, and a well column, return the 
      connected component label corresponding to the well.

      Inputs:
        self (Mask): a valid Mask object.

        well_type (str): "sample" or "ref" for sample and reference wells
          respectively. "overall" can be used to get any well; this should NOT 
          be used by the user.

        well_row (int): the 0-indexed row of the target well.

        well_col (int): the 0-indexed column of the target well.
      
      Outputs:
        cc_label (int): the label of the connected component corresponding to
          the specified well.
    '''
    if well_type == "overall":
      return self.overall_well_to_cc[well_row][well_col]
    if well_type == "sample":
      return self.sample_well_to_cc[well_row][well_col]
    if well_type == "ref":
      return self.reference_well_to_cc[well_row][well_col]
  
  def get_well_mask(self, well_type, well_row, well_col):
    '''
      Given a Mask, a well type, a well row, and  a well column, return a mask 
      array for that connected component that corresponds to that well, as well 
      as the left and top of the bounding box, and the height and width of the 
      bounding box.

      Inputs:
        self (Mask): a valid Mask object.

        well_type (str): "sample" or "ref" for sample and reference wells
          respectively. "overall" can be used to get any well; this should NOT 
          be used by the user.

        well_row (int): the 0-indexed row of the target well.

        well_col (int): the 0-indexed column of the target well.

      Outputs:
        well_mask_array (int32 np.ndarray): an array of the mask for the 
          specified well, with ones in the position of the connected component
          corresponding to the well, and 0's elsewhere.

        bb_top (int): the row in the overall image of the top-most pixel 
          in the well mask's bounding box.

        bb_left (int): the column in the overall image of the left-most pixel
          in the well mask's bounding box.
        
        bb_height (int): the height of the well mask's bounding box in pixels.

        bb_width (int): the width of the well mask's bounding box in pixels.

    '''
    cc_label = self.get_well_cc_label(well_type, well_row, well_col)

    well_mask_array = self.cc_masks[cc_label]

    bb_top = self.bb_tops[cc_label]
    bb_left = self.bb_lefts[cc_label]
    bb_height = self.bb_heights[cc_label]
    bb_width = self.bb_widths[cc_label]

    return well_mask_array, bb_top, bb_left, bb_height, bb_width

  def get_well_centroid(self, well_type, well_row, well_col):
    '''
      Given a Mask, a well type, a well row, and a well column, return the 
      centroid of the well.

      Inputs:
        self (Mask): a valid Mask object.

        well_type (str): "sample" or "ref" for sample and reference wells
          respectively. "overall" can be used to get any well; this should NOT 
          be used by the user.

        well_row (int): the 0-indexed row of the target well.

        well_col (int): the 0-indexed column of the target well.
      
      Output:
        centroid (int): the centroid of the specified well.
    '''
    cc_label = self.get_well_cc_label(well_type, well_row, well_col)
    centroid = self.centroids[cc_label]
    return centroid

  def get_well_pixel_area(self, well_type, well_row, well_col):
    '''
      Given a Mask, a well type, a well row, and a well column, return the pixel 
      area of the well.

      Inputs:
        self (Mask): a valid Mask object.

        well_type (str): "sample" or "ref" for sample and reference wells
          respectively. "overall" can be used to get any well; this should NOT 
          be used by the user.

        well_row (int): the 0-indexed row of the target well.

        well_col (int): the 0-indexed column of the target well.

      Output:
        pixel_area (int): the number of pixels in the specified well.
    '''
    cc_label = self.get_well_cc_label(well_type, well_row, well_col)
    pixel_area = self.pixel_areas[cc_label]
    return pixel_area

  def display_mask_validation(self, mask_output_directory, title_prefix = None, 
                              display_figures = True, figsize = (10, 10), 
                              fontsize = 8):
    '''
      Given a Mask, display an image of the mask. Display the overall
      (well row, well column) pairs at the centroid of that well. Also,
      display the reference (well row, well column) pairs and the sample 
      (well_row, well_column) pairs. The name of the mask (extracted from the 
      file name of the mask) will be included in each plot. Finally, save each
      generated figure.
      NOTE: rows and columns are 0-indexed.

      Inputs:
        self (Mask): a valid Mask object.

        mask_output_directory (str): the directory for saving the mask figures.

        title_prefix (str): a prefix to the title of the plot. If None, no
          prefix will be added.

        display_figures (bool): if False, the figures are closed, preventing
          in-line display in the python notebook.

        figsize (int, int): width and height of output graphs in inches.

        fontsize (int): size of the font for the well annotations.
      
      Plots:
        1) An image of the mask.

        2) A graph of the overall (well row, well column) pairs at the centroid
          of the well.  

        3) A graph of the reference (well row, well column) pairs at the
          centroid of the well.
        
        4) A graph of the sample (well row, well column) pairs at the centroid
          of the well.
      
      Side Effects:
        1) Saves each of the plots generated in the specified directory
          (mask_output_directory).

      Output:
        None
    '''
    mask_name = self.get_name_no_extension()

    # Swap to RGB (since OpenCV reads images in BGR).
    image_in_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    # Show the mask.
    if title_prefix == None:
      title = "Mask: " + self.image_name
    else:
      title = "(" + title_prefix + ")" + " Mask: " + self.image_name
    fig, ax = plt.subplots(figsize = figsize)
    ax.set_title(title)
    ax.imshow(image_in_rgb)

    # Save the image of the mask.
    fig.savefig(os.path.join(mask_output_directory, mask_name + "_mask.png"))

    if not display_figures:
      plt.close(fig = fig)

    # Create a white image of the same shape to display the orderings.
    all_white_image = np.full(image_in_rgb.shape, 255, dtype = np.uint8)

    # Display each of the orderings.
    well_types = ["overall", "ref", "sample"]
    titles = ["Overall Well Ordering: " + self.image_name + 
              "\nDON'T USE FOR ANALYSIS\nUSE REFERENCE/SAMPLE ORDERING BELOW", 
              "Reference Well Ordering: " + self.image_name, 
              "Sample Well Ordering: " + self.image_name]
    columns_per_row_to_use = [self.overall_columns_per_row, 
                              self.reference_columns_per_row, 
                              self.sample_columns_per_row]

    for well_type, title, columns_per_row in zip(well_types, 
      titles, columns_per_row_to_use):
      fig, ax = plt.subplots(figsize = figsize)

      if title_prefix != None:
        title = "(" + title_prefix + ")" + " " + title

      ax.set_title(title)

      ax.imshow(all_white_image)

      # Plot invisible points for the annotations to center on.
      ax.scatter(self.centroids[1:, 0], self.centroids[1:, 1], marker = " ")
    
      # Plot the annotations.
      for row, num_columns in enumerate(columns_per_row):
        for col in range(num_columns):
          centroid = self.get_well_centroid(well_type, row, col)
          well_str = "(" + str(row) + "," + str(col) + ")"
          ax.annotate(well_str, (centroid[0], centroid[1]), ha = "center", 
                      va = "center", fontsize = fontsize)
      
      # Save the ordering figure.
      fig.savefig(os.path.join(mask_output_directory, mask_name + "_" + \
                               well_type + "_ordering" + ".png"))

      if not display_figures:
        plt.close(fig)
      