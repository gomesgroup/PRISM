# Bernhard Lab Image Processing

Automated image analysis pipeline for scienfitic data, designed for reproducibility, modularity, and compatibility with Bernhard Lab instrumentation.

---

## Overview

This repository contains Python modules, configuration files, and notebooks that work together to process and analyze images in batch. Core processing logic lives in `Image_Processing_Suite`, while your raw images and binary masks reside in a separate project folder following a specific layout.

---

## Repository Structure

```plaintext
Image_Processing_Suite/
│
├── image_processing/         # Core Python modules
├── config.yaml               # Analysis parameters (thresholds, filters, etc.)
└── process_images.ipynb      # Notebook to run the full pipeline
```

## Project Data Folder Layout

Create a project folder anywhere, with this structure:

```plaintext
<Your_Project_Folder>/
│
├── input/
│   ├── images/
│   │   ├── <channel1_name>/
│   │   └── <channel2_name>/
│   └── masks/
│       ├── <channel1_name>/
│       └── <channel2_name>/
│
└── inflection_finder.ipynb   # Notebook for downstream data inspection
```

Place images and masks in their respective folders. These names should match the channel names in `config.yaml`. Ensure each channel folder under images has a matching folder under masks.

The [inflection_finder.ipynb](inflection_finder.ipynb) notebook is provided for further data inspection and analysis after processing. It will find and annote inflection points in the processed data.
