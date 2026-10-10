# Image analysis

Turns the PRISM plate images into the inflection (half-conversion) time of
every well. The output is one workbook per plate,
`<group> + <acyl chloride> times_and_concs.xlsx`, which is the input to
`2_rate_fitting`. All 99 plate workbooks used in the study are in
`2_rate_fitting/inputs/original_plate_files_99.zip`.

## Contents

| path | what it is |
|---|---|
| `Image_Processing_Suite/image_processing/` | Python modules: read images and masks, extract per-well colour intensity over time |
| `Image_Processing_Suite/5-15-24-Ninetales.yaml` | configuration for one experiment (well layout, reference wells, channels, smoothing) |
| `Image_Processing_Suite/run_image_processing-5-15-24-Ninetales.ipynb` | runs the suite with that configuration |
| `Ninetales_Template/` | example project folder: well masks (`input/masks/white`, `input/masks/yellow`) and `inflection_finder.ipynb` |

## Running it

1. Make a project folder with `input/images/<channel>/` (the plate images) and
   `input/masks/<channel>/` (one mask per channel). The channel names must
   match the configuration. `Ninetales_Template/` shows the layout; it holds
   the masks but not the images.
2. Set `parent_data_directory`, `project_name` and `experiment_name` in the
   `.yaml` file. `parent_data_directory` is currently the path on the original
   lab computer (`e:\Image Analysis`) and has to be changed.
3. Run `run_image_processing-5-15-24-Ninetales.ipynb`, then
   `inflection_finder.ipynb` in the project folder.

## Notes

- `inflection_finder.ipynb` also fits rate constants by the original method.
  That part is superseded by `2_rate_fitting` and is marked in the notebook;
  it is kept for the record.
- In the inflection finder, `plot_ranges` is set by hand for each plate to
  the wells that reacted. Times are written only for wells inside the range,
  so a blank cell in a plate workbook can mean either no colour change or a
  well left out of the range.
- `inflection_finder.ipynb` is large (about 4 MB) because it keeps the outputs
  of an example plate.
