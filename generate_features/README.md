# Amide Feature Generator Setup

This guide helps you set up the environment to run the amide feature generator script.

## Quick Setup

### Option 1: Automated Setup (Recommended)

1. **Clone/Navigate to this directory**
2. **Run the setup script:**
   ```bash
   ./setup.sh
   ```
3. **Activate the environment:**
   ```bash
   mamba activate amide-features
   # or if using conda:
   conda activate amide-features
   ```

### Option 2: Manual Setup

1. **Create environment from YAML:**
   ```bash
   mamba env create -f environment.yml
   # or: conda env create -f environment.yml
   ```

2. **Activate environment:**
   ```bash
   mamba activate amide-features
   ```

## Dependencies

The environment includes:

### Python Packages
- **pandas, numpy**: Data manipulation and numerical computing
- **rdkit**: Molecular informatics and cheminformatics
- **morfeus-ml**: Molecular descriptor calculations
- **rowan-python** and **stjames**: micro-pKa calculations for acyl chlorides and molecular structure handling for rowan.
- **qupkake**: pKa calculations for amines

### External Tools
- **OpenBabel**: Chemical file format conversion
- **XTB**: Semi-empirical quantum chemistry calculations

## Usage

### Basic Example
```bash
python generate_features/amide_feature_generator.py \
  --amine_smiles 'CNCCC(OC1=CC=C(C(F)(F)F)C=C1)C2=CC=CC=C2' \
  --amine_name 'prozac' \
  --acyl_smiles 'CC(C)(C)c1ccc(cc1)C(Cl)=O' \
  --acyl_name 'acyl03' \
  --verbose
```

### Required Arguments
- `--amine_smiles`: SMILES string of the amine substrate
- `--amine_name`: Name of the amine substrate  
- `--acyl_smiles`: SMILES string of the acyl chloride substrate
- `--acyl_name`: Name of the acyl chloride substrate

### Optional Arguments
- `--verbose`: Enable verbose output

## Notes

1. **Rowan API Key**: The script includes a hardcoded Rowan API key. For production use, consider setting this as an environment variable.

2. **Output Files**: The script creates:
   - `results/` directory with JSON cache files
   - `files_xyz/` directory with XYZ coordinate files
   - `files_sdf/` directory with SDF structure files
   - Combined CSV results in `results/`

3. **External Dependencies**: Make sure `obabel`, `qupkake`, and `xtb` packages are installed.

