#!/bin/bash

# Setup script for Amide Feature Generator Environment
# This script sets up the conda/mamba environment needed to run the amide feature generator

set -e  # Exit on any error

echo "=== Amide Feature Generator Environment Setup ==="
echo ""

# Check if mamba is available, fallback to conda
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "Using mamba for package management"
else
    CONDA_CMD="conda"
    echo "Using conda for package management"
fi

# Create the environment
echo "Creating environment 'amide-features'..."
$CONDA_CMD env create -f environment.yml

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To activate the environment, run:"
echo "  $CONDA_CMD activate amide-features"
echo ""
echo "To test the installation, run:"
echo "  python generate_features/amide_feature_generator.py --help"
echo ""
echo "Example usage:"
echo "  python generate_features/amide_feature_generator.py \\"
echo "    --amine_smiles 'CNCCC(OC1=CC=C(C(F)(F)F)C=C1)C2=CC=CC=C2' \\"
echo "    --amine_name 'prozac' \\"
echo "    --acyl_smiles 'CC(C)(C)c1ccc(cc1)C(Cl)=O' \\"
echo "    --acyl_name 'acyl03' \\"
echo "    --verbose"
echo ""
echo "Note: Make sure you have the Rowan API key configured in the script."
