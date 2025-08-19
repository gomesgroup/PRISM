import os
import sys
import glob
import subprocess
import argparse
from contextlib import contextmanager

import pandas as pd
import numpy as np
import json

from openbabel import pybel
from morfeus import read_xyz, XTB, Sterimol, SASA, BuriedVolume, Pyramidalization, Dispersion, LocalForce
from rdkit import Chem
from rdkit.Chem import AllChem

import rowan
import stjames
rowan.api_key = "rowan-sk86fcce05-d2a2-4562-a0bc-1c9fab1c5cb5"

@contextmanager
def changed_dir(sub_dir, results_folder="files_sdf"):
    """
    Context manager to change directory and return to original afterwards.
    
    Args:
        sub_dir (str): Subdirectory to change to
        results_folder (str): Base results folder
    """
    previous_dir = os.getcwd()
    target_dir = os.path.join(results_folder, sub_dir)
    safe_create_dir(target_dir)
    os.chdir(target_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)
        
def safe_create_dir(dir_path: str):
    """
    Method to create a directory if it doesn't already exist.
    
    Args:
        dir_path (str): Path to directory to create
    """
    if not (os.path.exists(dir_path) and os.path.isdir(dir_path)):
        os.makedirs(dir_path)
        print(f"Created directory: {dir_path}")

def save_final_results(results: dict, substrate_type: str="amine"):
    os.makedirs('results', exist_ok=True)
    
    existing_results = {}
    results_file = f'results/results_{substrate_type}s.json'
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                existing_results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_results = {}
    existing_results.update(results)

    with open(results_file, 'w') as f:
        json.dump(existing_results, f)

def check_if_computed(smiles_cannonical: str, name: str, substrate_type: str="amine", verbose: bool = False):
    os.makedirs('results', exist_ok=True)
    try:
        with open(f'results/results_{substrate_type}s.json', 'r') as f:
            final_results = json.load(f)
        
        if smiles_cannonical in final_results:
            if verbose:
                print(f"Descriptors already computed for {name} (SMILES: {smiles_cannonical}). Using cached results.")
            return final_results[smiles_cannonical]
    except FileNotFoundError:
        if verbose:
            print("No cached results found. Starting fresh computation.")
        return {}

def convert_smile_to_xyz(smiles: str, name: str, output_dir: str = 'files_xyz'):
    
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    xyz_file = os.path.join(output_dir, f"{name}.xyz")
    # from rdkit to xyz block
    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    
    # Get conformer
    conf = mol.GetConformer()
    
    # Write XYZ file
    with open(xyz_file, 'w') as f:
        f.write(f"{mol.GetNumAtoms()}\n")
        f.write(f"{name}\n")
        
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            f.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")
    
    return xyz_file

def convert_sdf_to_xyz(sdf_file, output_dir='files_xyz'):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_name = os.path.basename(sdf_file)
    xyz_name = os.path.splitext(base_name)[0] + '.xyz'
    output_path = os.path.join(output_dir, xyz_name)
    
    suppl = Chem.SDMolSupplier(sdf_file)
    molecule_count = len([mol for mol in suppl if mol is not None])
    # print(f"Number of molecules in the SDF: {molecule_count}")
    
    try:
        if molecule_count > 1:
            sdf_output_path = sdf_file.replace('.sdf', '_first.sdf')
            _cmd0 = ["sed", r"/^\$\$\$\$/q", sdf_file]
            result = subprocess.run(_cmd0, check=True, capture_output=True, text=True)
            
            with open(sdf_output_path, "w") as f_out:
                f_out.write(result.stdout)
            
            _cmd1 = ["mv", sdf_output_path, sdf_file]
            subprocess.run(_cmd1, check=True, capture_output=True, text=True)
            
            _cmd2 = ["rm", "-f", sdf_output_path]
            subprocess.run(_cmd2, check=True)

        cmd = ['obabel', '-isdf', sdf_file, '-oxyz', '-O', output_path]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        # print(f"Converted {sdf_file} to {output_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error converting {sdf_file} to XYZ: {e}")
        return None
    
    return output_path

def get_nitrogen_indices(SMILES):
    '''Get atom index for nitrogen on 0-based index'''
    mol = Chem.MolFromSmiles(SMILES)

    nitrogen_indices = []
    secondary_indices = []
    amine_type = ""
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'N':
            neighbors = atom.GetNeighbors()
            # num_of_Hs = len([atom.GetIdx() for atom in neighbors if atom.GetSymbol() == 'H'])
            
            if len(neighbors) < 3:
                nitrogen_indices.append(atom.GetIdx())
                
                aromatic_carbons = []
                alkyl_carbons = []
                for neighbor in neighbors:
                    if neighbor.GetSymbol() == 'C':
                        if neighbor.GetIsAromatic():
                            aromatic_carbons.append(neighbor.GetIdx())
                        else:
                            alkyl_carbons.append(neighbor.GetIdx())
                            
                if len(aromatic_carbons) > 0 and len(alkyl_carbons) > 0:
                    amine_type = f"2_mixture"
                elif len(aromatic_carbons) > 0:
                    amine_cat = len(aromatic_carbons)
                    amine_type = f"{amine_cat}_aromatic"
                elif len(alkyl_carbons) > 0:
                    amine_cat = len(alkyl_carbons)
                    amine_type = f"{amine_cat}_aliphatic"
                
                secondary_idx = [atom.GetIdx() for atom in neighbors if atom.GetSymbol() != 'H']
                secondary_indices.extend(secondary_idx)
                
    return ",".join([str(idx) for idx in nitrogen_indices]), ",".join([str(idx) for idx in secondary_indices]), amine_type

def get_alpha_carbon_indices(smiles: str):
    '''Get atom index for alpha carbon on 0-based index'''
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    alpha_carbon_index, num_hydrogens, acyl_type = [], 0, ""
    main_index, secondary_indices = [], []
    
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            #### find carbonyl carbon(Cl = 17)
            has_chlorine_neighbor = any(neighbor.GetAtomicNum() == 17 for neighbor in atom.GetNeighbors())
        
            if has_chlorine_neighbor:
                main_index.append(atom.GetIdx())
                main_neighbors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() != 'H' and n.GetSymbol() != 'O']
                secondary_indices.extend(main_neighbors)

                #### find alpha carbon(C = 6) with hydrogens
                ring_info = mol.GetRingInfo()
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 6: 
                        num_hydrogens = len([h for h in neighbor.GetNeighbors() if h.GetAtomicNum() == 1])
                        has_hydrogen = any(h.GetAtomicNum() == 1 for h in neighbor.GetNeighbors())
                        
                        if has_hydrogen:
                            alpha_carbon_index.append(neighbor.GetIdx())
                            acyl_type = "aliphatic"
                            for n in neighbor.GetNeighbors():
                                if n.GetIsAromatic():
                                    acyl_type = "mixture"  
                        else:
                            if neighbor.GetIsAromatic():
                                is_heteroaromatic = False
                                for ring in ring_info.AtomRings():
                                    if neighbor.GetIdx() in ring:
                                        ring_atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
                                        has_heteroatom = any(atom.GetSymbol() not in ['C', 'H'] for atom in ring_atoms)
                                        if has_heteroatom:
                                            is_heteroaromatic = True
                                            break
                                if is_heteroaromatic:
                                    acyl_type = "hetero"
                                else:
                                    acyl_type = "aromatic"

    # alpha carbon, main atom, secondary carbon, num hydrogens
    alpha = ",".join([str(idx) for idx in list(set(alpha_carbon_index))])
    main = ",".join([str(idx) for idx in list(set(main_index))])
    secondary = ",".join([str(idx) for idx in list(set(secondary_indices))])
    return alpha, main, secondary, num_hydrogens, acyl_type

def process_additional_features(df_amine, df_acyl):
    selected_feats = ["acyl_class_aromatic", "acyl_Charges_secondary_1", "acyl_pka_aHs_x_has_acidic_H", "acyl_BV_secondary_2", 
                    "amine_class_1_mixture", "amine_Charges_secondary_1", "amine_pka_basic", "amine_BV_secondary_avg"]

    #### process categorical variables
    # Process categorical variables - create binary columns for each class
    if 'amine_class' in df_amine.columns:
        unique_classes = ["1_aliphatic", "1_aromatic", "1_mixture", "2_mixture", "2_aliphatic"]
        for class_val in unique_classes:
            df_amine[f'amine_class_{class_val}'] = (df_amine['amine_class'] == class_val)
        df_amine = df_amine.drop(columns=['amine_class'])

    if 'acyl_class' in df_acyl.columns:
        unique_classes =["aromatic", "aliphatic", "hetero", "mixture"]
        for class_val in unique_classes:
            df_acyl[f'acyl_class_{class_val}'] = (df_acyl['acyl_class'] == class_val)
        df_acyl = df_acyl.drop(columns=['acyl_class'])

    # Create binary column based on whether acyl_pka_aHs has a value
    df_acyl['acyl_pka_aHs_x_has_acidic_H'] = df_acyl['acyl_pka_aHs'].apply(lambda x: 0 if pd.isna(x) else x)

    df = pd.concat([df_amine, df_acyl], axis=1)
    df = df[selected_feats]
    
    return df

#### All features      
def get_electronic_features(
        xyz_file:str,
        main_atoms:int = None
        ):
    ''''
    Collects electronic global and atomistic feaures at the atom indices given; 1-based index.
    '''
    # Read XYZ file and set up XTB calculation
    file = xyz_file.split('.')[0]
    xyz_file_name = f'{file}.xyz'
    
    elements, coordinates = read_xyz(xyz_file_name)
    xtb = XTB(elements, coordinates)

    # Calculate molecular properties
    IP = xtb.get_ip(corrected=True)
    EA = xtb.get_ea()
    HOMO = xtb.get_homo()
    LUMO = xtb.get_lumo()
    global_E = xtb.get_global_descriptor("electrophilicity", corrected=True)
    global_N = xtb.get_global_descriptor("nucleophilicity", corrected=True)
    hardness = IP - EA
    
    if main_atoms:
        Charges = xtb.get_charges()
        fukui_E = xtb.get_fukui("electrophilicity")
        fukui_N = xtb.get_fukui("nucleophilicity")
    else:
        Charges = fukui_E = fukui_N = None

    return (IP, EA, HOMO, LUMO, global_E, global_N, Charges, fukui_E, fukui_N, hardness)

def get_steric_features(
        xyz_file: str,
        main_atom: int = None,
        secondary_atoms: int = None
    ):
    ''''
    Collects steric feaures: sterimol, buried volume, surface area, and volume at the atom indices given.
    Input: XYZ file, and atom indices: main_atoms and secondary_atoms; 1-based index.
    '''
    file = xyz_file.split('.')[0]
    xyz_file_name = f'{file}.xyz'
    elements, coordinates = read_xyz(xyz_file_name)

    # Calculate molecular descriptors
    bv = BuriedVolume(elements, coordinates, main_atom, radius=3.5, include_hs=False)
    sasa = SASA(elements, coordinates)
    pyr = Pyramidalization(coordinates, main_atom)
    disp = Dispersion(elements, coordinates)
    
    # Calculate sterimol and buried volume for secondary atoms if provided
    L_, B_1, B_5, BV = [], [], [], []
    if secondary_atoms:
        for atom in secondary_atoms: 
            sterimol = Sterimol(elements, coordinates, main_atom, atom) 
            L_.append(sterimol.L_value)
            B_1.append(sterimol.B_1_value)
            B_5.append(sterimol.B_5_value)
            
        for atom in [main_atom] + secondary_atoms:
            bv_atom = BuriedVolume(elements, coordinates, atom)
            BV.append(bv_atom.buried_volume)

    SA = sasa.area
    SV = sasa.volume
    PYR = pyr.P
    DISP_mol = disp.p_int
    DISP_atom = disp.atom_p_int[main_atom]

    return (L_, B_1, B_5, BV, SA, SV, PYR, DISP_mol, DISP_atom)

def get_acyl_pka_feature(smiles: str, name: str):
    
    #### get alpha carbon indices
    alpha_carbon_indices, main_index, secondary_indices, num_aHs, acyl_type = get_alpha_carbon_indices(smiles)
    alpha_carbon_indices = int(alpha_carbon_indices.split(",")[0])
    main_index = int(main_index.split(",")[0])
    secondary_indices = [int(idx) for idx in secondary_indices.split(",")]
    
    molecules = {}
    molecules[name] = {}  

    if num_aHs > 0:
        #### run pka calculation with Rowan
        mol = stjames.Molecule.from_smiles(smiles)  
        result = rowan.submit_pka_workflow(
            initial_molecule=mol,
            deprotonate_elements=[6], # 6 is the carbon element
            mode="careful",  # options: "careful", "rapid", or "careless"
            name=name
        )
        result.wait_for_result()
        result.fetch_latest(in_place=True)
        
        #### parse pka value
        all_pka_values, all_acidic_idx = [], []
        for a in result.data["conjugate_bases"]:
            pka_value = a["pka"]
            atom_idx = a["atom_index"]
            if atom_idx == alpha_carbon_indices:
                all_pka_values.append(float(pka_value))
                all_acidic_idx.append(int(atom_idx))
    else:
        print(f"No alpha hydrogens found for {name}")
        all_pka_values = None
        all_acidic_idx = alpha_carbon_indices

    molecules[name]["idx_acidic"] = all_acidic_idx
    molecules[name]["pka_aHs"] = all_pka_values
    molecules[name]["num_aHs"] = num_aHs
    molecules[name]['main_idx'] = main_index
    molecules[name]["secondary_idx"] = secondary_indices
    molecules[name]["class"] = acyl_type
    
    #### convert smile to xyz
    xyz_file = convert_smile_to_xyz(smiles, name)

    return molecules, xyz_file

def get_amine_pka_feature(smiles: str, name: str):
    '''Collect pka options: basic, acidic'''
    
    calc_dir = f"files_sdf/{name}"
    sdf_file = f"{calc_dir}/data/output/{name}.sdf"
    if os.path.exists(sdf_file):
        supplier = Chem.SDMolSupplier(sdf_file)
        # print(f"pka value already calculated for {name}: {len(supplier)} pka values")
        
    #### run pka calculation with qupKake
    else:
        with changed_dir(name):
            qupkake_command = f"qupkake smiles '{smiles}' --name '{name}' --output {name}"
            subprocess.run(qupkake_command, shell=True)

            if os.path.exists(f"data/output/{name}.sdf"):
                supplier = Chem.SDMolSupplier(f"data/output/{name}.sdf")
            else:
                print(f"Warning: Failed to generate {name}.sdf")
                return {}
    
    #### convert sdf to xyz
    xyz_file = convert_sdf_to_xyz(sdf_file, output_dir='files_xyz')
    
    #### get nitrogen indices; 0-based index
    nitrogen_indices, secondary_indices, amine_type = get_nitrogen_indices(smiles)
    nitrogen_indices = int(nitrogen_indices.split(",")[0])
    secondary_indices = [int(idx) for idx in secondary_indices.split(",")]

    molecules = {}
    for mol in supplier:
        if mol is not None:
            mol_name = mol.GetProp('_Name')
            pka_type = mol.GetProp('pka_type') if mol.HasProp('pka_type') else None
            pka_value = mol.GetProp('pka') if mol.HasProp('pka') else None
            idx_value = int(mol.GetProp('idx')) if mol.HasProp('idx') else None
            
            #### only parsing basic pka value for nitrogen indexed
            if pka_type == "basic" and idx_value == nitrogen_indices:
                if mol_name in molecules:
                    molecules[mol_name].update({
                        f'main_idx': idx_value,
                        f'secondary_idx': secondary_indices,
                        f'pka_{pka_type}': pka_value,
                        f'class': amine_type
                    })
                #### can parse for acidic pka 
                else:
                    molecules[mol_name] = {
                        f'main_idx': idx_value,
                        f'secondary_idx': secondary_indices,
                        f'pka_{pka_type}': pka_value,
                        f'class': amine_type
                    }
                    
    return molecules, xyz_file

def collect_descriptors(
    smiles: str,
    name: str,
    substrate_type: str = "acyl",
    verbose: bool = False
    ):
    
    #### Check if smiles already computed
    smiles_cannonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    
    results = check_if_computed(smiles_cannonical, name, substrate_type, verbose)
    if results:
        return pd.DataFrame(results, index=[0])

    #### Generate xyz file and get pka values
    if verbose:
        print(f"Generating xyz file and getting pka values for {name} with substrate type {substrate_type}")
    if substrate_type == "acyl":
        pka_results, xyz_file = get_acyl_pka_feature(smiles, name)
        main_atoms = pka_results[name]['main_idx']
        secondary_atoms = pka_results[name]['secondary_idx']
        print(f"Acyl has main atoms: {main_atoms} and secondary atoms: {secondary_atoms}")
        results.update({
        'name': name,
        'acyl_pka_aHs': pka_results[name]['pka_acidic'],
        'num_Hs': pka_results[name]['num_aHs'],
        'class': pka_results[name]['class']
    })
        
    elif substrate_type == "amine":
        pka_results, xyz_file = get_amine_pka_feature(smiles, name)
        main_atoms = pka_results[name]['main_idx']
        secondary_atoms = pka_results[name]['secondary_idx']
        print(f"Amine has main atoms: {main_atoms} and secondary atoms: {secondary_atoms}")
        results.update({
        'name': name,
        'pka_basic': pka_results[name]['pka_basic'],
        'class': pka_results[name]['class']
    })   
    
    #### Get electronic descriptors
    if verbose:
        print("Calculating Electronic Features")
    electronic_results = get_electronic_features(xyz_file, main_atoms)
    results.update({
        'IP': electronic_results[0],
        'EA': electronic_results[1],
        'HOMO': electronic_results[2],
        'LUMO': electronic_results[3],
        'global_E': electronic_results[4],
        'global_N': electronic_results[5],
        'hardness': electronic_results[9],
    })

    #### Get steric descriptors
    if verbose:
        print("Calculating Steric Features")
    steric_results = get_steric_features(xyz_file, main_atoms, secondary_atoms)
    
    main_atoms = [main_atoms]
    results.update({
        # 'BV': steric_results[3],
        **{f'BV_main_{i+1}': steric_results[3][i] for i in range(len(main_atoms))},
        'SA': steric_results[4],
        'SV': steric_results[5],
        'PYR': steric_results[6],
        'DISP_mol': steric_results[7],
        'DISP_atom': steric_results[8],
        **{f'Charges_main_atom': electronic_results[6][i] for i in main_atoms},
        **{f'fukui_E_main_atom': electronic_results[7][i] for i in main_atoms},
        **{f'fukui_N_main_atom': electronic_results[8][i] for i in main_atoms}
    })
    
    if secondary_atoms:
        results.update({
            **{f'BV_secondary_{i+1}': steric_results[3][i+len(main_atoms)] for i in range(len(secondary_atoms)) if secondary_atoms},
            **{f'L_main_atom_{i+1}': steric_results[0][i] for i in range(len(secondary_atoms))},
            **{f'B1_main_atom_{i+1}': steric_results[1][i] for i in range(len(secondary_atoms))},
            **{f'B5_main_atom_{i+1}': steric_results[2][i] for i in range(len(secondary_atoms))},
            **{f'Charges_secondary_{i+1}': electronic_results[6][atom] for i, atom in enumerate(secondary_atoms)},
            **{f'fukui_E_secondary_{i+1}': electronic_results[7][atom] for i, atom in enumerate(secondary_atoms)},
            **{f'fukui_N_secondary_{i+1}': electronic_results[8][atom] for i, atom in enumerate(secondary_atoms)},
            'BV_secondary_avg': np.mean([steric_results[3][i+len(main_atoms)] for i in range(len(secondary_atoms))]) if secondary_atoms else None,
            'L_avg': np.mean([steric_results[0][i] for i in range(len(secondary_atoms))]) if secondary_atoms else None,
            'B1_avg': np.mean([steric_results[1][i] for i in range(len(secondary_atoms))]) if secondary_atoms else None,
            'B5_avg': np.mean([steric_results[2][i] for i in range(len(secondary_atoms))]) if secondary_atoms else None,
            'Charges_secondary_avg': np.mean([electronic_results[6][atom] for i, atom in enumerate(secondary_atoms)]) if secondary_atoms else None,
            'fukui_E_secondary_avg': np.mean([electronic_results[7][atom] for i, atom in enumerate(secondary_atoms)]) if secondary_atoms else None,
            'fukui_N_secondary_avg': np.mean([electronic_results[8][atom] for i, atom in enumerate(secondary_atoms)]) if secondary_atoms else None,
        })
    else:
        print("Warning: No secondary atoms provided, skipping steric feature calculations.")
        
    #### Add substrate type as prefix to all keys
    results = {f'{substrate_type}_{key}': value for key, value in results.items()}
    final_results = {smiles_cannonical: results}
    
    #### save results in dictionary with all same substrate type
    if verbose:
        print(f"Saving results for {name} with substrate type {substrate_type}")
    save_final_results(final_results, substrate_type)

    return pd.DataFrame(results, index=[0])


#### main workflow
def main(args):

    if args.amine_smiles and args.amine_name:
        df_amine = collect_descriptors(smiles=args.amine_smiles, name=args.amine_name, substrate_type="amine", verbose=args.verbose)
    else:
        raise ValueError("Amine smiles and name are required")
    
    if args.acyl_smiles and args.acyl_name:
        df_acyl = collect_descriptors(smiles=args.acyl_smiles, name=args.acyl_name, substrate_type="acyl", verbose=args.verbose)
    else:
        raise ValueError("Acyl smiles and name are required")
    
    #### process additional features
    df_final = process_additional_features(df_amine, df_acyl)
    
    #### save dataframe to csv
    #### REMOVE THIS STEP WHEN APPLYING MODEL PREDICTION
    df_final.to_csv(f"results/combined_{args.amine_name}_{args.acyl_name}_features.csv", index=False)
    #### 

    return df_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate features for amide coupling")
    parser.add_argument("--amine_smiles", type=str, required=True, help="SMILES string of the amine substrate")
    parser.add_argument("--amine_name", type=str, required=True, help="Name of the amine substrate")
    parser.add_argument("--acyl_smiles", type=str, required=True, help="SMILES string of the acyl chloride substrate")
    parser.add_argument("--acyl_name", type=str, required=True, help="Name of the acyl chloride substrate")
    parser.add_argument("--verbose", action="store_true", default=False, help="Verbose output")
    args = parser.parse_args()
    main(args)
