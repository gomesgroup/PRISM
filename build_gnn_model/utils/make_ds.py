from ngdataset import AMIDEDataset
import os
import numpy as np
import torch 
import json
from ase.data import atomic_numbers

'''
Phase 1: Processes acid, amine, and intermediate separately using the AIMNet2 model 
to obtain the atomic representations to build the atomic/molecular features.

Makes the train and test h5 datasets using the AIMNet2 model: model_wb_nse_0.jpt
Activate `ml_env` conda environment before running.
'''
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level from utils/ to updated/
data_dir = os.path.join(project_root, 'data')

def read_xyz(fname):
    coord, numbers = [], []
    with open(fname) as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            row = line.split()
            if len(row) == 0:
                break
            coord.append([float(row[1]), float(row[2]), float(row[3])])
            numbers.append(atomic_numbers[row[0]])
            line = f.readline()

    coord = np.array(coord).reshape(-1,3)
    numbers = np.array(numbers).reshape(-1)
    return coord, numbers

def load_splits_from_json(json_file):
    """Load train, validation, and test splits from JSON file."""
    with open(json_file, 'r') as f:
        splits_data = json.load(f)
    
    # Extract reaction names, rates, and features for each split
    def extract_rxns_and_rates(split_dict):
        rxns = []
        rates = []
        features_dict = {}
        
        for rxn_key, data in split_dict.items():
            rxns.append(rxn_key)
            
            if isinstance(data, dict):
                # New format: data is a dictionary with 'hte_lnk' and other features
                if 'hte_lnk' in data:
                    rates.append(data['hte_lnk'])
                else:
                    raise ValueError(f"Missing 'hte_lnk' key in data for reaction {rxn_key}")
                
                # Extract additional features (excluding 'hte_lnk' which is the rate)
                features = {k: v for k, v in data.items() if k != 'hte_lnk'}
                if features:
                    features_dict[rxn_key] = features
            else:
                # Old format: data is directly the rate value
                rates.append(data)
        
        return rxns, rates, features_dict
    
    train_rxns, train_rates, train_features = extract_rxns_and_rates(splits_data['train'])
    val_rxns, val_rates, val_features = extract_rxns_and_rates(splits_data['val'])
    test_rxns, test_rates, test_features = extract_rxns_and_rates(splits_data['test'])
    
    print(f"Loaded splits from {json_file}:")
    print(f"  Train: {len(train_rxns)} reactions")
    print(f"  Val: {len(val_rxns)} reactions")
    print(f"  Test: {len(test_rxns)} reactions")
    
    # Print feature information if available
    if train_features:
        print(f"  Train features: {len(train_features)} reactions with additional features")
        print(f"    Feature keys: {set().union(*[f.keys() for f in train_features.values()])}")
    if val_features:
        print(f"  Val features: {len(val_features)} reactions with additional features")
    if test_features:
        print(f"  Test features: {len(test_features)} reactions with additional features")
    
    return (train_rxns, train_rates, train_features), (val_rxns, val_rates, val_features), (test_rxns, test_rates, test_features)

if __name__ == '__main__':
    
    # Load splits from JSON file using absolute path
    # json_file = "9_hte-all-original_splits_train_val_tests_lnk.json" #### EDIT HERE
    # json_file = "10_hte-rel-original_splits_train_val_tests_lnk.json"
    json_file = "11_hte-all-corrected_splits_train_val_tests_lnk.json"
    splits_file = os.path.join(data_dir, json_file)
    (train_rxns, train_rates, train_features), (val_rxns, val_rates, val_features), (test_rxns, test_rates, test_features) = load_splits_from_json(splits_file)
    
    model_path = os.path.join(script_dir, 'aimnet2', 'model_wb_nse_0.jpt')
    model = torch.jit.load(model_path, map_location='cpu')

    def process_rxns(rxns, rates, features, output_file, model):
        """Process reactions and save to H5 file."""
        print(f"Processing {len(rxns)} reactions for {output_file}")
        _process_single_split(rxns, rates, features, output_file, model)
    
    def _process_single_split(rxns, rates, features, output_file, model):
        ds = None
        missing_ints = []
        
        for count, rxn_key in enumerate(rxns):
            # Parse reaction key format: rxn_acid_amine
            parts = rxn_key.split('_')
            if len(parts) != 3 or parts[0] != 'rxn':
                raise ValueError(f"Invalid reaction key format: {rxn_key}. Expected format: rxn_acid_amine")
            
            acid_num = parts[1]
            amine_num = parts[2]
            
            # Format numbers with zero padding to match file naming convention
            acid_file = os.path.join(data_dir, 'acyls', f'acid_{acid_num.zfill(2)}.xyz')
            amine_file = os.path.join(data_dir, 'amines', f'amine_{amine_num.zfill(2)}.xyz')
            int_file = os.path.join(data_dir, 'int1s', f'INT1_rxn_{acid_num.zfill(2)}_{amine_num.zfill(2)}.xyz')
            
            acid_coord, acid_numbers = read_xyz(acid_file)
            amine_coord, amine_numbers = read_xyz(amine_file)
            
            if not os.path.exists(int_file):
                print(f"Missing INT1 file for {rxn_key}")
                missing_ints.append(rxn_key)
                continue
                
            int_coord, int_numbers = read_xyz(int_file)
            
            # Process acyl chloride
            n = acid_coord.shape[0]

            _in = dict(coord=torch.tensor(acid_coord).reshape(1,-1,3),
                      numbers = torch.tensor(acid_numbers).reshape(1,-1),
                      charge = torch.zeros(1),
                      mult = torch.ones(1))

            _out = model(_in)

            acid_a = _out['a'].numpy().reshape(n, -1)
            acid_aim = _out['aim'].numpy().reshape(n, -1)
            acid_q = (_out['charges'] +  _out['spin_charges']).flatten()

            # Process amine
            n = amine_coord.shape[0]

            _in = dict(coord=torch.tensor(amine_coord).reshape(1,-1,3),
                      numbers = torch.tensor(amine_numbers).reshape(1,-1),
                      charge = torch.zeros(1),
                      mult = torch.ones(1))

            _out = model(_in)

            amine_a = _out['a'].numpy().reshape(n, -1)
            amine_aim = _out['aim'].numpy().reshape(n, -1)
            amine_q = (_out['charges'] +  _out['spin_charges']).flatten()
            
            
            n = int_coord.shape[0]

            _in = dict(coord=torch.tensor(int_coord).reshape(1,-1,3),
                      numbers = torch.tensor(int_numbers).reshape(1,-1),
                      charge = torch.zeros(1),
                      mult = torch.ones(1))

            _out = model(_in)

            int_a = _out['a'].numpy().reshape(n, -1)
            int_aim = _out['aim'].numpy().reshape(n, -1)
            int_q = (_out['charges'] +  _out['spin_charges']).flatten()

            # Use the reaction key as the identifier
            _id = rxn_key
            d = dict(
                    amine_a = amine_a,
                    amine_q = amine_q,
                    amine_aim = amine_aim,
                    acid_a = acid_a,
                    acid_q = acid_q,
                    acid_aim = acid_aim,
                    int_a = int_a,
                    int_q = int_q,
                    int_aim = int_aim,
                    rate = np.array([rates[count]]).reshape(1)
                    )
            
            # Add additional features if available for this reaction
            if rxn_key in features:
                for feature_name, feature_value in features[rxn_key].items():
                    # Convert feature values to numpy arrays for consistency
                    if isinstance(feature_value, (int, float)):
                        d[feature_name] = np.array([feature_value]).reshape(1)
                    else:
                        d[feature_name] = np.array([feature_value]).reshape(1)
            dd = dict()
            dd[_id] = d

            if ds is None:
                ds = AMIDEDataset(dd)
            else:
                tds = AMIDEDataset(dd)
                ds.merge(tds)
                
        #### make directory if it doesn't exist
        splits_dir = os.path.join(project_root, 'splits')
        os.makedirs(splits_dir, exist_ok=True)
        ds.save_h5(output_file)
        print(f"Saved {len(rxns)} reactions to {output_file}")
        
    # Process all three splits: train, validation, and test using absolute paths
    splits_dir = os.path.join(project_root, 'splits')
    process_rxns(train_rxns, train_rates, train_features, os.path.join(splits_dir, 'train.h5'), model)
    process_rxns(val_rxns, val_rates, val_features, os.path.join(splits_dir, 'val.h5'), model)
    process_rxns(test_rxns, test_rates, test_features, os.path.join(splits_dir, 'test.h5'), model)

