import os
import torch
import numpy as np
from pathlib import Path

data_dir = Path(os.environ["DATA_DIR_EXP"])
out_file = data_dir / 'value90_value_dict_new.npz'

train_max_value_dict = dict()
val_max_value_dict = dict()
test_max_value_dict = dict()

for dict_file, mode in zip([train_max_value_dict, val_max_value_dict, test_max_value_dict], ['training', 'validation', 'test']):
    file_paths  = [x for x in Path(data_dir / mode).iterdir() if x.is_file()]
    for file_path in file_paths:
        # load nufft files, average along time, and save 90th percentile
        image_data = np.load(file_path)
        rnufft = torch.from_numpy(image_data['rnufft']['real'] + 1j * image_data['rnufft']['imag']).permute(2, 1, 0)
        rnufft_avg = torch.abs(torch.mean(rnufft, dim=-1))
        name = Path(file_path.name).stem     
        dict_file[name] = np.percentile(rnufft_avg, 90)
        
np.savez(
    out_file,
    train_max_value_dict=train_max_value_dict,
    val_max_value_dict=val_max_value_dict,
    test_max_value_dict=test_max_value_dict)