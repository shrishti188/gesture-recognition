

import h5py

model_path = "D:/Sign-Language-detection/Model/keras_model.h5"

with h5py.File(model_path, 'r') as f:
    def print_hdf5_group(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name} - Shape: {obj.shape}, Dtype: {obj.dtype}")

    f.visititems(print_hdf5_group)

