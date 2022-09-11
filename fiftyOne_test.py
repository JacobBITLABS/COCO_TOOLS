import fiftyone as fo

name = "d_dataset_scaled_val1"

data_path = "scaled_val"
labels_path = "scaled_val.json"

# Import dataset by explicitly providing paths to the source media and labels
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=data_path,
    labels_path=labels_path,
    name=name,
)

dataset = fo.load_dataset(name)

session = fo.launch_app(dataset, desktop=True)
session.wait() 