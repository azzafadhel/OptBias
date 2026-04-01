import os
import pickle
import numpy as np
import design_bench
from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset
from design_bench.datasets.continuous.superconductor_dataset import SuperconductorDataset
from design_bench.datasets.continuous.hopper_controller_dataset import HopperControllerDataset
from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
from design_bench.datasets.discrete.tf_bind_10_dataset import TFBind10Dataset

# Dataset mappings
NAME_TO_FULL_DATASET = {
    'AntMorphology-Exact-v0': AntMorphologyDataset,
    'DKittyMorphology-Exact-v0': DKittyMorphologyDataset,
    'TFBind8-Exact-v0': TFBind8Dataset,
    'TFBind10-Exact-v0': TFBind10Dataset,
    'Superconductor-RandomForest-v0': SuperconductorDataset,
    'HopperController-Exact-v0': HopperControllerDataset,
}

TASK_ABBREVIATIONS = {
    'ant': 'AntMorphology-Exact-v0',
    'dkitty': 'DKittyMorphology-Exact-v0',
    'superconductor': 'Superconductor-RandomForest-v0',
    'hopper': 'HopperController-Exact-v0',
    'tf-bind-8': 'TFBind8-Exact-v0',
    'tf-bind-10': 'TFBind10-Exact-v0',

}

continuous_tasks = ['ant', 'dkitty', 'superconductor', 'hopper']
discrete_tasks = ['tf-bind-8', 'tf-bind-10']

save_dir = "./oracles"
os.makedirs(save_dir, exist_ok=True)

x_all_continuous = {}
y_all_continuous = {}
x_all_discrete = {}
y_all_discrete = {}

oracle_x_continuous = {}
oracle_y_continuous = {}
oracle_x_discrete = {}
oracle_y_discrete = {}

def save_pickle(obj, filename):
    with open(os.path.join(save_dir, filename), "wb") as f:
        pickle.dump(obj, f)

def print_stats(name, x, y):
    x_np = x if isinstance(x, np.ndarray) else x.numpy()
    y_np = y if isinstance(y, np.ndarray) else y.numpy()
    #print(f"[{name}] x shape: {x_np.shape}, x.min: {x_np.min():.4f}, x.max: {x_np.max():.4f}")
    print(f"[{name}] y shape: {y_np.shape}, y.min: {y_np.min():.4f}, y.max: {y_np.max():.4f}")

# Offline data (x_all_*, y_all_*)
print("\n--- OFFLINE DATASETS ---")
for task_key, dataset_name in TASK_ABBREVIATIONS.items():
    if task_key == "tf-bind-10":
        task = design_bench.make(dataset_name, dataset_kwargs={"max_samples": 10000})
        X = task.to_logits(task.x).reshape(task.x.shape[0], -1)
    else:
        task = design_bench.make(dataset_name)
        X = task.x
    Y = task.y
    
    
    if task_key == "tf-bind-8":
        X = task.to_logits(task.x).reshape(task.x.shape[0], -1)

    print_stats(f"offline-{task_key}", X, Y)

    if task_key in continuous_tasks:
        oracle_x_continuous[task_key] = X
        oracle_y_continuous[task_key] = Y
    else:
        oracle_x_discrete[task_key] = X
        oracle_y_discrete[task_key] = Y

# Oracle data (entire dataset)
print("\n--- ORACLE DATASETS ---")
for task_key, dataset_name in TASK_ABBREVIATIONS.items():
    dataset_class = NAME_TO_FULL_DATASET[dataset_name]
    dataset = dataset_class()
    X_full, Y_full = dataset.x, dataset.y

    print_stats(f"oracle-{task_key}", X_full, Y_full)

    if task_key in continuous_tasks:
        x_all_continuous[task_key] = X_full
        y_all_continuous[task_key] = Y_full
    else:
        x_all_discrete[task_key] = X_full
        y_all_discrete[task_key] = Y_full

# Save pickles
save_pickle(x_all_continuous, "x_all_continuous.pkl")
save_pickle(y_all_continuous, "y_all_continuous.pkl")
save_pickle(x_all_discrete, "x_all_discrete.pkl")
save_pickle(y_all_discrete, "y_all_discrete.pkl")

save_pickle(oracle_x_continuous, "oracle_x_continuous.pkl")
save_pickle(oracle_y_continuous, "oracle_y_continuous.pkl")
save_pickle(oracle_x_discrete, "oracle_x_discrete.pkl")
save_pickle(oracle_y_discrete, "oracle_y_discrete.pkl")
