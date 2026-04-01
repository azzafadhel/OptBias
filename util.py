from tqdm import tqdm, trange
import numpy as np
import os, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy
from glob import glob
from pprint import pprint
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)


TASKS = {
    "tfbind8": "TFBind8-Exact-v0",  
    "gfp": "GFP-GP-v0",
    "utr": "UTR-ResNet-v0",
    "hopper": "HopperController-Exact-v0",  
    "rf": "Superconductor-RandomForest-v0",
    "ant": "AntMorphology-Exact-v0",  
    "dkitty": "DKittyMorphology-Exact-v0",  
}



def freeze(model):
    assert isinstance(model, nn.Module)
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(model):
    assert isinstance(model, nn.Module)
    for p in model.parameters():
        p.requires_grad = True


def init_weights(m, method="kaiming"):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        if method == "kaiming":
            torch.nn.init.kaiming_uniform_(m.weight)
        else:
            torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.00)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch_seed = seed
    np_seed = seed
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)


def save_object(obj, filename):
    with open(filename, "wb") as output:
        pickle.dump(obj, output)


def load_object(filename):
    with open(filename, "rb") as output:
        return pickle.load(output)


def check(FOLDER):
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)






    
def normalize_x(x: torch.Tensor, mean: torch.Tensor = None, std: torch.Tensor = None):
    """
    Normalize input tensor x per-dimension.
    If mean/std not provided, compute from x.
    Returns:
      - x_norm: normalized x
      - mean: per-dimension mean used
      - std: per-dimension std used (zeros replaced by 1.0)
    """
    #x = torch.tensor(x, dtype=torch.float32) 
    if mean is None or std is None:
        mean = x.mean(dim=0)
        std = x.std(dim=0)
        std[std == 0] = 1.0
    x_norm = (x - mean) / std
    return x_norm, mean, std


def normalize_y(y: torch.Tensor, mean: torch.Tensor = None, std: torch.Tensor = None):
    """
    Normalize target tensor y.
    If mean/std not provided, compute from y.
    Returns:
      - y_norm: normalized y
      - mean: scalar mean used
      - std: scalar std used (zero replaced by 1.0)
    """
    if mean is None or std is None:
        mean = y.mean()
        std = y.std()
        if std == 0:
            std = torch.tensor(1.0, device=y.device)
    y_norm = (y - mean) / std
    return y_norm, mean, std


def denormalize_x(x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """
    Denormalize input tensor using provided mean and std.
    """
    return x_norm * std + mean


def denormalize_y(y_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """
    Denormalize target tensor using provided mean and std.
    """
    return y_norm * std + mean
    








def sampling_data_from_GP(
    x_offline: torch.Tensor, 
    device: torch.device,
    GP_Model,
    num_gradient_steps: int = 100,
    num_functions: int = 128,
    num_points: int = 100,
    learning_rate: float = 0.001,
    delta_lengthscale: float = 0.1,
    delta_variance: float = 0.1,
):



    orig_ls = GP_Model.kernel.lengthscale
    orig_var = GP_Model.variance

    
    lr_vec = torch.cat([
        -learning_rate * torch.ones(num_points, x_offline.shape[1], device=device),
         learning_rate * torch.ones(num_points, x_offline.shape[1], device=device)
    ], dim=0)

    out = {}

    for f in range(num_functions):
       
        new_ls = orig_ls + delta_lengthscale * (2 * torch.rand(1, device=device) - 1)
        new_var = orig_var + delta_variance * (2 * torch.rand(1, device=device) - 1)
        if hasattr(GP_Model, "set_hyper"):
            GP_Model.set_hyper(lengthscale=new_ls, variance=new_var)
        else:
            GP_Model.kernel.lengthscale = new_ls
            GP_Model.variance = new_var


        idx = torch.randperm(x_offline.size(0), device=device)[:num_points]
        low_x  = x_offline[idx].clone().detach().requires_grad_(True)   
        high_x = x_offline[idx].clone().detach().requires_grad_(True)   
        joint_x = torch.cat([low_x, high_x], dim=0) 


        path_x = [joint_x.clone().detach()]
        with torch.no_grad():
            path_y = [GP_Model.mean_posterior(joint_x).detach()]

        for _ in range(num_gradient_steps):
            mu = GP_Model.mean_posterior(joint_x)
            grad = torch.autograd.grad(mu.sum(), joint_x, retain_graph=False, create_graph=False)[0]
            joint_x = joint_x + lr_vec * grad

            path_x.append(joint_x.clone().detach())
            with torch.no_grad():
                path_y.append(GP_Model.mean_posterior(joint_x).detach())

 
        path_x = torch.stack(path_x)  
        path_y = torch.stack(path_y)  
        T = path_x.shape[0]

    
        inc_x_list, inc_y_list = [], []
        for i in range(num_points):
            descent_x = path_x[:, i, :]                 
            ascent_x  = path_x[:, i + num_points, :]    
            descent_y = path_y[:, i]                    
            ascent_y  = path_y[:, i + num_points]       


            inc_x = torch.cat([descent_x.flip(0), ascent_x[1:]], dim=0)  
            inc_y = torch.cat([descent_y.flip(0), ascent_y[1:]], dim=0)  

            inc_x_list.append(inc_x)  
            inc_y_list.append(inc_y)  

        increasing_inputs  = torch.stack(inc_x_list, dim=1)
        increasing_outputs = torch.stack(inc_y_list, dim=1)

        out[f] = {
            "increasing_inputs":  increasing_inputs,   
            "increasing_outputs": increasing_outputs,  
            "all_steps_raw": {
                "inputs":  path_x, 
                "outputs": path_y,  
            },
        }


    if hasattr(GP_Model, "set_hyper"):
        GP_Model.set_hyper(lengthscale=orig_ls, variance=orig_var)
    else:
        GP_Model.kernel.lengthscale = orig_ls
        GP_Model.variance = orig_var

    return out




def load_offline_oracle_data(
    script_dir: str,
    task: str,
    topk: float,  
    device: torch.device,
    sampling_strategy: str,
    seedrng : int 
):
    """
   

    Returns:
      - x_best: Tensor (topk × D), normalized
      - y_best: Tensor (topk), normalized
      - x_norm: Full normalized x (N × D)
      - y_norm: Full normalized y (N, 1)
      - mean_x: Tensor (D,)
      - std_x:  Tensor (D,)
      - mean_y: float
      - std_y:  float
    """
    print("=" * 60)
    print("LOADING OFFLINE ORACLE DATA - DEBUG INFO")
    print("=" * 60)
    
    print(f"Input parameters:")
    print(f"  script_dir: {script_dir}")
    print(f"  task: {task}")
    print(f"  topk: {topk}")
    print(f"  device: {device}")
    print(f"  sampling_strategy: {sampling_strategy}")
    print(f"  seedrng: {seedrng}")
    print()

    # load raw data
    if task in ["tf-bind-8", "tf-bind-10"]:
        oxp = os.path.join(script_dir, "oracle_x_discrete.pkl")
        oyp = os.path.join(script_dir, "oracle_y_discrete.pkl")
        data_type = "discrete"
    elif task in ["rna1", "rna2", "rna3"]:
        oxp = os.path.join(script_dir, "oracle_x_rna.pkl")
        oyp = os.path.join(script_dir, "oracle_y_rna.pkl")
        data_type = "rna"
    else:
        oxp = os.path.join(script_dir, "oracle_x_continuous.pkl")
        oyp = os.path.join(script_dir, "oracle_y_continuous.pkl")
        data_type = "continuous"
    
    print(f"Data type detected: {data_type}")
    print(f"Loading from:")
    print(f"  X file: {oxp}")
    print(f"  Y file: {oyp}")
    
    try:
        with open(oxp, "rb") as f:
            Oracle_x = pickle.load(f)
        with open(oyp, "rb") as f:
            Oracle_y = pickle.load(f)
        print("Files loaded successfully!")
    except Exception as e:
        print(f"Error loading files: {e}")
        raise
    
    print(f"Oracle_x keys: {list(Oracle_x.keys()) if hasattr(Oracle_x, 'keys') else 'Not a dict'}")
    print(f"Oracle_y keys: {list(Oracle_y.keys()) if hasattr(Oracle_y, 'keys') else 'Not a dict'}")
    print()

    x_raw = torch.tensor(Oracle_x[task], dtype=torch.float32)  # (N, D)
    y_raw = torch.tensor(Oracle_y[task], dtype=torch.float32)  # (N,)
    
    print(f"Raw data loaded:")
    print(f"  x_raw shape: {x_raw.shape}")
    print(f"  y_raw shape: {y_raw.shape}")
    print(f"  x_raw stats: min={x_raw.min():.4f}, max={x_raw.max():.4f}, mean={x_raw.mean():.4f}")
    print(f"  y_raw stats: min={y_raw.min():.4f}, max={y_raw.max():.4f}, mean={y_raw.mean():.4f}")
    print()

    # normalize
    x_norm, mean_x, std_x = normalize_x(x_raw)
    y_norm, mean_y, std_y = normalize_y(y_raw)
    y_norm = y_norm.view(-1, 1)  # Ensure shape [N, 1]
    
    print(f"After normalization:")
    print(f"  x_norm shape: {x_norm.shape}")
    print(f"  y_norm shape: {y_norm.shape}")
    print(f"  mean_x shape: {mean_x.shape}, values: {mean_x[:5]}...")
    print(f"  std_x shape: {std_x.shape}, values: {std_x[:5]}...")
    print(f"  mean_y: {mean_y}, std_y: {std_y}")
    print(f"  x_norm stats: min={x_norm.min():.4f}, max={x_norm.max():.4f}, mean={x_norm.mean():.4f}")
    print(f"  y_norm stats: min={y_norm.min():.4f}, max={y_norm.max():.4f}, mean={y_norm.mean():.4f}")
    print()

    total_n = x_norm.shape[0]
    if topk > 1:
        len_eval = int(topk)
    else:
        len_eval = int(topk * total_n)
    print(f"Selection parameters:")
    print(f"  Total samples: {total_n}")
    print(f"  topk parameter: {topk}")
    print(f"  Evaluating {len_eval} samples ({len_eval/total_n*100:.2f}% of total)")
    
    if sampling_strategy == "random":
        eval_ids = torch.arange(len_eval)
        print(f"  Strategy 'rand': selecting first {len_eval} samples")
    elif sampling_strategy == "randup":
        eval_ids = torch.arange(total_n - len_eval, total_n)
        print(f"  Strategy 'randup': selecting last {len_eval} samples (indices {total_n - len_eval} to {total_n-1})")
    elif sampling_strategy == "poor":
        eval_ids = torch.argsort(y_norm.squeeze(-1))[:len_eval]
        print(f"  Strategy 'poor': selecting {len_eval} samples with lowest y values")
    elif sampling_strategy == "top":
        eval_ids = torch.argsort(y_norm.squeeze(-1), descending=True)[:len_eval]
        print(f"  Strategy 'top': selecting {len_eval} samples with highest y values")
    elif sampling_strategy == 'rand_rng':
        
        rng = np.random.default_rng(seed=seedrng)
        eval_ids = rng.choice(len(x_norm), size=len_eval, replace=False)
        print(f"  Strategy 'rand_rng': randomly selecting {len_eval} samples with seed {seedrng}")
    else:
        raise ValueError(f"Unknown sampling_strategy '{sampling_strategy}'")
    
    print(f"  Selected indices range: [{eval_ids.min().item()}, {eval_ids.max().item()}]")
    print(f"  First 10 selected indices: {eval_ids[:10].tolist()}")

    # select
    x_best = x_norm[eval_ids].view(len_eval, -1).to(device)
    y_best = y_norm[eval_ids].view(len_eval, 1).to(device)
    
    print(f"\nSelected data:")
    print(f"  x_best shape: {x_best.shape}")
    print(f"  y_best shape: {y_best.shape}")
    print(f"  x_best stats: min={x_best.min():.4f}, max={x_best.max():.4f}, mean={x_best.mean():.4f}")
    print(f"  y_best stats: min={y_best.min():.4f}, max={y_best.max():.4f}, mean={y_best.mean():.4f}")

    # move full data to device
    x_norm = x_norm.to(device)
    y_norm = y_norm.to(device)
    
    print(f"\nFinal output shapes:")
    print(f"  x_best: {x_best.shape} on {x_best.device}")
    print(f"  y_best: {y_best.shape} on {y_best.device}")
    print(f"  x_norm: {x_norm.shape} on {x_norm.device}")
    print(f"  y_norm: {y_norm.shape} on {y_norm.device}")
    print(f"  mean_x: {mean_x.shape} on {mean_x.device}")
    print(f"  std_x: {std_x.shape} on {std_x.device}")
    print("=" * 60)

    return x_best, y_best, x_norm, y_norm, mean_x.to(device), std_x.to(device), mean_y, std_y



    
    
    
    
