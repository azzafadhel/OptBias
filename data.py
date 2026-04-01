import json
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import itertools
from gpytorch.kernels import (
    RBFKernel, MaternKernel, RQKernel, PeriodicKernel,
    CosineKernel, PolynomialKernel
)

# CHANGED: we no longer import sampling_data_from_GP from util; keep load_offline_oracle_data
# from util import sampling_data_from_GP, load_offline_oracle_data
from util import load_offline_oracle_data  # CHANGED
import torch
from pathlib import Path
from datetime import datetime
import pickle



def save_function_bank(function_bank, task_name: str, seed: int, outdir: str = "./function_banks"):
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"{task_name}_seed{seed}_function_bank.pkl")
    with open(fname, "wb") as f:
        pickle.dump(function_bank, f)
    print(f"Saved function bank to {fname}")

def load_function_bank(task_name: str, seed: int, outdir: str = "./function_banks"):
    fname = os.path.join(outdir, f"{task_name}_seed{seed}_function_bank.pkl")
    with open(fname, "rb") as f:
        function_bank = pickle.load(f)
    print(f"Loaded function bank from {fname}")
    return function_bank

def sampling_data_from_GP(
    x_offline: torch.Tensor, 
    device: torch.device,
    GP_Model,
    *,
    num_gradient_steps: int,
    num_functions: int,
    num_points: int,
    learning_rate: float,
    function_bank,
    index_bank,
):
    """
    Deterministic version: uses banks for hypers + indices.
    """
    # Cache original GP hypers
    orig_ls = GP_Model.kernel.lengthscale
    orig_var = GP_Model.variance

    D = x_offline.shape[1]
    lr_vec = torch.cat([
        -learning_rate * torch.ones(num_points, D, device=device),
         learning_rate * torch.ones(num_points, D, device=device)
    ], dim=0)

    out = {}
    for f in range(num_functions):
        new_ls, new_var = function_bank[f]
        GP_Model.set_hyper(float(new_ls), float(new_var))

        idx = index_bank[f].to(device)
        low_x  = x_offline[idx].clone().detach().requires_grad_(True)
        high_x = x_offline[idx].clone().detach().requires_grad_(True)
        joint_x = torch.cat([low_x, high_x], dim=0)

        path_x, path_y = [joint_x.clone().detach()], [GP_Model.mean_posterior(joint_x).detach()]
        for _ in range(num_gradient_steps):
            mu = GP_Model.mean_posterior(joint_x)
            grad = torch.autograd.grad(mu.sum(), joint_x)[0]
            joint_x = joint_x + lr_vec * grad
            path_x.append(joint_x.clone().detach())
            path_y.append(GP_Model.mean_posterior(joint_x).detach())

        path_x, path_y = torch.stack(path_x), torch.stack(path_y)
        inc_x_list, inc_y_list = [], []
        for i in range(num_points):
            inc_x = torch.cat([path_x[:, i, :].flip(0), path_x[1:, i+num_points, :]], dim=0)
            inc_y = torch.cat([path_y[:, i].flip(0), path_y[1:, i+num_points]], dim=0)
            inc_x_list.append(inc_x)
            inc_y_list.append(inc_y)

        out[f] = {
            "increasing_inputs":  torch.stack(inc_x_list, dim=1),
            "increasing_outputs": torch.stack(inc_y_list, dim=1),
            "all_steps_raw": {"inputs": path_x, "outputs": path_y},
            "hyperparams": {"lengthscale": float(new_ls), "variance": float(new_var)},
            "indices": idx.cpu(),
        }

    GP_Model.set_hyper(orig_ls, orig_var)
    return out




kernel_dict = {
    'rbf': RBFKernel,
    'matern': MaternKernel,
    'rq': RQKernel,
    'period': PeriodicKernel,
    'cosine': CosineKernel,
    'poly': PolynomialKernel
}

class GPClass: 
    def __init__(self,device, x_train,y_train, lengthscale, variance, noise, mean_prior, kernel):
        
        self.device = device 
        self.x_train = x_train
        self.y_train = y_train
        self.kernel = kernel_dict[kernel]().to(device)
        self.noise = noise
        self.variance = variance
        self.mean_prior = mean_prior
        self.kernel.lengthscale = lengthscale
        
    def set_hyper(self, lengthscale, variance): 
        
        self.variance = variance 
        self.kernel.lengthscale = lengthscale
        if hasattr(self, 'coef'):
            del self.coef
        
        with torch.no_grad():
       
            K_train_train = self.variance*self.kernel.forward(self.x_train, self.x_train)
            K_train_train.diagonal().add_(self.noise)  
            L = torch.linalg.cholesky(K_train_train)
            b = (self.y_train - self.mean_prior)
            self.coef = torch.cholesky_solve(b, L).squeeze(-1).detach()
    

    
    def mean_posterior(self, x_test): 
        # Posterior mean
        K_train_test = self.variance * self.kernel.forward(self.x_train, x_test)
        mu_star = self.mean_prior + torch.matmul(K_train_test.T, self.coef)


        return mu_star


def restructure_synthetic_data(original_synthetic,
                               train_frac=0.7, val_frac=0.15, test_frac=0.15,
                               seed=0):
    """
    Restructure synthetic trajectory data (already in per-function flat lists)
    into train/val/test splits.

    Expected input structure:
      original_synthetic[func_name] = {
         'low':  list of (x, y) tuples,
         'high': list of (x, y) tuples
      }
    """
    rng = np.random.RandomState(seed)
    new_synthetic = {}

    for func_name, lh in original_synthetic.items():
        low_list  = lh['low']   # list of (x, y)
        high_list = lh['high']  # list of (x, y)

        # shuffle independently to avoid coupling
        idx_low  = rng.permutation(len(low_list))
        idx_high = rng.permutation(len(high_list))

        def split_list(data_list, idx):
            n = len(data_list)
            n_train = int(train_frac * n)
            n_val   = int(val_frac   * n)
            train = [data_list[i] for i in idx[:n_train]]
            val   = [data_list[i] for i in idx[n_train:n_train+n_val]]
            test  = [data_list[i] for i in idx[n_train+n_val:]]
            return train, val, test

        low_train,  low_val,  low_test  = split_list(low_list,  idx_low)
        high_train, high_val, high_test = split_list(high_list, idx_high)

        new_synthetic[func_name] = {
            'train': {'low': low_train,  'high': high_train},
            'val':   {'low': low_val,    'high': high_val},
            'test':  {'low': low_test,   'high': high_test},
        }

    return new_synthetic



class FewShotGPDataProvider(Dataset):
    """
    Episodes drawn from each synthetic function's low/high splits.
    Support comes from low-value samples, query from high-value samples.
    """
    def __init__(self, args, device):
        self.device = device
        if hasattr(args, 'script_dir') and hasattr(args, 'task'):
            args.x_best, args.y_best, args.x_train, args.y_train, mean_x, std_x, mean_y, std_y = load_offline_oracle_data(
                script_dir=args.script_dir,
                task=args.task,
                topk=args.topk,
                device=device,
                sampling_strategy=args.sampling_strategy,
                seedrng = 1  
            )
        self.x_train = args.x_train
        self.y_train = args.y_train
        self.x_best  = args.x_best
        self.y_best  = args.y_best
        self.mean_x = mean_x
        self.std_x  = std_x
        self.mean_y = mean_y
        self.std_y  = std_y
        
        self.num_functions      = args.num_functions
        self.num_points         = args.num_points
        self.num_ctx            = args.num_ctx
        self.num_tar            = args.num_tar
        self.num_gradient_steps = args.num_gradient_steps
        self.learning_rate      = args.learning_rate
        self.delta_lengthscale  = args.delta_lengthscale
        self.delta_variance     = args.delta_variance
        self.threshold_diff     = args.threshold_diff
        self.init_seed          = args.seed  
        
     
        selected_x = self.x_best
        selected_y = self.y_best
            
        self.gp_model = GPClass(
            device=device,
            x_train=selected_x,
            y_train=selected_y,
            lengthscale=torch.tensor(args.init_lengthscale, device=device),
            variance=torch.tensor(args.init_variance,   device=device),
            noise=torch.tensor(args.noise,              device=device),
            mean_prior=torch.tensor(0.0,                device=device),
            kernel=args.kernel
        )






        function_bank_dir="/function_banks"
        index_bank_dir = "/index_banks"

        # -------------------------------
        # Load saved function + index banks
        # -------------------------------
        fbank_path = os.path.join(function_bank_dir, f"{args.task}_seed{args.seed}_function_bank.pkl")
        ibank_path = os.path.join(index_bank_dir, f"{args.task}_seed{args.seed}_index_bank.pt")

        if not os.path.exists(fbank_path):
            raise FileNotFoundError(f"No function bank found at {fbank_path}. Generate it first.")
        if not os.path.exists(ibank_path):
            raise FileNotFoundError(f"No index bank found at {ibank_path}. Generate it first.")

        with open(fbank_path, "rb") as f:
            function_bank = pickle.load(f)
  
        index_bank = torch.load(ibank_path, map_location="cpu")

    
        index_bank = index_bank.to(device)



        if self.num_functions > len(function_bank):
            raise ValueError(f"Requested {self.num_functions} funcs but bank only has {len(function_bank)}")
        if self.num_points != index_bank.shape[1]:
            raise ValueError(f"Config num_points={self.num_points} but index bank has {index_bank.shape[1]}")

    
        function_bank = function_bank[:self.num_functions]
        index_bank = index_bank[:self.num_functions]

    
        self.raw_out = sampling_data_from_GP(
            x_offline=self.x_train.to(device),
            device=device,
            GP_Model=self.gp_model,
            num_functions=self.num_functions,
            num_gradient_steps=self.num_gradient_steps,
            num_points=self.num_points,
            learning_rate=self.learning_rate,
            function_bank=function_bank,
            index_bank=index_bank,
        )

        


        original_synth_flat = {}  # NEW
        for f, blob in self.raw_out.items():
            X_inc = blob["increasing_inputs"]    # [L, N, D]
            y_inc = blob["increasing_outputs"]   # [L, N]
            L, N, D = X_inc.shape
            mid = (L - 1) // 2  # == T-1

            support_x = X_inc[:mid]    # [T-1, N, D]
            support_y = y_inc[:mid]    # [T-1, N]
            target_x  = X_inc[mid:]    # [T,   N, D]
            target_y  = y_inc[mid:]    # [T,   N]

            # Flatten (time, seed) into a single list of tuples
            low_list, high_list = [], []
            # support leg
            sup_x_flat = support_x.reshape(-1, D)   # [(T-1)*N, D]
            sup_y_flat = support_y.reshape(-1)      # [(T-1)*N]
            for i in range(sup_x_flat.size(0)):
                low_list.append((sup_x_flat[i].detach(), sup_y_flat[i].detach()))
            # target leg
            tar_x_flat = target_x.reshape(-1, D)    # [T*N, D]
            tar_y_flat = target_y.reshape(-1)       # [T*N]
            for i in range(tar_x_flat.size(0)):
                high_list.append((tar_x_flat[i].detach(), tar_y_flat[i].detach()))

            original_synth_flat[f] = {'low': low_list, 'high': high_list}
        # ===================== end NEW conversion =====================

        self.synthetic = restructure_synthetic_data(
            original_synth_flat,
            train_frac=args.train_val_test_split[0],
            val_frac=args.train_val_test_split[1],
            test_frac=args.train_val_test_split[2],
            seed=self.init_seed
        )


        all_funcs = list(self.synthetic.keys())
        self._func_splits = {
            'train': all_funcs,
            'val':   all_funcs,
            'test':  all_funcs
        }

        self.episodes_per_function = {}
        for split, funcs in self._func_splits.items():
            if not funcs:
                self.episodes_per_function[split] = 0
                continue

            eps_counts = []
            for func in funcs:
                low_list  = self.synthetic[func][split]['low']
                high_list = self.synthetic[func][split]['high']
                eps_low   = len(low_list)  // self.num_ctx
                eps_high  = len(high_list) // self.num_tar
                eps_counts.append(min(eps_low, eps_high))
            self.episodes_per_function[split] = min(eps_counts) if eps_counts else 0

        # Set seeds per split
        self.seed = {split: self.init_seed for split in self._func_splits}
        self.current_set = 'train'


    def switch_set(self, set_name: str):
        assert set_name in self._func_splits, f"Unknown split '{set_name}'"
        self.current_set = set_name

    def __len__(self):
        funcs = self._func_splits[self.current_set]
        eps = self.episodes_per_function[self.current_set]
        return len(funcs) * eps

    def __getitem__(self, idx: int):
        funcs = self._func_splits[self.current_set]
        eps = self.episodes_per_function[self.current_set]
        func_idx = idx // eps
        ep_idx   = idx % eps
        func_name = funcs[func_idx]
        return (*self._get_samples(func_name, ep_idx), ep_idx)

    def _get_samples(self, func_name: str, ep_idx: int):
        split = self.current_set

        # CHANGED: now synthetic[func][split] has flat 'low' and 'high' lists
        low  = self.synthetic[func_name][split]['low']   # list[(x, y)]
        high = self.synthetic[func_name][split]['high']  # list[(x, y)]

        sl, el = ep_idx * self.num_ctx, ep_idx * self.num_ctx + self.num_ctx
        sq, eq = ep_idx * self.num_tar, ep_idx * self.num_tar + self.num_tar

        # ←–– GUARD
        if el > len(low) or eq > len(high):
            # no more full episodes left for this function
            raise StopIteration

        support = low[sl:el]
        query   = high[sq:eq]

        sx = torch.stack([x.view(-1) for x, _ in support]).to(self.device)
        sy = torch.tensor([y for _, y in support], dtype=torch.float, device=self.device)
        qx = torch.stack([x.view(-1) for x, _ in query]).to(self.device)
        qy = torch.tensor([y for _, y in query],   dtype=torch.float, device=self.device)
        return sx, qx, sy, qy


import itertools
import torch

class MetaLearningSystemDataLoader:
    """
    Builds each batch (train/val/test) by cycling through functions and pulling
    one episode per function until batch_size is reached, never repeating points,
    and then stops once all episodes are exhausted.
    """

    def __init__(self, args, device, current_iter: int = 0):
        self.num_gpus         = args.num_of_gpus
        self.samples_per_iter = args.samples_per_iter
        self.batch_size       = args.batch_size      # episodes per batch
        self.num_workers      = args.num_dataprovider_workers
        self.device           = device

        # our FewShot provider knows about .synthetic and .episodes_per_function
        self.dataset = FewShotGPDataProvider(args, device)

        # episode pointers per split/function
        self.func_splits = self.dataset._func_splits
        self.next_ep_idx = {
            split: {f: 0 for f in self.func_splits[split]}
            for split in self.func_splits
        }

        self._offset_iters(current_iter)

    def _offset_iters(self, current_iter: int):
        world_batch = self.num_gpus * self.batch_size * self.samples_per_iter
        # you can keep track of total if you need it:
        self.total_train_iters_produced = current_iter * world_batch
        
    def reset_episode_indices(self, split: str):
        for f in self.func_splits[split]:
            self.next_ep_idx[split][f] = 0


    def _get_interleaved_batches(self, split: str):

        self.dataset.switch_set(split)

        funcs = self.func_splits[split]
        per_func_eps = self.dataset.episodes_per_function[split]
        batch_size = self.batch_size
        device = self.device


        chunks = [funcs[i:i + batch_size] for i in range(0, len(funcs), batch_size)]

        for ep_idx in range(int(per_func_eps)):  
            for chunk in chunks: 
                support_eps, query_eps = [], []

                for f in chunk:
                    if self.next_ep_idx[split][f] > ep_idx:
                        continue 

                    try:
                        sx, qx, sy, qy = self.dataset._get_samples(f, ep_idx)
                    except StopIteration:
                        continue  

                    self.next_ep_idx[split][f] += 1
                    support_eps.append((sx, sy))
                    query_eps.append((qx, qy))

                if len(support_eps) != batch_size:
                    continue  

                sx_b = torch.stack([s for s, _ in support_eps], dim=0).to(device)
                sy_b = torch.stack([y for _, y in support_eps], dim=0).to(device)
                qx_b = torch.stack([q for q, _ in query_eps], dim=0).to(device)
                qy_b = torch.stack([y for _, y in query_eps], dim=0).to(device)

                yield sx_b, qx_b, sy_b, qy_b

        self.reset_episode_indices(split)

    def get_train_batches(self):
        return self._get_interleaved_batches('train')

    def get_val_batches(self):
        return self._get_interleaved_batches('val')

    def get_test_batches(self):
        return self._get_interleaved_batches('test')
