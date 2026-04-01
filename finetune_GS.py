import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
from util import load_offline_oracle_data, freeze, unfreeze, denormalize_x, denormalize_y, normalize_x, normalize_y, load_offline_oracle_data_discrete
from util import load_stats_from_txt
from few_shot_learning_system import MAMLFewShotRegressor
from utils.parser_utils import get_args
import design_bench
import random

import logging
logging.getLogger().setLevel(logging.ERROR)  # only show errors


import random, numpy as np, torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)
seed=SEED


task_to_min = {'TFBind8-Exact-v0': 0.0, 'TFBind10-Exact-v0': -1.8585, 'AntMorphology-Exact-v0': -386.9004, 'DKittyMorphology-Exact-v0': -880.4585, "Superconductor-RandomForest-v0": 0.0002, "HopperController-Exact-v0": 87.9349}
task_to_max = {'TFBind8-Exact-v0': 1.0, 'TFBind10-Exact-v0': 2.1287, 'AntMorphology-Exact-v0': 590.2444, 'DKittyMorphology-Exact-v0': 340.9099, "Superconductor-RandomForest-v0":185.0000, "HopperController-Exact-v0": 1361.6106}
task_to_best = {'TFBind8-Exact-v0': 0.4393, 'TFBind10-Exact-v0': 0.0053, 'AntMorphology-Exact-v0': 165.3265, 'DKittyMorphology-Exact-v0': 199.3625,  "Superconductor-RandomForest-v0":74.0000, "HopperController-Exact-v0":1361.6106}

TASKS = {
    "tf-bind-8": "TFBind8-Exact-v0",  # requires morphing-agents
    "tf-bind-10": "TFBind10-Exact-v0",
    "gfp": "GFP-GP-v0",
    "utr": "UTR-ResNet-v0",
    "hopper": "HopperController-Exact-v0",  # requires morphing-agents
    "superconductor": "Superconductor-RandomForest-v0",
    "ant": "AntMorphology-Exact-v0",  # requires morphing-agents
    "dkitty": "DKittyMorphology-Exact-v0",  # requires morphing-agents
    "rna1": "rna1",
    "rna2": "rna2",
    "rna3": "rna3",
}

# ========== Setup ==========
args, device_ = get_args()
print("args.task:", args.task)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("embed_dim =", args.embed_dim)
print("num_stages =", args.num_stages)

# Initialize model
model = MAMLFewShotRegressor(
    im_shape=(args.batch_size, args.dimension),
    device=device,
    args=args
)



task = args.task
algo = "gradient_ascent"
n_runs = 1
opt_iter = 300
n_sol = 128
lr = 0.001
init_mode = "topk"
topk_points = args.num_points
index_training=50 #######################
# ========== Load model ==========

model_path = f"path/{args.task}_OptReg_Loss_batch_size_{args.num_functiosn}_num_tar_{args.num_tar}_num_ctx_{args.num_ctx}_sampling_strategy_{args.sampling_strategy}_{args.topk}r_seed_{SEED}_num_points_{args.num_points}_num_functions_{args.num_functiosn}/saved_models/"


state = model.load_model_prime(
    model_save_dir=model_path,
    model_name="train_model",
    model_idx=index_training,
    device=device
)


model.regressor = model.regressor.to(device)

# ========== Load offline data ==========
task_x, task_y, x_norm, y_norm, mean_x, std_x, mean_y, std_y= load_offline_oracle_data(
    script_dir=args.script_dir,
    task=args.task,
    topk=args.num_points,
    device=device,
    sampling_strategy= args.sampling_strategy
    
)

support_x = task_x.clone().detach().float().to(device)
support_y = task_y.clone().detach().float().to(device)
query_x   = task_x.clone().detach().float().to(device)
query_y   = task_y.clone().detach().float().to(device)


weights = model.get_inner_loop_parameter_dict(model.regressor.named_parameters())
weights = {k.replace("module.", ""): v for k, v in weights.items()}


for i in range(20):
    for step in range(args.number_of_evaluation_steps_per_iter):
        loss, _ = model.net_forward_finetune(
            x=support_x,
            y=support_y,
            weights=weights,
            training=True,      
            backup_running_statistics=False,
            num_step=step
        )
        

        weights = model.apply_inner_loop_update(
            loss=loss,
            names_weights_copy=weights,
            use_second_order=False,
            current_step_idx=step
        )


with torch.no_grad():
    pred_y = model.net_forward_search(
        x=task_x, 
        weights=weights, 
        training=False, 
        backup_running_statistics=False, 
        num_step=0
    ).flatten().cpu()

true_y = task_y.flatten().cpu()



x = (pred_y - pred_y.mean()) / (pred_y.std() + 1e-8)
y = (true_y - true_y.mean()) / (true_y.std() + 1e-8)
corr = (x * y).mean()

print("Correlation surrogate vs oracle:", corr.item())

sgn = 1.0 if corr >= 0 else -1



with torch.no_grad():
    for name, param in model.regressor.named_parameters():
        cleaned_name = name.replace("module.", "")
        if cleaned_name in weights:
            param.data.copy_(weights[cleaned_name].squeeze(0))  # remove [1,...] shape


if args.task == "tf-bind-10":
    oracle = design_bench.make(TASKS[args.task], dataset_kwargs={"max_samples": 10000})
else:
    oracle= design_bench.make(TASKS[args.task])
print("oracle.x.shape:", oracle.x.shape)
offline_x = oracle.x

if oracle.is_discrete:

    oracle.map_to_logits()
    offline_x = oracle.x
    L, V = oracle.x.shape[1], oracle.x.shape[2]
    print("oracle.x.shape2:", oracle.x.shape)
    
y_min=task_to_min[TASKS[args.task]]
y_max=task_to_max[TASKS[args.task]]

    
print(f"y_min: {y_min}, y_max: {y_max}")  



model.regressor = model.regressor.to(device)

weights = model.get_inner_loop_parameter_dict(model.regressor.named_parameters())
weights = {k.replace("module.", ""): v for k, v in weights.items()}

with torch.no_grad():
 
    freeze(model.regressor)
    


    x_norm = x_norm.to(device)
    preds = model.net_forward_search(
        x=x_norm,
        weights=weights,
        training=False,
        backup_running_statistics=False,
        num_step=0
    )

    preds = preds.squeeze()  
    selected = 256
    n_sol = 128
    set_seed(seed)

    top_candidates = torch.topk(preds, k=selected).indices  
    random_subset = torch.randperm(selected)[:n_sol]  
    x_best = x_norm[top_candidates[random_subset]]
    y_best = preds[top_candidates[random_subset]]
    

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy


def search_surrogate(x, y, oracle, config, model, n_sol, opt_iter, init='topk'):
    x_start = x
    freeze(model.regressor)

    if init == 'topk':
        #x_opt = nn.Parameter(torch.tensor(deepcopy(x_start[:n_sol]), dtype=torch.float32).to(device), requires_grad=True)
        x_opt = nn.Parameter(torch.tensor(deepcopy(x_start), dtype=torch.float32).to(device), requires_grad=True)

    elif init == 'rand_idx':
        x_opt = nn.Parameter(deepcopy(x_start[torch.randperm(x_start.shape[0])[:n_sol]]).to(device), requires_grad=True)
    elif init == 'rand':
        x_opt = nn.Parameter(torch.randn(n_sol, x.shape[1]).to(device), requires_grad=True)
    else:
        raise ValueError(f"Invalid init mode: {init}")

    optimizer = optim.Adam([x_opt], lr=lr)
    solutions = []

    for itr in range(opt_iter):

        optimizer.zero_grad()
        pred =  model.net_forward_search(
        x=x_opt,
        weights=weights,
        training=False,
        backup_running_statistics=False,
        num_step=0
            )
        loss = - pred.sum()  
        loss.backward()
        optimizer.step()

        
    x_denorm_tensor= denormalize_x(x_opt, mean_x, std_x)

    if oracle.is_discrete: 
        x_denorm_tensor = x_denorm_tensor.reshape(x_denorm_tensor.shape[0],oracle.x.shape[1],oracle.x.shape[2])

    x_np = x_denorm_tensor.detach().cpu().numpy()  

    print(f"x_np shape: {x_np.shape}")
    y_np = oracle.predict(x_np)
    y_np = (y_np - y_min) / (y_max - y_min)
    print(f"result max: {np.max(y_np):.4f}")
    print(f"result min: {np.min(y_np):.4f}")

    print(f"result median: {np.median(y_np):.4f}")
    print(f"result mean: {np.mean(y_np):.4f}")
    

    solutions.append(y_np)
    unfreeze(model.regressor)
    return solutions





# ========== Run optimization ==========
run_solutions = {}
config = {"normalize_ys": True}
print("config:", config)
for r in range(n_runs):
    print(f"Run {r}")
    
    run_solutions[r]  = search_surrogate(x_best, y_best, oracle, config, model, n_sol, opt_iter, init='topk')

    
    
    
    

