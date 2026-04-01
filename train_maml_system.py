from data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from few_shot_learning_system import MAMLFewShotRegressor
from utils.parser_utils import get_args
import random
import numpy as np
import torch


def set_global_seed(seed):
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





args, device = get_args()
set_global_seed(args.seed)
args.experiment_name = f"path/{args.task}_OptBias_Loss_batch_size_{args.batch_size}_num_tar_{args.num_tar}_num_ctx_{args.num_ctx}_sampling_strategy_{args.sampling_strategy}_{args.topk}r_seed_{args.seed}_num_points_{args.num_points}_num_functions_{args.num_functions}/saved_models"

model = MAMLFewShotRegressor(im_shape=(args.batch_size, args.dimension), device=device,args=args)


data = MetaLearningSystemDataLoader#args=args, device=device


maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
maml_system.run_experiment()
