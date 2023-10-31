import torch
#https://stackoverflow.com/questions/59129812/how-to-avoid-cuda-out-of-memory-in-pytorch
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

import gc
gc.collect()