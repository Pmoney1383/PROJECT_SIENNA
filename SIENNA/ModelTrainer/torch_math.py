import torch
import numpy as np

def run_matrix_multiplication(a, b):
    a_tensor = torch.tensor(a, dtype=torch.float32).cuda()
    b_tensor = torch.tensor(b, dtype=torch.float32).cuda()
    result = torch.matmul(a_tensor, b_tensor)
    return result.cpu().numpy().tolist()
