import sys
import torch
import sklearn
import pandas
import matplotlib
import dvc

print("Python executable:", sys.executable)
print("Torch version:", torch.__version__)
print("Torch CUDA available:", torch.cuda.is_available())
