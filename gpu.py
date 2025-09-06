import torch
print(torch.cuda.is_available())      # True
print(torch.__version__)              # Should print with "+cu121"
