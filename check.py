# import torch
# print(torch.cuda.is_available())  # Should return True
# print(torch.cuda.get_device_name(0))  # Should print "NVIDIA GeForce RTX 3050" or similar
import torch

checkpoint = torch.load("best_model (1).pth", map_location="cpu")
print(type(checkpoint))
if isinstance(checkpoint, dict):
    print(checkpoint.keys())  # Print the keys in the dictionary to understand the structure
