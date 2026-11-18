import torch

model_path = "final_model.pt"
state_dict = torch.load(model_path, map_location=torch.device("cpu"))

print(f"Successfully loaded: {model_path}\n")
print(f"Total layers/tensors stored: {len(state_dict)}\n")

print("--- Layer Names and Tensor Shapes ---")
total_params = 0
for name, tensor in state_dict.items():
    num_params = tensor.numel()
    total_params += num_params
    print(f"{name:30} | Shape: {str(list(tensor.shape)):20} | Params: {num_params:,}")

print(f"\nTotal Model Parameters: {total_params:,}")