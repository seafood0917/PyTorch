# Training on GPU
# Can transfer the nn onto the GPU just like a Tensor onto the GPU
# Define our device as the first visible cuda device if we have CUDA availble
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# If device is a CUDA device.
# Then these methods will recursively go over all modules and convert their
# parameters and buffers to CUDA tensors.
net.to(device)

# will have to send the inputs and targets at every step to the GPU too.
inputs, labels = data[0].to(device), data[1].to(device)
