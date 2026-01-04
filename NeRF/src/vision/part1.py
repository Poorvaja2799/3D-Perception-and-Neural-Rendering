import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import imageio
import time
import torch
from torch import Tensor

def positional_encoding(x, num_frequencies=6, incl_input=True):
  r"""Apply positional encoding to the input.

  Args:
    x (torch.Tensor): Input tensor to be positionally encoded. 
      The dimension of x is [N, D], where N is the number of input coordinates,
      and D is the dimension of the input coordinate.
    num_frequencies (optional, int): The number of frequencies used in
     the positional encoding default: 6).
    incl_input (optional, bool): If True, concatenate the input with the 
        computed positional encoding (default: True).

    The list "results" will hold all the positional encodings. Appen
  
  Returns:
    (torch.Tensor): Positional encoding of the input tensor. 

  Example:
      >>> x = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
      >>> encoded = positional_encoding(x, num_frequencies=4, incl_input=True)
      >>> print(encoded.shape)  # Example: [N, D * (2 * num_frequencies + 1)]

  Notes:
      - The section marked between `1(a) BEGIN` and `1(a) END` applies the sine
        and cosine transformations to the input tensor `x`, appending each 
        transformed tensor to a list of results.
      - Frequencies are scaled as powers of 2: `2^i * π` for `i` in 
        range(num_frequencies).

  """
  results = []
  if incl_input:
    results.append(x)
  #############################  1(a) BEGIN  ############################
  
  # encode input tensor and append the encoded tensor to the list of results.
  # For each frequency band, append sin(x * freq) and cos(x * freq)
  if num_frequencies > 0:
    for i in range(num_frequencies):
      freq = (2.0 ** i) * np.pi
      results.append(torch.sin(x * freq))
      results.append(torch.cos(x * freq))
  
  #############################  1(a) END  ##############################
  return torch.cat(results, dim=-1)

class Model2d(nn.Module):
    """
    A simple feedforward neural network model for 2D input.

    Attributes:
        hidden_features (int): Number of units in the hidden layers. Defaults to 128.
        in_features (int): Number of input features. Defaults to 2.
        layer_in (nn.Linear): The input layer that maps from input features to hidden features.
        layer_hidden (nn.Linear): A hidden layer with a specified hidden features.
        layer_out (nn.Linear): The output layer that maps from hidden features to 3 output units.

    Methods:
        forward(x):
            Defines the forward pass of the network.
            Args:
                x (Tensor): The input tensor of shape `(batch_size, in_features)`.
            Returns:
                Tensor: The output tensor of shape `(batch_size, 3)` with values in the range [0, 1].
    """

    def __init__(self, hidden_features=128, in_features=2):
        super().__init__()
        #############################  1(b) BEGIN  ############################        
        
        # Define the layers:
        # - layer_in: Maps `in_features` to `hidden_features`.
        # - layer_hidden: A hidden layer that keeps the dimensionality `hidden_features`.
        # - layer_out: Maps `hidden_features` to the output size of 3.
        self.layer_in = nn.Linear(in_features, hidden_features)
        self.layer_hidden = nn.Linear(hidden_features, hidden_features)
        self.layer_out = nn.Linear(hidden_features, 3)
        
        #############################  1(b) END  ##############################

        def weights_init(m):
          if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

        self.apply(weights_init)        

    def forward(self, x: Tensor) -> Tensor:
        #############################  1(b) BEGIN  ############################   
       
        # Implement the forward pass:
        # - Apply ReLU activation after `layer_in` and `layer`.
        # - Use Sigmoid activation after `layer_out` to ensure output values are in [0, 1].
        x = self.layer_in(x)
        x = F.relu(x)
        x = self.layer_hidden(x)
        x = F.relu(x)
        x = self.layer_out(x)
        x = torch.sigmoid(x)
       
        #############################  1(b) END  ##############################
        return x

def train_2d_model(model, height, width, testimg, num_frequencies=0):
  device = next(model.parameters()).device

  # Optimizer parameters
  lr = 5e-4
  num_iters = 10000

  # Misc parameters
  display_every = 2000  # Number of iters after which stats are displayed

  """
  Optimizer
  """
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  """
  Train-Eval-Repeat!
  """
  # Seed RNG, for repeatability
  seed = 5670
  torch.manual_seed(seed)
  np.random.seed(seed)

  # Lists to log metrics etc.
  psnrs = []
  iternums = []

  t = time.time()
  t0 = time.time()

  # 2D Coordinates
  nx, ny = (height, width)
  x = np.linspace(-1, 1, nx, endpoint=False)
  y = np.linspace(-1, 1, ny, endpoint=False)
  xv, yv = np.meshgrid(x, y)
  coordinates = np.concatenate((xv[:,:,np.newaxis], yv[:,:,np.newaxis]), axis=2, dtype=np.float32).reshape(-1, 2)
  coordinates = torch.tensor(coordinates).to(device)

  if num_frequencies > 0:
    embedded_coordinates = positional_encoding(coordinates, num_frequencies=num_frequencies)
  else:
    embedded_coordinates = coordinates

  for i in range(num_iters+1):
    optimizer.zero_grad()
    # Run one iteration
    pred = model(embedded_coordinates)
    pred = pred.reshape(height, width, 3)
    # Compute mean-squared error between the predicted and target images. Backprop!
    loss = F.mse_loss(pred, testimg)
    loss.backward()
    optimizer.step()

    # Display images/plots/stats
    if i % display_every == 0:
      psnr = -10. * torch.log10(loss)
      print("Iteration %d " % i, "Loss: %.4f " % loss.item(), "PSNR: %.2f" % psnr.item(), \
            "Time: %.2f secs per iter" % ((time.time() - t) / display_every), "%.2f secs in total" % (time.time() - t0))
      t = time.time()
      
      psnrs.append(psnr.item())
      iternums.append(i)

      plt.figure(figsize=(13, 4))
      plt.subplot(131)
      plt.imshow(pred.detach().cpu().numpy())
      plt.title(f"Iteration {i}")
      plt.subplot(132)
      plt.imshow(testimg.cpu().numpy())
      plt.title("Target image")
      plt.subplot(133)
      plt.plot(iternums, psnrs)
      plt.title("PSNR")
      plt.show()

  print('Done!')
  last_psnr = -10. * torch.log10(loss)
  if psnr.item() >= 30:
    print(f'\x1b[32m"Your PSNR is {last_psnr}!"\x1b[0m')
  else:
    print(f'\x1b[31m"Your PSNR {last_psnr} is smaller then 30!"\x1b[0m')