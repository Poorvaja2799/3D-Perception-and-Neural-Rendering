import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from vision.part1 import positional_encoding
from vision.part2 import NerfModel, render_image_nerf

def train_nerf(images, tform_cam2world, cam_intrinsics, testpose, testimg, height, width, 
               near_thresh, far_thresh, device, num_frequencies=6, depth_samples_per_ray=64, 
               lr=5e-4, num_iters=1000, display_every=25, seed=4476):
    """
    Train a Neural Radiance Field (NeRF) model.

    Args:
        images (torch.Tensor): Tensor of training images.
        tform_cam2world (torch.Tensor): Transformation matrices.
        cam_intrinsics (torch.Tensor): Camera intrinsic parameters.
        testpose (torch.Tensor): Camera pose for test image.
        testimg (torch.Tensor): Test image.
        height (int): Image height.
        width (int): Image width.
        near_thresh (float): Near threshold for rendering.
        far_thresh (float): Far threshold for rendering.
        device (torch.device): Device to use for computation (e.g., 'cuda' or 'cpu').
        num_frequencies (int): Number of frequencies for positional encoding.
        depth_samples_per_ray (int): Number of depth samples along each ray.
        lr (float): Learning rate for the optimizer.
        num_iters (int): Number of training iterations.
        display_every (int): Frequency of displaying results.
        seed (int): Random seed for reproducibility.
    """
    # Number of functions used in the positional encoding
    encode = lambda x: positional_encoding(x, num_frequencies=num_frequencies)
    encode_channels = num_frequencies * 2 * 3 + 3

    # Seed RNG, for repeatability
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create output directory
    os.makedirs('output', exist_ok=True)

    """
    Model
    """
    # TODO: Define and initialize the NeRF model.
    # 1. Create an instance of the NerfModel with the correct number of input channels.
    # 2. Ensure the model is placed on the correct device.
    # 3. Define a weight initialization function for the model.
    #    - Use Xavier Uniform initialization for torch.nn.Linear layers.
    # 4. Apply the weight initialization to the model using .apply(weights_init).

  
    model = NerfModel(in_channels=encode_channels)
    model.to(device)

    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(weights_init)

  
    """
    Optimizer
    """
    # TODO: Initialize the Adam optimizer with the model's parameters and learning rate.
    # The Adam optimizer adjusts the learning rate dynamically during training and updates the model's parameters.
    # Use the model's parameters (obtained via model.parameters()) for optimization.
    # Set the learning rate (lr) for controlling the step size of the updates.
  
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  

    # Lists to log metrics etc.
    psnrs = []
    iternums = []
    best_psnr = 0

    # Training loop
    t0 = time.time()
    t = t0
    for i in range(num_iters + 1):
        # Randomly pick a target image
        target_img_idx = np.random.randint(images.shape[0])
        target_img = images[target_img_idx].to(device)
        target_tform_cam2world = tform_cam2world[target_img_idx].to(device)


        """
        TODO: Run one iteration of NeRF and get the rendered RGB image.
        Use the render_image_nerf function to generate an RGB image and depth map.
        Pass the necessary parameters such as image dimensions, camera intrinsics, and transformation matrix.
        """
 
        # Prepare intrinsics for this target (support per-image or single intrinsics)
        intr = cam_intrinsics[target_img_idx] if getattr(cam_intrinsics, 'ndim', 0) == 3 else cam_intrinsics
        if not isinstance(intr, torch.Tensor):
            intr = torch.from_numpy(np.array(intr))
        intr = intr.to(device)

        rgb_predicted, depth_predicted = render_image_nerf(
            height, width, intr, target_tform_cam2world, near_thresh, far_thresh,
            depth_samples_per_ray, encode, model, rand=True)

        rgb_predicted = rgb_predicted.to(device)

  

        """
        TODO: Compute mean-squared error between the predicted and target images and backpropagate.
        Calculate the loss as the mean-squared error (MSE) between the predicted RGB image and the target image.
        Perform backpropagation to compute gradients and update the model's weights.
        """
 
        loss = F.mse_loss(rgb_predicted, target_img)
        loss.backward()

  


        optimizer.step()
        optimizer.zero_grad()

        # Display results
        if i % display_every == 0: 
            # TODO: Render the held-out view.
            # Use the render_image_nerf function to generate an RGB image and depth map for the test pose.
            # Disable gradient tracking as we are only visualizing and not training.
  
            intr_test = cam_intrinsics if getattr(cam_intrinsics, 'ndim', 0) == 2 else cam_intrinsics[0]
            if not isinstance(intr_test, torch.Tensor):
                intr_test = torch.from_numpy(np.array(intr_test))
            intr_test = intr_test.to(device)

            with torch.no_grad():
                rgb_predicted, depth_predicted = render_image_nerf(
                    height, width, intr_test, testpose.to(device), near_thresh,
                    far_thresh, depth_samples_per_ray, encode, model, rand=False)
            
  

            """
            TODO: Compute the loss and PSNR.
            Calculate the mean-squared error (MSE) loss between the predicted image and the test image.
            Compute the Peak Signal-to-Noise Ratio (PSNR) based on the MSE loss.
            """
  
            loss = F.mse_loss(rgb_predicted, testimg.to(device))
            psnr = -10. * torch.log10(loss)

              

            print(f"Iteration {i} | Loss: {loss.item():.4f} | PSNR: {psnr.item():.2f} | "
                    f"Time per iter: {(time.time() - t) / display_every:.2f}s | "
                    f"Total time: {(time.time() - t0) / 60:.2f} min")
            t = time.time()

            psnrs.append(psnr.item())
            iternums.append(i)

            if psnr > best_psnr:
                print(f'PSNR imrpoved from {best_psnr} to {psnr}')
                print(f'Saving model to nerf_model.pth')
                best_psnr = psnr
                torch.save(model.state_dict(), os.path.join('output', 'nerf_model.pth'))

            # Visualization
            plt.figure(figsize=(16, 4))
            plt.subplot(141)
            plt.imshow(rgb_predicted.detach().cpu().numpy())
            plt.title(f"Iteration {i}")
            plt.subplot(142)
            plt.imshow(testimg.detach().cpu().numpy())
            plt.title("Target image")
            plt.subplot(143)
            plt.plot(iternums, psnrs)
            plt.title("PSNR")
            plt.subplot(144)
            plt.imshow(depth_predicted.detach().cpu().numpy())
            plt.title("Depth")
            plt.show()          
 
    print("Training completed!")

    return model, encode


# Main block for standalone execution
if __name__ == "__main__":
    print("This script is designed to encapsulate training logic. Run `train_nerf` by importing this module.")

