import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Callable

class NerfModel(nn.Module):
    
    def __init__(self, in_channels: int, filter_size: int=256):
        """This network will have a total of 8 fully connected layers. The activation function will be ReLU

        The number of input features to layer 5 will be a bit different. Refer to the docstring for the forward pass.
        Do not include an activation after layer 8 in the Sequential block. Layer 8's should output 4 features.

        Args
        ---
        in_channels (int): the number of input features from 
            the data
        filter_size (int): the number of in/out features for all layers. Layers 1 (because of in_channels), 5, and 8 are
            a bit different.
        """
        super().__init__()

        self.fc_layers_group1: nn.Sequential = None  # For layers 1-3
        self.layer_4: nn.Linear = None
        self.fc_layers_group2: nn.Sequential = None  # For layers 5-8
        self.loss_criterion = None

        # Layers 1-3: sequential block
        self.fc_layers_group1 = nn.Sequential(
            nn.Linear(in_channels, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU()
        )

        # Layer 4: a single linear layer (followed by activation in forward)
        self.layer_4 = nn.Linear(filter_size, filter_size)

        # Layers 5-8: note layer 5 input will be concat([post_act_layer4, post_act_layer3])
        # so the first linear expects 2 * filter_size -> filter_size
        self.fc_layers_group2 = nn.Sequential(
            nn.Linear(2 * filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, 4)  # no activation here
        )

        # Loss criterion (not used directly by tests but provided)
        self.loss_criterion = nn.MSELoss()

  
  
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform the forward pass of the model. 
        
        NOTE: The input to layer 5 should be the concatenation of post-activation values from layer 4 with 
        post-activation values from layer 3. Therefore, be extra careful about how self.layer_4 is used, the order of
        concatenation, and what the specified input shape to layer 5 should be. The output from layer 5 and the 
        dimensions thereafter should be filter_size.
        
        Args
        ---
        x (torch.Tensor): input of shape 
            (batch_size, in_channels)
        
        Returns
        ---
        rgb (torch.Tensor): The predicted rgb values with 
            shape (batch_size, 3)
        sigma (torch.Tensor): The predicted density values with shape (batch_size)
        """
        rgb = None
        sigma = None

 
        # Pass through layers 1-3 (fc_layers_group1). The output of this block
        # is the post-activation values from layer 3 which we'll reuse.
        post3 = self.fc_layers_group1(x)

        # Layer 4 (apply linear then activation)
        post4 = self.layer_4(post3)
        post4 = F.relu(post4)

        # Concatenate post-activation values from layer 4 with those from layer 3
        # Order: layer4 then layer3 (as specified in the docstring)
        concat45 = torch.cat([post4, post3], dim=-1)

        # Pass through layers 5-8
        out = self.fc_layers_group2(concat45)

        # Split output into rgb (3) and sigma (1)
        rgb = torch.sigmoid(out[..., :3])
        sigma = F.relu(out[..., 3])

  

        return rgb, sigma

def get_rays(height: int, width: int, intrinsics: torch.Tensor, tform_cam2world: torch.Tensor) \
    -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the origin and direction of rays passing through all pixels of an image (one ray per pixel).
    
    Args
    ---
    height (int): 
        the height of an image.
    width (int): the width of an image.
    intrinsics (torch.Tensor): Camera intrinsics matrix of shape (3, 3).
    tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
        transforms a 3D point from the camera coordinate space to the world frame coordinate space.
    
    Returns
    ---
    ray_origins (torch.Tensor): A tensor of shape :math:`(height, width, 3)` denoting the centers of
        each ray. Note that desipte that all ray share the same origin, 
        here we ask you to return the ray origin for each ray as (height, width, 3).
    ray_directions (torch.Tensor): A tensor of shape :math:`(height, width, 3)` denoting the
        direction of each ray.
    """
    device = tform_cam2world.device
    ray_directions = torch.zeros((height, width, 3), device=device)  # placeholder
    ray_origins = torch.zeros((height, width, 3), device=device)  # placeholder

  
    # Move tensors to the correct device
    intrinsics = intrinsics.to(device)
    tform_cam2world = tform_cam2world.to(device)

    # Create pixel coordinate grid (v: rows, u: cols)
    v = torch.arange(0, height, device=device)
    u = torch.arange(0, width, device=device)
    vv, uu = torch.meshgrid(v, u, indexing='ij')

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Camera-space directions (assume z = 1).
    # Use integer pixel coordinates (u, v) without 0.5 offset — matches dataset expectation.
    x_cam = (uu.float() - cx) / fx
    y_cam = (vv.float() - cy) / fy
    dirs_cam = torch.stack([x_cam, y_cam, torch.ones_like(x_cam)], dim=-1)  # (H, W, 3)

    # Rotate directions into world space (do not normalize; tests expect un-normalized dirs)
    R = tform_cam2world[:3, :3]
    ray_directions = dirs_cam @ R.T  # (H, W, 3)

    # Ray origins: camera origin in world coords, expanded per-pixel
    cam_origin_world = tform_cam2world[:3, 3]
    ray_origins = cam_origin_world.view(1, 1, 3).expand(height, width, 3).to(device)

  

    return ray_origins, ray_directions

def sample_points_from_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near_thresh: float,
    far_thresh: float,
    num_samples: int,
    randomize:bool = True
) -> tuple[torch.tensor, torch.tensor]:
    """Sample 3D points on the given rays. The near_thresh and far_thresh
    variables indicate the bounds of sampling range.
    
    Args
    ---
    ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
        `get_rays` method (shape: :math:`(height, width, 3)`).
    ray_directions (torch.Tensor): Direction of each ray in the "bundle" as returned by the
        `get_rays` method (shape: :math:`(height, width, 3)`).
    near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
        coordinate that is of interest/relevance).
    far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
        coordinate that is of interest/relevance).
    num_samples (int): Number of samples to be drawn along each ray. Samples are drawn
        randomly, whilst trying to ensure "some form of" uniform spacing among them.
    randomize (optional, bool): Whether or not to randomize the sampling of query points.
        By default, this is set to `True`. If disabled (by setting to `False`), we sample
        uniformly spaced points along each ray (i.e., the lower bound of each bin).
    
    Returns
    ---
    query_points (torch.Tensor): Query 3D points along each ray
        (shape: :math:`(height, width, num_samples, 3)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
        (shape: :math:`(height, width, num_samples)`).
    """
    device = ray_origins.device
    height, width = ray_origins.shape[:2]
    depth_values = torch.zeros((height, width, num_samples), device=device) # placeholder
    query_points = torch.zeros((height, width, num_samples, 3), device=device) # placeholder
    
  
    # Create depth bins and optionally jitter samples within each bin (stratified sampling)
    # base t values in [0,1) at bin starts: 0, 1/num_samples, ..., (num_samples-1)/num_samples
    t_vals = torch.arange(0, num_samples, device=device, dtype=torch.float32) / float(num_samples)

    # Shape to (1,1,num_samples) for broadcasting
    t_vals = t_vals.view(1, 1, num_samples)

    if randomize:
        # jitter within each bin of width 1/num_samples
        jitter = torch.rand((height, width, num_samples), device=device)
        t_vals = t_vals + jitter / float(num_samples)
    else:
        t_vals = t_vals.expand(height, width, num_samples)

    # Depth values in [near_thresh, far_thresh)
    depth_values = near_thresh + t_vals * (far_thresh - near_thresh)

    # Compute query points: o + d * depth
    # ray_origins: (H, W, 3), ray_directions: (H, W, 3)
    # Expand to (H, W, num_samples, 3)
    origins_exp = ray_origins.unsqueeze(2)  # (H, W, 1, 3)
    dirs_exp = ray_directions.unsqueeze(2)  # (H, W, 1, 3)
    depth_exp = depth_values.unsqueeze(-1)  # (H, W, num_samples, 1)
    query_points = origins_exp + dirs_exp * depth_exp

  
    
    return query_points, depth_values

def cumprod_exclusive(x: torch.tensor) -> torch.tensor:
    """ Helper function that computes the cumulative product of the input tensor, excluding the current element
    Example:
    > cumprod_exclusive(torch.tensor([1,2,3,4,5]))
    > tensor([ 1,  1,  2,  6, 24])
    
    Args:
    -   x: Tensor of length N
    
    Returns:
    -   cumprod: Tensor of length N containing the cumulative product of the tensor
    """

    cumprod = torch.cumprod(x, -1)
    cumprod = torch.roll(cumprod, 1, -1)
    cumprod[..., 0] = 1.
    return cumprod

def compute_compositing_weights(sigma: torch.Tensor, depth_values: torch.Tensor) -> torch.Tensor:
    """This function will compute the compositing weight for each query point.

    Args
    ---
    sigma (torch.Tensor): Volume density at each query location (X, Y, Z)
        (shape: :math:`(height, width, num_samples)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
        (shape: :math:`(height, width, num_samples)`).
    
    Returns:
    weights (torch.Tensor): Rendered compositing weight of each sampled point 
        (shape: :math:`(height, width, num_samples)`).
    """

    device = depth_values.device
    weights = torch.ones_like(sigma, device=device) # placeholder

  
    # sigma: (H, W, N), depth_values: (H, W, N)
    # compute deltas between consecutive depth samples
    deltas = depth_values[..., 1:] - depth_values[..., :-1]
    # append a large delta for the last interval
    last_delta = torch.full_like(depth_values[..., :1], 1e10)
    deltas = torch.cat([deltas, last_delta], dim=-1)

    # compute alpha = 1 - exp(-sigma * delta)
    alpha = 1.0 - torch.exp(-sigma * deltas)

    # compute transmittance T as exclusive cumulative product of exp(-sigma * delta)
    exp_term = torch.exp(-sigma * deltas)
    T = cumprod_exclusive(exp_term)

    # weights = alpha * T
    weights = alpha * T

  

    return weights

def get_minibatches(inputs: torch.Tensor, chunksize: int = 1024 * 32) -> list[torch.Tensor]:
    """Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

def render_image_nerf(height: int, width: int, intrinsics: torch.tensor, tform_cam2world: torch.tensor,
                      near_thresh: float, far_thresh: float, depth_samples_per_ray: int,
                      encoding_function: Callable, model:NerfModel, rand:bool=False) \
                      -> tuple[torch.Tensor, torch.Tensor]:
    """ This function will utilize all the other rendering functions that have been implemented in order to sample rays,
    pass those rays to the NeRF model to get color and density predictions, and then use volume rendering to create
    an image of this view. 

    Hints: 
    ---
    It is a good idea to "flatten" the height/width dimensions of the data when passing to the NeRF (maintain the color
    channel dimension) and then "unflatten" the outputs. 
    To avoid running into memory limits, it's recommended to use the given get_minibatches() helper function to 
    divide up the input into chunks. For each minibatch, supply them to the model and then concatenate the corresponding
    output vectors from each minibatch to form the complete outpute vectors. 
    
    Args
    ---
    height (int): 
        the pixel height of an image.
    width (int): the pixel width of an image.
    intrinsics (torch.tensor): Camera intrinsics matrix of shape (3, 3).
    tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
        transforms a 3D point from the camera coordinate space to the world frame coordinate space.
    near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
        coordinate that is of interest/relevance).
    far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
        coordinate that is of interest/relevance).
    depth_samples_per_ray (int): Number of samples to be drawn along each ray. Samples are drawn
        randomly, whilst trying to ensure "some form of" uniform spacing among them.
    encoding_function (Callable): The function used to encode the query points (e.g. positional encoding)
    model (NerfModel): The NeRF model that will be used to render this image
    randomize (optional, bool): Whether or not to randomize the sampling of query points.
        By default, this is set to `True`. If disabled (by setting to `False`), we sample
        uniformly spaced points along each ray (i.e., the lower bound of each bin).
    
    Returns
    ---
    rgb_predicted (torch.tensor): 
        A tensor of shape (height, width, num_channels) with the color info at each pixel.
    depth_predicted (torch.tensor): A tensor of shape (height, width) containing the depth from the camera at each pixel.
    """

    rgb_predicted, depth_predicted = None, None

  
    # 1) Sample rays for the image
    ray_origins, ray_directions = get_rays(height, width, intrinsics, tform_cam2world)

    # 2) Sample points along rays (stratified if rand True)
    query_points, depth_values = sample_points_from_rays(ray_origins, ray_directions,
                                                        near_thresh, far_thresh,
                                                        depth_samples_per_ray, randomize=rand)

    H, W, N = query_points.shape[:3]

    # 3) Encode query points and prepare batches for the model
    model_device = next(model.parameters()).device
    pts_flat = query_points.reshape(-1, 3).to(model_device)
    embedded = encoding_function(pts_flat)
    if isinstance(embedded, torch.Tensor):
        embedded = embedded.to(model_device)

    # 4) Run model in minibatches
    rgb_chunks = []
    sigma_chunks = []
    for mb in get_minibatches(embedded):
        rgb_mb, sigma_mb = model(mb)
        rgb_chunks.append(rgb_mb)
        sigma_chunks.append(sigma_mb)

    rgb_all = torch.cat(rgb_chunks, dim=0)
    sigma_all = torch.cat(sigma_chunks, dim=0)

    # Reshape to (H, W, N, 3) and (H, W, N)
    rgb_samples = rgb_all.view(H, W, N, 3)
    sigma_samples = sigma_all.view(H, W, N)

    # Ensure depth_values on model device
    depth_values = depth_values.to(model_device)

    # 5) Compute compositing weights
    weights = compute_compositing_weights(sigma_samples, depth_values)

    # 6) Composite color and depth
    rgb_predicted = (weights.unsqueeze(-1) * rgb_samples).sum(dim=2)
    depth_predicted = (weights * depth_values).sum(dim=2)
    
  

    return rgb_predicted, depth_predicted