# DiffStateGrad helper method
import numpy as np
import torch


def compute_rank_for_explained_variance(singular_values, explained_variance_cutoff):
    """
   Computes average rank needed across channels to explain target variance percentage.
   
   Args:
       singular_values: List of arrays containing singular values per channel
       explained_variance_cutoff: Target explained variance ratio (0-1)
   
   Returns:
       int: Average rank needed across RGB channels
   """
    total_rank = 0
    for channel_singular_values in singular_values:
        squared_singular_values = channel_singular_values ** 2
        cumulative_variance = np.cumsum(squared_singular_values) / np.sum(squared_singular_values)
        rank = np.searchsorted(cumulative_variance, explained_variance_cutoff) + 1
        total_rank += rank
    return int(total_rank / 3)

def compute_svd_and_adaptive_rank(z_t, var_cutoff):
    """
    Compute SVD and adaptive rank for the input tensor.
    
    Args:
        z_t: Input tensor (current image representation at time step t)
        var_cutoff: Variance cutoff for rank adaptation
        
    Returns:
        tuple: (U, s, Vh, adaptive_rank) where U, s, Vh are SVD components
               and adaptive_rank is the computed rank
    """
    # Compute SVD of current image representation
    U, s, Vh = torch.linalg.svd(z_t[0], full_matrices=False)
    
    # Compute adaptive rank
    s_numpy = s.detach().cpu().numpy()

    adaptive_rank = compute_rank_for_explained_variance([s_numpy], var_cutoff)
    
    return U, s, Vh, adaptive_rank

def apply_diffstategrad(norm_grad, iteration_count, period, U=None, s=None, Vh=None, adaptive_rank=None,
                        projection_mode="core"):
    """
    Compute projected gradient using DiffStateGrad algorithm.
    
    Args:
        norm_grad: Normalized gradient
        iteration_count: Current iteration count
        period: Period of SVD projection
        U: Left singular vectors from SVD
        s: Singular values from SVD
        Vh: Right singular vectors from SVD
        adaptive_rank: Computed adaptive rank
        projection_mode: Projection type. "core" matches the original DiffStateGrad
                         implementation; "tangent" uses the rank-r matrix tangent
                         projection; "none" disables projection.
        
    Returns:
        torch.Tensor: Projected gradient if period condition is met, otherwise original gradient
    """
    if projection_mode in ["none", None] or period == 0:
        return norm_grad

    if period != 0 and iteration_count % period == 0:
        if any(param is None for param in [U, s, Vh, adaptive_rank]):
            raise ValueError("SVD components and adaptive_rank must be provided when iteration_count % period == 0")

        if projection_mode == "fixed":
            projection_mode = "core"
        if projection_mode == "normal_removed":
            projection_mode = "tangent"
        if projection_mode not in ["core", "tangent"]:
            raise ValueError(f"Unknown projection_mode '{projection_mode}'")
        
        A = U[:, :, :adaptive_rank]
        B = Vh[:, :adaptive_rank, :]
        grad = norm_grad[0]

        # Original DiffStateGrad: core-only spectral projection
        low_rank_grad = torch.matmul(A.permute(0, 2, 1), grad) @ B.permute(0, 2, 1)
        core_grad = torch.matmul(A, low_rank_grad) @ B

        if projection_mode == "core":
            projected_grad = core_grad
        else:
            # Proper rank-r matrix tangent projection:
            # P_T(G) = U U^T G + G V V^T - U U^T G V V^T.
            left_grad = torch.matmul(A, torch.matmul(A.permute(0, 2, 1), grad))
            right_grad = torch.matmul(torch.matmul(grad, B.permute(0, 2, 1)), B)
            projected_grad = left_grad + right_grad - core_grad
        
        # Reshape projected gradient to match original shape
        projected_grad = projected_grad.float().unsqueeze(0)  # Add batch dimension back
        
        return projected_grad
    
    return norm_grad
