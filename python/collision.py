import torch
import collision_cuda

def check_collision(triangles1: torch.Tensor, triangles2: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Check if any triangle pairs from two batched meshes are within the threshold distance.

    Args:
        triangles1: Tensor of shape [B, T1, 3, 3] (float32, CUDA)
        triangles2: Tensor of shape [B, T2, 3, 3] (float32, CUDA)
        threshold: Distance threshold

    Returns:
        collisions: Bool tensor of shape [B], True if any triangle pair is within the threshold
    """
    assert triangles1.is_cuda and triangles2.is_cuda, "Inputs must be CUDA tensors"
    assert triangles1.shape[0] == triangles2.shape[0], "Batch size must match"
    assert triangles1.shape[-2:] == (3, 3), "Triangles1 must have shape [B, T1, 3, 3]"
    assert triangles2.shape[-2:] == (3, 3), "Triangles2 must have shape [B, T2, 3, 3]"

    # print(f"[check_collision] triangles1 shape: {triangles1.shape}, triangles2 shape: {triangles2.shape}, threshold: {threshold}")

    return collision_cuda.tridist_forward(triangles1, triangles2, threshold)