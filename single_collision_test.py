import trimesh
import numpy as np
from collision_cuda import BVHWrapper
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

def get_aabbs_from_obj(path, translation=None):
    # Load obj
    verts, faces, _ = load_obj(path)
    faces_idx = faces.verts_idx

    # Create triangle vertex sets: (N, 3, 3)
    tris = verts[faces_idx]  # shape: (num_faces, 3, 3)

    # Apply translation if needed
    if translation is not None:
        translation_tensor = torch.tensor(translation, dtype=torch.float32)
        tris += translation_tensor

    # Compute per-triangle AABBs
    mins = tris.min(dim=1).values  # (num_faces, 3)
    maxs = tris.max(dim=1).values  # (num_faces, 3)

    # Convert to list of (min, max) tuples
    return [(tuple(lo.tolist()), tuple(hi.tolist())) for lo, hi in zip(mins, maxs)]

def run_cross_bvh_query(obj1_path, obj2_path, obj2_translation=None):
    print("🔹 Loading first mesh")
    aabbs1 = get_aabbs_from_obj(obj1_path)

    print("🔹 Loading second mesh")
    aabbs2 = get_aabbs_from_obj(obj2_path)  # No translation here

    print(f"📦 Building BVHs: {len(aabbs1)} and {len(aabbs2)} AABBs")
    bvh1 = BVHWrapper()
    bvh2 = BVHWrapper()
    bvh1.build(aabbs1)
    bvh2.build(aabbs2)

    # Apply translation after BVH build
    if obj2_translation is not None:
        print(f"🚚 Translating second BVH by {obj2_translation}")
        bvh2.translate(obj2_translation)

    print("🔍 Querying for collisions between meshes...")
    pairs = bvh1.query_with(bvh2)

    print(f"✅ {len(pairs)} collision pairs found.")

if __name__ == "__main__":
    run_cross_bvh_query("../4.obj", "../3.obj", obj2_translation=(358.54258925551823722,241.00750025091221573,41.52429261232017410))