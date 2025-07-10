import os
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.transforms import Translate
from pytorch3d.structures import Meshes
from python.collision import check_collision

DEVICE = torch.device("cuda")
MODEL_DIR = "tests/test_data"
THRESHOLD = 28.0  # mm

def load_triangle_tensor(obj_path):
    mesh = load_objs_as_meshes([obj_path], device="cpu")
    verts = mesh.verts_list()[0].to(DEVICE)
    faces = mesh.faces_list()[0].to(DEVICE)
    triangles = verts[faces].unsqueeze(0).contiguous()
    return triangles

def test_pair(obj1, obj2, translation=[0, 0, 0], expected=True):
    tri1 = load_triangle_tensor(obj1)
    tri2 = load_triangle_tensor(obj2)
    trans = torch.tensor(translation, device=DEVICE).view(1, 1, 1, 3)
    tri2 = tri2 + trans
    result = check_collision(tri1, tri2, THRESHOLD)
    assert bool(result.item()) == expected, f"Test failed: {obj1} vs {obj2} → got {result}"

if __name__ == "__main__":
    test_pair("tests/test_data/cube.obj", "tests/test_data/cube.obj", [0, 0, 10], expected=False)
    test_pair("tests/test_data/cube.obj", "tests/test_data/cube.obj", [0, 0, 1], expected=True)
    print("✅ All triangle-level tests passed.")