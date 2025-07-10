import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.transforms import Translate
from pytorch3d.structures import Meshes
from python.collision import check_collision

DEVICE = torch.device("cuda")
THRESHOLD = 28  # In units consistent with cube size
MODEL_DIR = "tests/test_data"

def load_mesh(obj_name):
    path = os.path.join(MODEL_DIR, obj_name)
    mesh = load_objs_as_meshes([path], device="cpu")
    verts = mesh.verts_list()[0].to(DEVICE)
    faces = mesh.faces_list()[0].to(torch.int64).to(DEVICE)
    triangles = verts[faces].unsqueeze(0).contiguous()
    return triangles

def run_case(name, obj1, obj2, offset, expected):
    tri1 = load_mesh(obj1)
    tri2 = load_mesh(obj2) + torch.tensor(offset, device=DEVICE).view(1, 1, 1, 3)
    result = check_collision(tri1, tri2, THRESHOLD)
    # print(f"dist: {result}")
    actual = bool(result.item())
    status = "✅" if actual == expected else "❌"
    print(f"{status} [{name}] → expected {expected}, got {actual}")
    assert actual == expected, f"❌ Test '{name}' failed!"

def run_all_tests():
    print("▶ Running detailed triangle collision tests...")

    # # General cases
    # run_case("intersecting cubes", "cube.obj", "cube.obj", [0, 0, 0], expected=True)
    # run_case("close", "cube.obj", "cube.obj", [0, 0, 0.4], expected=True)
    # run_case("clearly separated", "cube.obj", "cube.obj", [0, 0, 5], expected=False)

    # # Boundary case: just at threshold
    # run_case("exact threshold", "cube.obj", "cube.obj", [0, 1.009, 1.009], expected=False)  # edge-to-edge

    # # Barely overlapping
    # run_case("just inside threshold", "cube.obj", "cube.obj", [0, 0, 1.01], expected=False)

    # # Degenerate triangle
    # run_case("degenerate (zero) triangle", "cube.obj", "degenerate.obj", [0, 0, 0], expected=False)

    # # One triangle very large
    # run_case("large vs small", "large_cube.obj", "cube.obj", [0, 0, 0.2], expected=True)

    # # Vertex-only touch
    # run_case("vertex overlap", "cube.obj", "cube.obj", [1.0, 1.0, 1.0], expected=True)

    run_case("test1", "3.obj", "3.obj", [309.29227925339847616,209.05064316097497112,188.07327153045596901], expected=True)

    print("✅ All detailed tests passed.")

if __name__ == "__main__":
    run_all_tests()