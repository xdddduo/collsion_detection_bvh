import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import csv
import torch
import logging
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Translate
from python.collision import check_collision

# --- CONFIGURATION ---
INTERFERENCE_THRESHOLD_MM = 28
DEVICE = torch.device("cuda")
LOG_FILE = "collision_debug.log"
BASE_MODEL_DIR = os.environ.get("MODEL_DIR", "tests/test_data")

# --- SETUP LOGGING ---
logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
console = logging.getLogger()
console.addHandler(logging.StreamHandler())  # Console output

# --- GLOBAL MODEL CACHE ---
model_cache = {}

def load_mesh_cached(model_id: int):
    if model_id in model_cache:
        return model_cache[model_id]

    path = os.path.join(BASE_MODEL_DIR, f"{model_id}.obj")
    logging.debug(f"Attempting to load mesh: {path}")

    if not os.path.exists(path):
        logging.error(f"Model file '{path}' not found.")
        return None

    try:
        # Load on CPU first to avoid CUDA internal bug
        mesh = load_objs_as_meshes([path], device=DEVICE)
        # verts = [v.to(DEVICE) for v in mesh_cpu.verts_list()]
        # faces = [f.to(torch.int64).to(DEVICE) for f in mesh_cpu.faces_list()]
        # mesh = Meshes(verts=verts, faces=faces)
        model_cache[model_id] = mesh
        logging.info(f"Successfully loaded model {model_id}.obj via CPU -> CUDA transfer")
        return mesh
    except Exception as e:
        logging.exception(f"Failed to load model {model_id}: {e}")
        return None

def mesh_to_triangle_tensor(mesh: Meshes) -> torch.Tensor:
    try:
        verts = mesh.verts_packed()
        faces = mesh.faces_packed()
        if faces.shape[0] == 0:
            logging.warning("Mesh has 0 triangles.")
        triangles = verts[faces]  # (T, 3, 3)
        return triangles.unsqueeze(0).contiguous()  # (1, T, 3, 3)
    except Exception as e:
        logging.exception("Failed to convert mesh to triangle tensor.")
        return None

def pad_triangle_tensor(triangle_tensor, max_triangles):
    """
    Pads a triangle tensor to have `max_triangles` triangles.
    Args:
        triangle_tensor (torch.Tensor): Tensor of shape (1, T, 3, 3).
        max_triangles (int): Maximum number of triangles to pad to.
    Returns:
        torch.Tensor: Padded tensor of shape (1, max_triangles, 3, 3).
    """
    current_triangles = triangle_tensor.shape[1]
    if current_triangles < max_triangles:
        padding = torch.zeros(
            (1, max_triangles - current_triangles, 3, 3),
            dtype=triangle_tensor.dtype,
            device=triangle_tensor.device
        )
        return torch.cat([triangle_tensor, padding], dim=1)
    return triangle_tensor

def run_test_batch(batch_cases):
    logging.info(f"\n--- Processing Batch of {len(batch_cases)} Test Cases ---")

    try:
        meshes1 = []
        meshes2 = []
        translations = []
        expected_results = []

        for case_num, model1_id, model2_id, t2_xyz, expected in batch_cases:
            logging.info(f"Test Case {case_num}: Models {model1_id}.obj vs {model2_id}.obj")
            logging.info(f"Model 2 Translation: {t2_xyz}")
            logging.info(f"Expected Interference: {'Yes' if expected else 'No'}")

            mesh1 = load_mesh_cached(model1_id)
            mesh2 = load_mesh_cached(model2_id)

            if mesh1 is None or mesh2 is None:
                logging.warning(f"Model loading failed for case {case_num}. Skipping.")
                continue

            translation = torch.tensor(t2_xyz, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            meshes1.append(mesh1)
            meshes2.append(mesh2)
            translations.append(translation)
            expected_results.append(expected)

        if not meshes1 or not meshes2:
            logging.warning("No valid test cases in batch. Skipping.")
            return 0, len(batch_cases)

        # Convert meshes to triangle tensors
        triangle_tensors1 = [mesh_to_triangle_tensor(mesh) for mesh in meshes1]
        triangle_tensors2 = [
            mesh_to_triangle_tensor(Meshes(
                verts=Translate(t, dtype=torch.float32, device=DEVICE).transform_points(mesh.verts_padded()),
                faces=mesh.faces_padded()
            )) for mesh, t in zip(meshes2, translations)
        ]

        # Find the maximum number of triangles
        max_triangles1 = max(tensor.shape[1] for tensor in triangle_tensors1)
        max_triangles2 = max(tensor.shape[1] for tensor in triangle_tensors2)

        # Pad triangle tensors
        padded_tri1_batch = torch.cat([pad_triangle_tensor(tensor, max_triangles1) for tensor in triangle_tensors1], dim=0)
        padded_tri2_batch = torch.cat([pad_triangle_tensor(tensor, max_triangles2) for tensor in triangle_tensors2], dim=0)

        logging.info(f"Batch Triangle Tensor Shapes: tri1={padded_tri1_batch.shape}, tri2={padded_tri2_batch.shape}")

        # Perform collision check for the batch
        results = check_collision(padded_tri1_batch, padded_tri2_batch, threshold=INTERFERENCE_THRESHOLD_MM)
        passed = sum(bool(result.item()) == expected for result, expected in zip(results, expected_results))

        logging.info(f"Batch Results: Passed {passed}/{len(batch_cases)}")
        return passed, len(batch_cases)

    except Exception as e:
        logging.exception("Batch processing failed with an exception.")
        return 0, len(batch_cases)

    finally:
        torch.cuda.empty_cache()

def main(csv_path, batch_size=10):
    if not os.path.exists(csv_path):
        logging.critical(f"CSV file '{csv_path}' not found.")
        return

    total, passed = 0, 0
    batch_cases = []

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            total += 1
            # logging.debug(f"CSV Line {total}: {line}")
            if len(line) != 6:
                logging.warning(f"Skipping malformed line {total}: {line}")
                continue
            try:
                m1, m2 = int(line[0]), int(line[1])
                t_xyz = [float(line[2]), float(line[3]), float(line[4])]
                expected = bool(int(line[5]))
                batch_cases.append((total, m1, m2, t_xyz, expected))

                # Process batch when it reaches the batch size
                if len(batch_cases) == batch_size:
                    # logging.info(f"\n[Batch Ready] Size: {len(batch_cases)} cases")
                    
                    # Optionally inspect triangle counts for diagnostics
                    # for i, (case_num, m1, m2, t_xyz, expected) in enumerate(batch_cases):
                    #     mesh1 = load_mesh_cached(m1)
                    #     mesh2 = load_mesh_cached(m2)
                    #     if mesh1 and mesh2:
                    #         num_tri1 = mesh1.num_faces_per_mesh()[0].item()
                    #         num_tri2 = mesh2.num_faces_per_mesh()[0].item()
                    #         logging.info(f"  Case {case_num}: Model {m1} (T1={num_tri1}) vs Model {m2} (T2={num_tri2})")

                    batch_passed, batch_total = run_test_batch(batch_cases)
                    passed += batch_passed
                    batch_cases = []  # Reset batch

            except Exception as e:
                logging.exception(f"Error parsing line {total}: {line}")

        # Process remaining cases in the last batch
        if batch_cases:
            batch_passed, batch_total = run_test_batch(batch_cases)
            passed += batch_passed

    logging.info("\n--- Test Summary ---")
    logging.info(f"Total: {total} | Passed: {passed} | Failed: {total - passed}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to test cases CSV")
    args = parser.parse_args()
    main(args.csv)