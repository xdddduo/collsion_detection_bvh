import csv
import time
from collision_cuda import BVHWrapper
import torch
from pytorch3d.io import load_obj


# ---------- Load and build BVH for all base meshes ----------

def get_aabbs_from_obj(path):
    verts, faces, _ = load_obj(path)
    faces_idx = faces.verts_idx
    tris = verts[faces_idx]  # shape: (N, 3, 3)

    mins = tris.min(dim=1).values
    maxs = tris.max(dim=1).values

    return [(tuple(lo.tolist()), tuple(hi.tolist())) for lo, hi in zip(mins, maxs)]


print("📦 Preloading and building BVHs for 1.obj to 5.obj...")

bvh_cache = {}
bvh2_cache = {}

for i in range(1, 6):
    path = f"../{i}.obj"
    aabbs = get_aabbs_from_obj(path)
    bvh = BVHWrapper()
    bvh.build(aabbs)
    bvh_cache[i] = bvh
    print(f"  ✅ Built BVH for {path}")


# ---------- Read CSV and Evaluate ----------

def evaluate_csv(csv_path):
    correct, total = 0, 0
    total_translate_time = 0.0
    total_query_time = 0.0

    start_all = time.time()

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            model1_id = int(row[0])
            model2_id = int(row[1])
            tx, ty, tz = map(float, row[2:5])
            label = int(row[5])

            bvh1 = bvh_cache[model1_id]

            # Load bvh2 (once)
            if model2_id not in bvh2_cache:
                path = f"../{model2_id}.obj"
                aabbs = get_aabbs_from_obj(path)
                bvh2 = BVHWrapper()
                bvh2.build(aabbs)
                bvh2_cache[model2_id] = bvh2
                print(f"🆕 Cached BVH for {path}")
            bvh2 = bvh2_cache[model2_id]

            # Translate
            t_start = time.time()
            bvh2.translate((tx, ty, tz))
            t_end = time.time()
            translate_time = t_end - t_start
            total_translate_time += translate_time

            # Query
            q_start = time.time()
            pairs = bvh1.query_with(bvh2)
            q_end = time.time()
            query_time = q_end - q_start
            total_query_time += query_time

            # Revert translation
            bvh2.translate((-tx, -ty, -tz))

            pred = 1 if len(pairs) > 0 else 0
            result = "✅ Correct" if pred == label else "❌ Wrong"
            if pred == label:
                correct += 1

            print(f"{result} | pred: {pred}, label: {label} | translate: {translate_time:.4f}s, query: {query_time:.4f}s | row: {row}")
            total += 1

    end_all = time.time()
    total_time = end_all - start_all
    acc = correct / total if total else 0.0

    print(f"\n🎯 Evaluation complete:")
    print(f"   Accuracy     : {correct}/{total} → {acc:.2%}")
    print(f"   Total time   : {total_time:.2f}s")
    print(f"   Avg/query    : {total_time / total:.4f}s")
    print(f"   Total translate time: {total_translate_time:.2f}s")
    print(f"   Total query time    : {total_query_time:.2f}s")


# ---------- Run ----------

if __name__ == "__main__":
    evaluate_csv("/hpc_stor03/sjtu_home/xiuping.zhu/courses/ece450/design/example.csv")