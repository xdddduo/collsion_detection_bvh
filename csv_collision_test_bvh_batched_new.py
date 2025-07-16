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
    path = f"tests/test_data/{i}.obj"
    aabbs = get_aabbs_from_obj(path)
    bvh = BVHWrapper()
    bvh.build(aabbs)
    bvh_cache[i] = bvh
    print(f"  ✅ Built BVH for {path}")


# ---------- Read CSV and Evaluate with NEW Batched BVH ----------

def evaluate_csv_batched_new(csv_path, batch_size=100):
    print(f"🔄 Reading CSV and preparing NEW batched BVH tests...")
    
    # Read all CSV data first
    test_data = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            model1_id = int(row[0])
            model2_id = int(row[1])
            tx, ty, tz = map(float, row[2:5])
            label = int(row[5])
            test_data.append((model1_id, model2_id, (tx, ty, tz), label))
    
    print(f"📊 Found {len(test_data)} tests to process")
    
    # Preload all BVH2s that will be needed
    for model1_id, model2_id, _, _ in test_data:
        if model2_id not in bvh2_cache:
            path = f"tests/test_data/{model2_id}.obj"
            aabbs = get_aabbs_from_obj(path)
            bvh2 = BVHWrapper()
            bvh2.build(aabbs)
            bvh2_cache[model2_id] = bvh2
            print(f"🆕 Cached BVH for {path}")
    
    # Batched processing
    total = len(test_data)
    correct = 0
    all_results = []
    start_time = time.time()
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = test_data[batch_start:batch_end]
        bvh2_list = []
        for model1_id, model2_id, (tx, ty, tz), _ in batch:
            bvh2 = bvh2_cache[model2_id]
            bvh2.translate((tx, ty, tz))
            bvh2_list.append(bvh2)
        bvh1 = bvh_cache[batch[0][0]]  # Use first BVH1 for all in batch (assume same)
        batch_results = bvh1.query_with_batched(bvh2_list)
        for i, (model1_id, model2_id, (tx, ty, tz), label) in enumerate(batch):
            pred = 1 if len(batch_results[i]) > 0 else 0
            if pred == label:
                correct += 1
            result = "✅ Correct" if pred == label else "❌ Wrong"
            print(f"{result} | pred: {pred}, label: {label} | test {batch_start + i + 1}/{total}")
        # Revert translations
        for idx, (model1_id, model2_id, (tx, ty, tz), _) in enumerate(batch):
            bvh2_list[idx].translate((-tx, -ty, -tz))
        all_results.extend(batch_results)
    end_time = time.time()
    total_time = end_time - start_time
    accuracy = correct / total
    print(f"\n🎯 NEW Batched BVH Evaluation complete:")
    print(f"   Accuracy     : {correct}/{total} → {accuracy:.2%}")
    print(f"   Total time   : {total_time:.4f}s")
    print(f"   Avg/query    : {total_time / total:.6f}s")
    print(f"   Throughput   : {total / total_time:.1f} tests/second")
    return accuracy, total_time


# ---------- Run ----------

if __name__ == "__main__":
    accuracy, total_time = evaluate_csv_batched_new("tests/test_data/example.csv", batch_size=100) 