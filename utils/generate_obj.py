import os

def write_unit_cube_obj(path, scale=1.0):
    s = scale / 2.0
    vertices = [
        (-s, -s, -s),
        ( s, -s, -s),
        ( s,  s, -s),
        (-s,  s, -s),
        (-s, -s,  s),
        ( s, -s,  s),
        ( s,  s,  s),
        (-s,  s,  s),
    ]
    faces = [
        (1, 2, 3), (1, 3, 4),  # bottom
        (5, 6, 7), (5, 7, 8),  # top
        (1, 5, 6), (1, 6, 2),  # front
        (2, 6, 7), (2, 7, 3),  # right
        (3, 7, 8), (3, 8, 4),  # back
        (4, 8, 5), (4, 5, 1),  # left
    ]

    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    print(f"✅ Wrote cube to {path} (scale={scale})")

def write_degenerate_obj(path):
    with open(path, "w") as f:
        f.write("v 0 0 0\n" * 3)
        f.write("f 1 2 3\n")
    print(f"✅ Wrote degenerate triangle to {path}")

def main():
    os.makedirs("tests/test_data", exist_ok=True)
    write_unit_cube_obj("tests/test_data/cube.obj", scale=1.0)
    write_unit_cube_obj("tests/test_data/large_cube.obj", scale=5.0)
    write_degenerate_obj("tests/test_data/degenerate.obj")

if __name__ == "__main__":
    main()