#!/usr/bin/env python3
"""
Analyse STL bounding boxes to determine joint positions in CAD coordinates.

Usage:
    python3 src/pady_robot/scripts/analyse_stl_bounds.py

The STLs are exported from Fusion 360 as a single assembly in absolute
CAD world coordinates (Y-up). This script prints the bounding box of each
mesh so you can identify joint positions and compute the URDF visual/collision
origin offsets.

Axis mapping (CAD → URDF):
    CAD X → URDF Y   (lateral)
    CAD Y → URDF Z   (vertical, +up)
    CAD Z → URDF X   (sagittal, +forward)

Required rotation for every visual/collision origin:
    rpy="1.5708 0 1.5708"   (Rx=90°, then Rz=90°)

Visual/collision xyz offset formula:
    Given a joint's absolute CAD position P = (Px, Py, Pz) mm, the offset is:
        URDF x = -Pz / 1000   (CAD Z → URDF X, negated)
        URDF y = -Px / 1000   (CAD X → URDF Y, negated for right leg; see note)
        URDF z = -Py / 1000   (CAD Y → URDF Z, negated)

    The offset cancels the absolute joint position so the mesh sits correctly
    relative to its link frame.

Note: for left-leg links, Px is positive (e.g. +107.86 mm) → URDF y = -0.10786
      for right-leg links, Px is negative (e.g. -107.86 mm) → URDF y = +0.10786
"""

import struct
import os

MESH_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'meshes', 'visual'
)

def read_stl_binary(path):
    """Return list of (v0, v1, v2) triangle tuples from a binary STL."""
    triangles = []
    with open(path, 'rb') as f:
        f.read(80)  # header
        count = struct.unpack('<I', f.read(4))[0]
        for _ in range(count):
            f.read(12)  # normal
            v = [struct.unpack('<3f', f.read(12)) for _ in range(3)]
            f.read(2)   # attribute
            triangles.append(v)
    return triangles


def bounding_box(triangles):
    xs, ys, zs = [], [], []
    for tri in triangles:
        for v in tri:
            xs.append(v[0]); ys.append(v[1]); zs.append(v[2])
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def main():
    stl_files = sorted(f for f in os.listdir(MESH_DIR) if f.endswith('.stl'))
    if not stl_files:
        print(f"No STL files found in {MESH_DIR}")
        return

    print(f"{'File':<22} {'X min→max (mm)':<26} {'Y min→max (mm)':<26} {'Z min→max (mm)'}")
    print('-' * 95)
    for fname in stl_files:
        path = os.path.join(MESH_DIR, fname)
        try:
            tris = read_stl_binary(path)
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounding_box(tris)
            print(f"{fname:<22} "
                  f"X: {xmin:8.2f} → {xmax:8.2f}   "
                  f"Y: {ymin:8.2f} → {ymax:8.2f}   "
                  f"Z: {zmin:8.2f} → {zmax:8.2f}")
        except Exception as e:
            print(f"{fname:<22} ERROR: {e}")

    print()
    print("Joint positions to read from bounding boxes (CAD frame, mm):")
    print("  base_link origin  : midpoint of base_link X, Y-max, Z-midpoint")
    print("  hip joints        : X = ±half hip width, Y = top of thigh, Z = 0")
    print("  knee joints       : X = ±half hip width, Y = top of shin, Z = 0")
    print("  foot joints       : X = ±half hip width, Y = bottom of shin, Z = forward offset")
    print()
    print("URDF visual/collision origin formula (for each link's joint at (Px,Py,Pz) mm):")
    print("  xyz = f\"-{Pz/1000:.4f} {-Px/1000:.4f} {-Py/1000:.4f}\"  "
          "(right leg: Px<0 so y term is positive)")
    print("  rpy = \"1.5708 0 1.5708\"  (always the same)")


if __name__ == '__main__':
    main()
