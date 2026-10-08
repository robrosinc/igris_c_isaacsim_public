# IGRIS-C appearance pass

The default wrist/hand-independent robot configuration loads
`from_urdf/usd/extras/wrist_hand_independent/igris_c_colored.usda`. That layer
composes the existing robot USD, adds visual material bindings, and includes
`igris_c_face_outer.usdc` and `igris_c_torso_outer.usdc`. The original USD,
joints, collisions, and inertias
are not rewritten.

`materials.mtl` stores the photorealistic five-finger material palette used by
the appearance tools. `link_materials.json` assigns materials to the public visual links.
The arms, neck-yaw link, waist links, hip-yaw (upper thigh), knees, and ankle-pitch
links use `mat_0`: white with metallic 0.8 and roughness 0.4. The four hip-pitch
and hip-roll links retain `mat_1`: gray with metallic 0.8 and roughness 0.4.
The ankle-roll links, whose meshes include the feet, use `mat_3`: silver
with metallic 1.0 and roughness 0.1.
The full source visual has several materials within most links; this pass
uses one material per public link, except for the selected head and torso.

Run the side-by-side Viser preview from the repository root in a separate
Python environment:

```bash
python -m pip install viser trimesh numpy usd-core
python scripts/tools/visualize_igris_c_appearance.py --port 8080
```

Open `http://127.0.0.1:8080`. The preview uses the public URDF's STL visuals
at zero joint angles and reads face and torso materials from the composed painted USD.
Lighting and transparency may differ from Isaac Sim. To validate files without
starting a server, add `--check`.

To rebuild the material layer after changing the mapping or palette, run
`python scripts/tools/build_igris_c_appearance.py` in that environment.
