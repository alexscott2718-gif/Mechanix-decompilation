# OMedia OMT Editor + Blender Tools

Demo release for **Hot Wheels: Mechanix** (1998) OMT asset files.

## Files

| File | Purpose |
|------|---------|
| `omf2_editor.py` | Standalone GUI editor for `.omt` files |
| `blender_omt_importer.py` | Blender 4.0 add-on — import/export `.omt` levels |

---

## omf2_editor.py

A Python/Tkinter editor for OMedia Formatted Stream (`.omt`) files — the asset container used by Mechanix and other OMedia-engine games.

**Requirements:** Python 3.8+, Tkinter (usually bundled with Python), Pillow (`pip install pillow`)

**Run:**
```
python omf2_editor.py
python omf2_editor.py level1.omt
```

**Features:**
- Browse all chunk types: `3DSh` (geometry), `3DMa` (materials), `Canv` (textures), `Wave` (audio), and more
- Preview textures inline
- Preview and play audio chunks (WAV)
- Export 3DSh geometry as Wavefront OBJ + MTL + PNG textures
- Import OBJ back into a 3DSh chunk
- Import WAV files into Wave chunks (preserves original sample rate)
- Edit chunk names
- Save modified OMT files

**Supported OMT versions:** 1 (`0MFS`), 2 (`0MF2`), 3 (`0MF3`)

---

## blender_omt_importer.py  *(Blender 4.0 add-on)*

**Install:**
1. Blender → Edit → Preferences → Add-ons → Install… → pick `blender_omt_importer.py` → enable **OMedia Level Tools (.omt)**

**Import** `File → Import → OMedia Level (.omt)`
- Imports all `3DSh` chunks as separate mesh objects
- Decodes `Canv` textures and assigns to `3DMa` materials via Principled BSDF
- **Scale** (default `0.01`): game units (~cm) → metres
- **Game Up Axis** (default `Y`): axis remapping to Blender Z-up

**Export** `File → Export → OMedia Level (.omt)`
- Exports selected meshes back to game-space coordinates
- Encodes geometry as `3DSh`, materials as `3DMa`, textures as `Canv`
- Round-trips cleanly if **Import Scale Used** and **Game Up Axis** match the import settings

**Export** `File → Export → OMedia Game OBJ (game coords)`
- Writes a plain `.obj` in game coordinates, compatible with the editor's OBJ import

---

## OMT Format Notes

OMedia Formatted Stream files store typed chunks identified by 4-character FourCC codes.
The file header is at a variable offset pointed to by bytes 4–7 (big-endian uint32).

Key chunk types used by Mechanix:

| FourCC | Type | Contents |
|--------|------|----------|
| `3DSh` | Geometry | Vertices, polygons, UVs, normals, material links |
| `3DMa` | Material | Phong + optional texture reference |
| `Canv` | Texture | PackBits-compressed RGBA image |
| `Wave` | Audio | RIFF WAV audio |
| `0HDR` | Header copy | In-file copy of the chunk index (runtime optimisation) |

Coordinates use **big-endian** IEEE 754 floats. Game units are approximately 1 cm.

---

## Compatibility

Tested against Hot Wheels: Mechanix (1998, Windows) `.omt` assets.
May work with other OMedia 2.5 SDK games.
