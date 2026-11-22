"""Blender 4.0 add-on: import Mechanix / OMedia .omt level files.

Install: Edit → Preferences → Add-ons → Install… → pick this file → enable.
Use:     File → Import → OMedia Level (.omt)
"""

bl_info = {
    "name": "OMedia Level Tools (.omt)",
    "author": "mechanix-port",
    "version": (1, 1, 0),
    "blender": (4, 0, 0),
    "location": "File > Import/Export > OMedia Level (.omt)",
    "description": "Import and export Mechanix / OMedia .omt geometry",
    "category": "Import-Export",
}

import struct
import zlib
import os
import bpy
import bmesh
import mathutils
from bpy_extras.io_utils import ImportHelper, ExportHelper
from bpy.props import StringProperty, FloatProperty, BoolProperty
from bpy.types import Operator


# ---------------------------------------------------------------------------
# OMT binary parser (self-contained)
# ---------------------------------------------------------------------------

_MAGICS = {b"0MFS": 1, b"0MF2": 2, b"0MF3": 3}


def _read_pstring(data, p):
    """Pascal-style length-prefixed string."""
    n = data[p]; p += 1
    return data[p:p + n].decode("latin-1"), p + n


def _chunk_payload(raw, compression):
    if compression and len(raw) >= 4:
        return zlib.decompress(raw[4:])
    return raw


def parse_omt(data: bytes):
    """Return dict: {chunk_type: {chunk_id: (name, payload_bytes)}}"""
    magic = data[:4]
    if magic not in _MAGICS:
        raise ValueError(f"Not an OMT file (magic {magic!r})")
    version = _MAGICS[magic]
    p = struct.unpack_from(">I", data, 4)[0]           # header offset
    n_types = struct.unpack_from(">I", data, p)[0]; p += 4
    result = {}
    for _ in range(n_types):
        ctype = data[p:p + 4].decode("latin-1"); p += 4
        n_chunks = struct.unpack_from(">I", data, p)[0]; p += 4
        result.setdefault(ctype, {})
        for _ in range(n_chunks):
            cid  = struct.unpack_from(">I", data, p)[0]; p += 4
            off  = struct.unpack_from(">I", data, p)[0]; p += 4
            size = struct.unpack_from(">I", data, p)[0]; p += 4
            name = ""
            if version >= 2:
                name, p = _read_pstring(data, p)
            comp = 0
            if version >= 3:
                comp = struct.unpack_from(">H", data, p)[0]; p += 2
            raw = data[off:off + size]
            payload = _chunk_payload(raw, comp)
            result[ctype][cid] = (name, payload)
    return result


# ---------------------------------------------------------------------------
# 3DSh decoder
# ---------------------------------------------------------------------------

def decode_3dsh(d: bytes):
    """Return (vertices, faces, normals).

    vertices: list of (x, y, z)
    faces:    list of (vis, uvs, nis, face_normal, mat_ref)
              mat_ref: ("3DMa", chunk_id) or None
    normals:  list of (nx, ny, nz) smooth per-vertex normals (may be [])
    """
    if d[:4] != b"3DSP":
        return [], [], []

    class _R:
        def __init__(self, d): self.d, self.p = d, 0
        def u32(self): v = struct.unpack_from(">I", self.d, self.p)[0]; self.p += 4; return v
        def i32(self): v = struct.unpack_from(">i", self.d, self.p)[0]; self.p += 4; return v
        def u16(self): v = struct.unpack_from(">H", self.d, self.p)[0]; self.p += 2; return v
        def i16(self): v = struct.unpack_from(">h", self.d, self.p)[0]; self.p += 2; return v
        def f32(self): v = struct.unpack_from(">f", self.d, self.p)[0]; self.p += 4; return v
        def f32n(self, n): v = struct.unpack_from(f">{n}f", self.d, self.p); self.p += 4 * n; return v
        def bool16(self): return self.i16() != 0
        def link(self):
            if not self.bool16(): return None
            t = self.d[self.p:self.p + 4].decode("latin-1"); self.p += 4
            return (t, self.i32())

    s = _R(d); s.p = 4
    version = s.u32()
    if version > 5:
        return [], [], []
    s.f32n(4); s.f32n(4)
    if 0 < version < 3:
        s.f32(); s.bool16()
    vertices = [s.f32n(3) for _ in range(s.u32())]
    faces = []
    for _ in range(s.u32()):
        try:
            if version > 2:
                s.u16(); s.u32()
                vis, uvs, nis = [], [], []
                for _ in range(s.i32()):
                    vi, ni, ci, u, v = s.u32(), s.u32(), s.u32(), s.f32(), s.f32()
                    if version > 4:
                        for _ in range(s.u32()): s.f32(); s.f32()
                    vis.append(vi); uvs.append((u, v)); nis.append(ni)
                normal = s.f32n(3)
                mat = s.link()
                if version > 4: s.i32()
            else:
                s.bool16(); s.bool16(); s.i16(); s.u32()
                if version > 0: s.f32(); s.f32()
                s.i16()
                vis, uvs, nis = [], [], []
                for _ in range(s.u16()):
                    vi = s.u32(); u = s.f32(); v = s.f32()
                    ni = s.u32() if version > 1 else vi
                    vis.append(vi); uvs.append((u, v)); nis.append(ni)
                normal = s.f32n(3)
                mat = s.link() if s.bool16() else None
            if len(vis) >= 3:
                faces.append((vis, uvs, nis, normal, mat))
        except Exception:
            break
    normals = []
    try:
        n_spheres = s.u32(); s.p += n_spheres * 16
        normals = [s.f32n(3) for _ in range(s.u32())]
    except Exception:
        pass
    return vertices, faces, normals


# ---------------------------------------------------------------------------
# 3DMa decoder (diffuse + texture link)
# ---------------------------------------------------------------------------

def decode_3dma(d: bytes):
    """Return dict with diffuse (0-1 float tuple), alpha, texture chunk id or None."""
    result = {"diffuse": (0.7, 0.7, 0.7), "alpha": 1.0, "texture_id": None}
    try:
        p = 2  # skip version
        mtype = struct.unpack_from(">h", d, p)[0]; p += 2
        if mtype > 2:
            return result
        result["alpha"] = struct.unpack_from(">f", d, p)[0]; p += 4
        p += 2  # nofog
        if mtype == 0:
            r, g, b = struct.unpack_from(">HHH", d, p)
            result["diffuse"] = (r / 65535, g / 65535, b / 65535)
        else:
            p += 6  # emission
            r, g, b = struct.unpack_from(">HHH", d, p); p += 6
            result["diffuse"] = (r / 65535, g / 65535, b / 65535)
            p += 6  # specular
            p += 6  # ambient
            p += 2  # gouraud
            if mtype == 2:
                p += 2  # shaded
                has_tex = struct.unpack_from(">h", d, p)[0]; p += 2
                if has_tex == 1:
                    filled = struct.unpack_from(">h", d, p)[0]; p += 2
                    if filled:
                        p += 4  # type tag ("Canv")
                        result["texture_id"] = struct.unpack_from(">i", d, p)[0]
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Canv texture decoder
# ---------------------------------------------------------------------------

def decode_canv(d: bytes):
    """Return (width, height, rgba_bytes) or raise."""
    if d[:4] != b"OmCv": raise ValueError("Not OmCv")
    if d[8:12] != b"OmGW": raise ValueError("No OmGW header")
    p = 12
    width, height, depth = struct.unpack_from(">HHH", d, p); p += 6
    compression, version = d[p], d[p + 1]; p += 2
    nbytes = struct.unpack_from(">H", d, p)[0]; p += 2
    dotransparent = struct.unpack_from(">h", d, p)[0]; p += 2
    if version == 0: raise NotImplementedError("v0 canvas")
    transp = 0xFFFFFFFF
    if dotransparent: transp = struct.unpack_from(">I", d, p)[0]; p += 4
    dopalette = struct.unpack_from(">h", d, p)[0]; p += 2
    palette = None
    if dopalette:
        hdr4 = d[p:p + 4]; p += 4
        c1, c2 = d[p], d[p + 1]; p += 2
        if c1 == 0xFF and c2 == 0xFF:
            n = struct.unpack_from(">i", d, p)[0]; p += 4 + n
        else:
            p += c1 - 1
        size = struct.unpack_from(">I", d, p)[0]; p += 4
        comps = struct.unpack_from(f">{size * 3}H", d, p); p += size * 6
        palette = [(comps[i] >> 8, comps[i + 1] >> 8, comps[i + 2] >> 8)
                   for i in range(0, size * 3, 3)]
        if hdr4 == b"OPa2": p += 4096

    unit = {0: 1, 1: 2, 2: 4}[compression]
    bufsize = struct.unpack_from(">I", d, p)[0]; p += 4
    buf = d[p:p + bufsize]

    def unpack_row(src, unit, dst_bytes):
        out = bytearray(); dq = dst_bytes // unit; q = 0
        while dq > 0:
            ctl = src[q:q + unit]; q += unit
            n = struct.unpack("<b", ctl)[0] if unit == 1 else struct.unpack("<h", ctl[:2])[0]
            if n >= 0:
                n += 1; dq -= n; out += src[q:q + n * unit]; q += n * unit
            elif n != -128:
                n = 1 - n; dq -= n; out += src[q:q + unit] * n; q += unit
        return bytes(out)

    rows = []; q = 0
    for _ in range(height):
        n = struct.unpack_from(">h", buf, q)[0]; q += 2
        rows.append(unpack_row(buf[q:q + n], unit, nbytes)); q += n

    out = bytearray(width * height * 4); o = 0
    for row in rows:
        for x in range(width):
            if depth == 16:
                v = (row[2*x] << 8) | row[2*x+1]
                r, g, b = ((v >> 10) & 31) << 3, ((v >> 5) & 31) << 3, (v & 31) << 3
            elif depth == 32:
                v = struct.unpack_from(">I", row, 4*x)[0]
                r, g, b = (v >> 16) & 255, (v >> 8) & 255, v & 255
            elif depth == 8:
                v = row[x]; r, g, b = palette[v]
            elif depth == 4:
                v = row[x >> 1]; v = ((v >> 4) if (x & 1) else v) & 15; r, g, b = palette[v]
            elif depth == 1:
                v = (row[x >> 3] >> (7 - (x & 7))) & 1; r, g, b = palette[v]
            else:
                raise ValueError(f"Unknown depth {depth}")
            a = 0 if (dotransparent and v == transp) else 255
            out[o:o + 4] = bytes((r, g, b, a)); o += 4
    return width, height, bytes(out)


# ---------------------------------------------------------------------------
# Blender import
# ---------------------------------------------------------------------------

def _make_blender_image(name, width, height, rgba_bytes):
    img = bpy.data.images.new(name, width=width, height=height, alpha=True)
    # Blender expects float RGBA, bottom-to-top row order
    n = width * height
    pixels = [0.0] * (n * 4)
    for y in range(height):
        src_row = (height - 1 - y) * width * 4
        dst_row = y * width * 4
        for x in range(width):
            s = src_row + x * 4
            d = dst_row + x * 4
            pixels[d]   = rgba_bytes[s]   / 255.0
            pixels[d+1] = rgba_bytes[s+1] / 255.0
            pixels[d+2] = rgba_bytes[s+2] / 255.0
            pixels[d+3] = rgba_bytes[s+3] / 255.0
    img.pixels = pixels
    img.pack()
    return img


def _make_blender_material(name, mat_info, canv_image):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        mat.node_tree.nodes.clear()
        bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
        out  = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
        mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    r, g, b = mat_info["diffuse"]
    a = mat_info["alpha"]
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    bsdf.inputs["Roughness"].default_value  = 0.8
    bsdf.inputs["Specular IOR Level"].default_value = 0.1

    if canv_image:
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.image = canv_image
        tex_node.location = (-300, 300)
        links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
        if canv_image.alpha_mode != 'NONE':
            links.new(tex_node.outputs["Alpha"], bsdf.inputs["Alpha"])
            mat.blend_method = 'CLIP'

    if a < 0.99:
        mat.blend_method = 'BLEND'
        bsdf.inputs["Alpha"].default_value = a

    return mat


def import_omt(filepath, scale=0.01, import_textures=True, axis_up="Y"):
    data = open(filepath, "rb").read()
    chunks = parse_omt(data)

    # Build Blender images from Canv chunks
    canv_images = {}
    if import_textures:
        for cid, (name, payload) in chunks.get("Canv", {}).items():
            try:
                w, h, rgba = decode_canv(payload)
                img_name = name or f"canv_{cid}"
                canv_images[cid] = _make_blender_image(img_name, w, h, rgba)
            except Exception as e:
                print(f"[omt-import] Canv {cid} ({name}): {e}")

    # Build Blender materials from 3DMa chunks
    blender_mats = {}
    for cid, (name, payload) in chunks.get("3DMa", {}).items():
        mat_info = decode_3dma(payload)
        tex_id   = mat_info.get("texture_id")
        img      = canv_images.get(tex_id) if tex_id is not None else None
        mat_name = name or f"mat_{cid}"
        blender_mats[cid] = _make_blender_material(mat_name, mat_info, img)

    # Build Blender meshes from 3DSh chunks
    collection = bpy.context.collection
    base_name  = os.path.splitext(os.path.basename(filepath))[0]

    mesh_chunks = chunks.get("3DSh", {})
    for cid, (name, payload) in mesh_chunks.items():
        verts, faces, normals = decode_3dsh(payload)
        if not verts or not faces:
            continue

        # Scale and remap axes.
        if axis_up == "Y":   # OMedia Y-up → Blender Z-up
            sv = [(x * scale, -z * scale, y * scale) for x, y, z in verts]
        elif axis_up == "X":
            sv = [(y * scale, z * scale, x * scale) for x, y, z in verts]
        else:                # Z-up: raw game coordinates
            sv = [(x * scale, y * scale, z * scale) for x, y, z in verts]

        # Build bmesh
        bm = bmesh.new()
        bm_verts = [bm.verts.new(v) for v in sv]
        bm.verts.ensure_lookup_table()

        uv_layer = bm.loops.layers.uv.new("UVMap")

        # Collect unique material ids in order of first appearance
        mat_ids_ordered = []
        seen_mats = set()
        for vis, uvs, nis, fn, mat_ref in faces:
            mid = mat_ref[1] if (mat_ref and mat_ref[0] == "3DMa") else None
            if mid not in seen_mats:
                seen_mats.add(mid); mat_ids_ordered.append(mid)

        mat_slot_index = {mid: i for i, mid in enumerate(mat_ids_ordered)}

        for vis, uvs, nis, fn, mat_ref in faces:
            # Fan-triangulate polygon
            for k in range(1, len(vis) - 1):
                tri = (vis[0], vis[k], vis[k + 1])
                try:
                    bm_face = bm.faces.new([bm_verts[i] for i in tri])
                except ValueError:
                    continue  # duplicate / degenerate
                mid = mat_ref[1] if (mat_ref and mat_ref[0] == "3DMa") else None
                bm_face.material_index = mat_slot_index.get(mid, 0)
                tri_uvs = [uvs[0], uvs[k], uvs[k + 1]]
                for loop, (u, v) in zip(bm_face.loops, tri_uvs):
                    loop[uv_layer].uv = (u, 1.0 - v)

        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        mesh = bpy.data.meshes.new(name or f"{base_name}_{cid}")
        bm.to_mesh(mesh); bm.free()
        mesh.update()

        obj = bpy.data.objects.new(mesh.name, mesh)
        collection.objects.link(obj)

        for mid in mat_ids_ordered:
            obj.data.materials.append(blender_mats.get(mid))

    return {"FINISHED"}


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class OmtImportOperator(Operator, ImportHelper):
    bl_idname = "import_scene.omt"
    bl_label  = "Import OMedia Level (.omt)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".omt"
    filter_glob: StringProperty(default="*.omt", options={"HIDDEN"})

    scale: FloatProperty(
        name="Scale",
        description="Multiply coordinates (default 0.01: game units → metres)",
        default=0.01, min=1e-6, max=100.0,
    )
    axis_up: bpy.props.EnumProperty(
        name="Game Up Axis",
        description="Which game axis maps to Blender Z (up). Try Y first.",
        items=[
            ("Y", "Y up  (game Y → Blender Z)", "Standard OMedia Y-up"),
            ("Z", "Z up  (no conversion)",       "Raw game coordinates"),
            ("X", "X up  (game X → Blender Z)", "Uncommon"),
        ],
        default="Y",
    )
    import_textures: BoolProperty(
        name="Import Textures",
        description="Decode Canv chunks and assign to materials",
        default=True,
    )

    def execute(self, context):
        try:
            import_omt(self.filepath, scale=self.scale,
                       import_textures=self.import_textures,
                       axis_up=self.axis_up)
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Imported {os.path.basename(self.filepath)}")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# OMT encoders (self-contained copies from omf2_editor)
# ---------------------------------------------------------------------------

def _ws(s):
    b = s.encode("latin-1")
    return bytes([len(b)]) + b


def _build_omt(pairs):
    """pairs: list of (type_str, id_int, name_str, payload_bytes)"""
    MAGIC = b"0MF2"
    out = bytearray(MAGIC + b"\x00\x00\x00\x00")
    placed = []
    for ctype, cid, cname, payload in pairs:
        off = len(out); out += payload
        placed.append((ctype, cid, cname, off, len(payload)))
    hdr_off = len(out)
    struct.pack_into(">I", out, 4, hdr_off)
    by_type = {}
    for c in placed:
        by_type.setdefault(c[0], []).append(c)
    hdr = bytearray()
    hdr += struct.pack(">I", len(by_type))
    for ctype, clist in by_type.items():
        hdr += ctype.encode("latin-1") + struct.pack(">I", len(clist))
        for ctype2, cid, cname, off, size in clist:
            hdr += struct.pack(">III", cid, off, size)
            hdr += _ws(cname)
    hdr += struct.pack(">I", 0)
    out += hdr
    return bytes(out)


def _packbits32(row_bytes):
    unit = 4
    n = len(row_bytes) // unit
    out = bytearray()
    i = 0
    while i < n:
        j = i + 1
        while j < n and j - i < 32767 and row_bytes[j*4:(j+1)*4] == row_bytes[i*4:(i+1)*4]:
            j += 1
        if j - i > 1:
            out += struct.pack("<h", -(j - i - 1)) + b"\x00\x00"
            out += row_bytes[i*4:(i+1)*4]
            i = j; continue
        k = i + 1
        while k < n and k - i < 32767:
            if k + 1 < n and row_bytes[k*4:(k+1)*4] == row_bytes[(k+1)*4:(k+2)*4]:
                break
            k += 1
        out += struct.pack("<h", k - i - 1) + b"\x00\x00"
        out += row_bytes[i*4:k*4]
        i = k
    return bytes(out)


def _encode_canv(rgba, width, height):
    TKEY = 0x00FF00FF
    TKEY_B = struct.pack(">I", TKEY)
    has_alpha = any(rgba[i*4+3] < 128 for i in range(width * height))
    rows = bytearray()
    for y in range(height):
        pix = bytearray(width * 4)
        for x in range(width):
            o = (y * width + x) * 4
            r, g, b, a = rgba[o], rgba[o+1], rgba[o+2], rgba[o+3]
            if has_alpha and a < 128:
                pix[x*4:(x+1)*4] = TKEY_B
            else:
                struct.pack_into(">I", pix, x * 4, (r << 16) | (g << 8) | b)
        enc = _packbits32(bytes(pix))
        rows += struct.pack(">h", len(enc)) + enc
    omgw = bytearray()
    omgw += struct.pack(">HHH", width, height, 32)
    omgw += bytes([2, 1])
    omgw += struct.pack(">H", width * 4)
    omgw += struct.pack(">h", 1 if has_alpha else 0)
    if has_alpha:
        omgw += struct.pack(">I", TKEY)
    omgw += struct.pack(">h", 0)
    omgw += struct.pack(">I", len(rows)) + rows
    return b"OmCv\x00\x00\x10\x0cOmGW" + bytes(omgw)


def _encode_3dma(diffuse=(255,255,255), *, alpha=0.0, emission=(255,255,255),
                 specular=(0,0,0), ambient=(0,0,0), gouraud=False,
                 shaded=True, nofog=False, texture_id=None):
    def _rgb(r, g, b): return struct.pack(">HHH", r<<8, g<<8, b<<8)
    out = bytearray()
    mtype = 2 if texture_id is not None else 1
    out += struct.pack(">hh", 1, mtype)
    out += struct.pack(">f",  alpha)
    out += struct.pack(">h",  1 if nofog else 0)
    out += _rgb(*emission) + _rgb(*diffuse) + _rgb(*specular) + _rgb(*ambient)
    out += struct.pack(">h", 1 if gouraud else 0)
    if texture_id is not None:
        out += struct.pack(">h", 1 if shaded else 0)
        out += struct.pack(">hh", 1, 1)
        out += b"Canv" + struct.pack(">i", texture_id)
    out += struct.pack(">h", 3)
    return bytes(out)


def _encode_3dsh(verts, faces, normals=None):
    import math as _m
    if verts:
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        cz = sum(v[2] for v in verts) / len(verts)
        r  = max(_m.sqrt((v[0]-cx)**2+(v[1]-cy)**2+(v[2]-cz)**2) for v in verts)
    else:
        cx = cy = cz = r = 0.0
    out = bytearray()
    out += b"3DSP" + struct.pack(">I", 3)
    out += struct.pack(">4f", cx, cy, cz, r)
    out += struct.pack(">4f", cx, cy, cz, r)
    out += struct.pack(">I", len(verts))
    for x, y, z in verts:
        out += struct.pack(">fff", x, y, z)
    out += struct.pack(">I", len(faces))
    for vis, uvs, nis, fn, mat in faces:
        ni_list = nis if nis else vis
        out += struct.pack(">HI", 0, 0) + struct.pack(">i", len(vis))
        for k in range(len(vis)):
            out += struct.pack(">IIIff", vis[k], ni_list[k], 0, uvs[k][0], uvs[k][1])
        out += struct.pack(">fff", fn[0], fn[1], fn[2])
        if mat and mat[0] == "3DMa":
            out += struct.pack(">h", 1) + b"3DMa" + struct.pack(">i", mat[1])
        else:
            out += struct.pack(">h", 0)
    out += struct.pack(">I", 0)
    norms = normals or []
    out += struct.pack(">I", len(norms))
    for nx, ny, nz in norms:
        out += struct.pack(">fff", nx, ny, nz)
    out += struct.pack(">I", 0)
    return bytes(out)


# ---------------------------------------------------------------------------
# OBJ export back to game coordinates
# ---------------------------------------------------------------------------

def export_game_obj(filepath, objects, scale=0.01, axis_up="Y"):
    """Write selected Blender mesh objects to a Wavefront OBJ in game coordinates.

    Reverses the import axis mapping so the file can be read back by the
    omf2_editor's OBJ importer (which expects raw game-space coordinates).

    Inverse of the import transforms:
      Y-up:  game(x,y,z) = (bx/scale,  bz/scale, -by/scale)
      Z-up:  game(x,y,z) = (bx/scale,  by/scale,  bz/scale)
    """
    inv = 1.0 / scale
    lines = []
    vert_offset = 1  # OBJ vertex indices are 1-based, global across objects

    for obj in objects:
        if obj.type != 'MESH':
            continue

        # Apply all modifiers / object transforms
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj  = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()

        # World-space vertex positions
        wm = obj.matrix_world
        raw_verts = [wm @ v.co for v in mesh.vertices]

        # Remap to game space
        if axis_up == "Y":
            gverts = [(v.x * inv, v.z * inv, -v.y * inv) for v in raw_verts]
        elif axis_up == "X":
            gverts = [(v.z * inv, v.x * inv, v.y * inv) for v in raw_verts]
        else:
            gverts = [(v.x * inv, v.y * inv, v.z * inv) for v in raw_verts]

        lines.append(f"o {obj.name}\n")
        for x, y, z in gverts:
            lines.append(f"v {x:.6f} {y:.6f} {z:.6f}\n")

        # UVs
        uv_map  = mesh.uv_layers.active
        uv_list = []
        uv_idx  = {}
        if uv_map:
            for poly in mesh.polygons:
                for li in poly.loop_indices:
                    uv = mesh.uv_layers.active.data[li].uv
                    key = (round(uv.x, 6), round(1.0 - uv.y, 6))  # flip V back
                    if key not in uv_idx:
                        uv_idx[key] = len(uv_list) + 1
                        uv_list.append(key)
            for u, v in uv_list:
                lines.append(f"vt {u:.6f} {v:.6f}\n")

        # Smooth normals
        mesh.calc_normals_split()
        norm_list = []
        norm_idx  = {}
        for poly in mesh.polygons:
            for li in poly.loop_indices:
                loop = mesh.loops[li]
                n    = loop.normal
                if axis_up == "Y":
                    key = (round(n.x, 4), round(n.z, 4), round(-n.y, 4))
                elif axis_up == "X":
                    key = (round(n.z, 4), round(n.x, 4), round(n.y, 4))
                else:
                    key = (round(n.x, 4), round(n.y, 4), round(n.z, 4))
                if key not in norm_idx:
                    norm_idx[key] = len(norm_list) + 1
                    norm_list.append(key)
        for nx, ny, nz in norm_list:
            lines.append(f"vn {nx:.4f} {ny:.4f} {nz:.4f}\n")

        # Faces grouped by material
        mat_faces = {}
        for poly in mesh.polygons:
            mat_faces.setdefault(poly.material_index, []).append(poly)

        cur_mat = object()
        for mat_idx, polys in sorted(mat_faces.items()):
            if mat_idx != cur_mat:
                cur_mat = mat_idx
                mat = obj.data.materials[mat_idx] if mat_idx < len(obj.data.materials) else None
                mname = mat.name if mat else "default"
                lines.append(f"usemtl {mname}\n")
            for poly in polys:
                tokens = []
                for li in poly.loop_indices:
                    loop  = mesh.loops[li]
                    vi    = loop.vertex_index + vert_offset
                    if uv_map:
                        uv  = mesh.uv_layers.active.data[li].uv
                        key = (round(uv.x, 6), round(1.0 - uv.y, 6))
                        vti = uv_idx.get(key, 0)
                    else:
                        vti = 0
                    n    = loop.normal
                    if axis_up == "Y":
                        nkey = (round(n.x, 4), round(n.z, 4), round(-n.y, 4))
                    elif axis_up == "X":
                        nkey = (round(n.z, 4), round(n.x, 4), round(n.y, 4))
                    else:
                        nkey = (round(n.x, 4), round(n.y, 4), round(n.z, 4))
                    ni = norm_idx.get(nkey, 0)
                    tokens.append(f"{vi}/{vti}/{ni}" if vti else f"{vi}//{ni}")
                lines.append("f " + " ".join(tokens) + "\n")

        vert_offset += len(mesh.vertices)
        eval_obj.to_mesh_clear()

    with open(filepath, "w") as f:
        f.writelines(lines)


class OmtExportOperator(Operator, ExportHelper):
    bl_idname  = "export_scene.omt_obj"
    bl_label   = "Export OMedia Game OBJ"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".obj"
    filter_glob: StringProperty(default="*.obj", options={"HIDDEN"})

    scale: FloatProperty(
        name="Import Scale Used",
        description="The scale factor used when importing (default 0.01). "
                    "Coordinates are divided by this to restore game units.",
        default=0.01, min=1e-6, max=100.0,
    )
    axis_up: bpy.props.EnumProperty(
        name="Game Up Axis",
        description="Must match what was used on import",
        items=[
            ("Y", "Y up  (OMedia standard)", ""),
            ("Z", "Z up  (raw / no conversion)", ""),
            ("X", "X up", ""),
        ],
        default="Y",
    )
    selected_only: BoolProperty(
        name="Selected Only",
        description="Export only selected objects (default: all mesh objects)",
        default=True,
    )

    def execute(self, context):
        if self.selected_only:
            objects = [o for o in context.selected_objects if o.type == 'MESH']
        else:
            objects = [o for o in context.scene.objects if o.type == 'MESH']
        if not objects:
            self.report({"WARNING"}, "No mesh objects to export")
            return {"CANCELLED"}
        try:
            export_game_obj(self.filepath, objects,
                            scale=self.scale, axis_up=self.axis_up)
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported {len(objects)} object(s) → {os.path.basename(self.filepath)}")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Export Blender scene → .omt
# ---------------------------------------------------------------------------

def _blender_mesh_to_game(obj, scale, axis_up):
    """Return (verts, faces, normals, mat_info_list) in game coordinates.

    mat_info_list: list of dicts with keys diffuse(0-255 tuple), texture(image or None)
    ordered by material slot index.
    """
    import math as _m

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj  = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    mesh.calc_normals_split()

    inv = 1.0 / scale
    wm  = obj.matrix_world

    # Vertices in game space
    raw = [wm @ v.co for v in mesh.vertices]
    if axis_up == "Y":
        gverts = [(v.x * inv, v.z * inv, -v.y * inv) for v in raw]
        def remap_n(n): return (n.x, n.z, -n.y)
    elif axis_up == "X":
        gverts = [(v.z * inv, v.x * inv, v.y * inv) for v in raw]
        def remap_n(n): return (n.z, n.x, n.y)
    else:
        gverts = [(v.x * inv, v.y * inv, v.z * inv) for v in raw]
        def remap_n(n): return (n.x, n.y, n.z)

    # Build smooth normal list (deduplicated)
    norm_map = {}; norm_list = []
    for li in range(len(mesh.loops)):
        n = mesh.loops[li].normal
        key = (round(n.x,4), round(n.y,4), round(n.z,4))
        if key not in norm_map:
            norm_map[key] = len(norm_list)
            rn = remap_n(n)
            norm_list.append(rn)

    # UVs (may be None)
    uv_layer = mesh.uv_layers.active

    # Build faces (one polygon per face, fan-triangulated into quads/tris as polygons)
    faces = []
    for poly in mesh.polygons:
        vis  = list(poly.vertices)
        uvs  = []
        nis  = []
        for li in poly.loop_indices:
            loop = mesh.loops[li]
            if uv_layer:
                u, v = uv_layer.data[li].uv
                uvs.append((u, 1.0 - v))   # flip V back to game convention
            else:
                uvs.append((0.0, 0.0))
            n   = loop.normal
            key = (round(n.x,4), round(n.y,4), round(n.z,4))
            nis.append(norm_map[key])
        # face normal in game space
        fn_b = poly.normal
        fn   = remap_n(fn_b)
        mat_ref = ("3DMa", poly.material_index) if poly.material_index < len(obj.data.materials) else None
        faces.append((vis, uvs, nis, fn, mat_ref))

    # Material info
    mat_info_list = []
    for mat_slot in obj.data.materials:
        info = {"diffuse": (180, 180, 180), "texture": None}
        if mat_slot and mat_slot.use_nodes:
            bsdf = mat_slot.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bc = bsdf.inputs["Base Color"].default_value
                info["diffuse"] = (int(bc[0]*255), int(bc[1]*255), int(bc[2]*255))
                # check for image texture linked to base color
                for link in mat_slot.node_tree.links:
                    if (link.to_node == bsdf
                            and link.to_socket.name == "Base Color"
                            and link.from_node.type == "TEX_IMAGE"):
                        info["texture"] = link.from_node.image
        mat_info_list.append(info)

    eval_obj.to_mesh_clear()
    return gverts, faces, norm_list, mat_info_list


def export_omt(filepath, objects, scale=0.01, axis_up="Y", export_textures=True):
    pairs = []   # (type, id, name, payload)
    canv_id  = 0
    mat_id   = 0
    shape_id = 0

    for obj in objects:
        if obj.type != 'MESH':
            continue

        gverts, faces, normals, mat_info_list = _blender_mesh_to_game(obj, scale, axis_up)
        if not gverts:
            continue

        # For each material slot: encode Canv + 3DMa, then remap face mat_ref to real ids
        slot_to_mat_id = {}
        for slot_idx, info in enumerate(mat_info_list):
            tex_chunk_id = None
            if export_textures and info["texture"] is not None:
                img = info["texture"]
                w, h = img.size
                # get RGBA bytes from Blender image (float → uint8, bottom-to-top → top-to-bottom)
                px = list(img.pixels)
                rgba = bytearray(w * h * 4)
                for y in range(h):
                    src_row = (h - 1 - y) * w * 4
                    dst_row = y * w * 4
                    for x in range(w):
                        for c in range(4):
                            rgba[dst_row + x*4 + c] = min(255, int(px[src_row + x*4 + c] * 255))
                payload = _encode_canv(bytes(rgba), w, h)
                pairs.append(("Canv", canv_id, img.name[:255], payload))
                tex_chunk_id = canv_id
                canv_id += 1

            r, g, b = info["diffuse"]
            ma_payload = _encode_3dma(
                diffuse=(r, g, b),
                emission=(r//4, g//4, b//4),
                ambient=(r//6, g//6, b//6),
                gouraud=True,
                texture_id=tex_chunk_id,
            )
            pairs.append(("3DMa", mat_id, f"mat_{obj.name}_{slot_idx}"[:255], ma_payload))
            slot_to_mat_id[slot_idx] = mat_id
            mat_id += 1

        # Remap material indices in faces to real 3DMa chunk ids
        remapped = []
        for vis, uvs, nis, fn, mat_ref in faces:
            if mat_ref and mat_ref[0] == "3DMa":
                real_id = slot_to_mat_id.get(mat_ref[1])
                mat_ref = ("3DMa", real_id) if real_id is not None else None
            remapped.append((vis, uvs, nis, fn, mat_ref))

        sh_payload = _encode_3dsh(gverts, remapped, normals)
        pairs.append(("3DSh", shape_id, obj.name[:255], sh_payload))
        shape_id += 1

    omt_bytes = _build_omt(pairs)
    open(filepath, "wb").write(omt_bytes)


class OmtExportOmtOperator(Operator, ExportHelper):
    bl_idname  = "export_scene.omt_omt"
    bl_label   = "Export OMedia Level (.omt)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".omt"
    filter_glob: StringProperty(default="*.omt", options={"HIDDEN"})

    scale: FloatProperty(
        name="Import Scale Used",
        description="Scale factor used when importing — coordinates are divided by this",
        default=0.01, min=1e-6, max=100.0,
    )
    axis_up: bpy.props.EnumProperty(
        name="Game Up Axis",
        description="Must match the axis setting used on import",
        items=[
            ("Y", "Y up  (OMedia standard)", ""),
            ("Z", "Z up  (raw / no conversion)", ""),
            ("X", "X up", ""),
        ],
        default="Y",
    )
    export_textures: BoolProperty(
        name="Export Textures",
        description="Encode image textures as Canv chunks inside the OMT",
        default=True,
    )
    selected_only: BoolProperty(
        name="Selected Only",
        description="Export only selected objects",
        default=True,
    )

    def execute(self, context):
        objects = (context.selected_objects if self.selected_only
                   else context.scene.objects)
        objects = [o for o in objects if o.type == 'MESH']
        if not objects:
            self.report({"WARNING"}, "No mesh objects to export")
            return {"CANCELLED"}
        try:
            export_omt(self.filepath, objects,
                       scale=self.scale, axis_up=self.axis_up,
                       export_textures=self.export_textures)
        except Exception as e:
            self.report({"ERROR"}, str(e))
            import traceback; traceback.print_exc()
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported {len(objects)} mesh(es) → {os.path.basename(self.filepath)}")
        return {"FINISHED"}


def menu_func_import(self, context):
    self.layout.operator(OmtImportOperator.bl_idname, text="OMedia Level (.omt)")


def menu_func_export_obj(self, context):
    self.layout.operator(OmtExportOperator.bl_idname, text="OMedia Game OBJ (game coords)")


def menu_func_export_omt(self, context):
    self.layout.operator(OmtExportOmtOperator.bl_idname, text="OMedia Level (.omt)")


def register():
    bpy.utils.register_class(OmtImportOperator)
    bpy.utils.register_class(OmtExportOperator)
    bpy.utils.register_class(OmtExportOmtOperator)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export_obj)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export_omt)


def unregister():
    bpy.utils.unregister_class(OmtImportOperator)
    bpy.utils.unregister_class(OmtExportOperator)
    bpy.utils.unregister_class(OmtExportOmtOperator)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export_obj)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export_omt)


if __name__ == "__main__":
    register()
