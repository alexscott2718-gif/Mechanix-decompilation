#!/usr/bin/env python3
"""OMF2 / OMT file editor — read, edit and build OMedia formatted stream (.omt) files.

Usage:
  python omf2_editor.py [file.omt]

Requires: Python 3.9+, tkinter (standard library).
Optional: Pillow (pip install Pillow) for canvas image preview.

Format reference (OMediaFormattedStream.cpp, OMT 2.5 SDK):
  Magic "0MFS"/"0MF2"/"0MF3" (v1/v2/v3), big-endian u32 header_offset,
  chunk payloads, then header: type table, free list.
  v3 chunks may be zlib-compressed (i32 uncompressed_size + deflate stream).
"""
import math
import os
import sys
import struct
import zlib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False
    DND_FILES = None
    TkinterDnD = None

# Allow importing game / omt packages when running from the tools/ subdirectory
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_TOOLS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Optional C rasterizer (gcc compiled at startup for full-res smooth rendering)
# ---------------------------------------------------------------------------

_C_RASTER_SRC = r"""
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* Flat-shaded painter's algorithm — triangles already sorted back-to-front */
void raster_flat(
    unsigned char* fb, int W, int H,
    const float* sx, const float* sy,   /* (N,3) screen coords */
    const unsigned char* rgb,            /* (N,3) shaded colour */
    int N
) {
    for (int i = 0; i < N; i++) {
        float x0=sx[i*3],y0=sy[i*3],x1=sx[i*3+1],y1=sy[i*3+1],x2=sx[i*3+2],y2=sy[i*3+2];
        int bx0=(int)fmaxf(0.f,fminf(x0,fminf(x1,x2)));
        int bx1=(int)fminf(W-1,fmaxf(x0,fmaxf(x1,x2)))+1;
        int by0=(int)fmaxf(0.f,fminf(y0,fminf(y1,y2)));
        int by1=(int)fminf(H-1,fmaxf(y0,fmaxf(y1,y2)))+1;
        float denom=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2);
        if(fabsf(denom)<1e-6f) continue;
        float inv_d=1.f/denom;
        unsigned char r=rgb[i*3],g=rgb[i*3+1],b=rgb[i*3+2];
        for(int py=by0;py<by1;py++){
            for(int px=bx0;px<bx1;px++){
                float fx=px+.5f,fy=py+.5f;
                float w0=((y1-y2)*(fx-x2)+(x2-x1)*(fy-y2))*inv_d;
                float w1=((y2-y0)*(fx-x2)+(x0-x2)*(fy-y2))*inv_d;
                if(w0>=0.f&&w1>=0.f&&(w0+w1)<=1.f){
                    int p=(py*W+px)*3;
                    fb[p]=r; fb[p+1]=g; fb[p+2]=b;
                }
            }
        }
    }
}

/* UV-textured z-buffered rasterizer (single RGBA texture, all triangles) */
void raster_tex(
    unsigned char* fb, float* zbuf, int W, int H,
    const float* sx, const float* sy,  /* (N,3) */
    const float* inv_z,                /* (N,3) 1/z_view per vertex */
    const float* uv,                   /* (N,6) u0,v0,u1,v1,u2,v2 */
    const unsigned char* tex, int TW, int TH, /* RGBA texture */
    const float* shade,                /* (N,) per-triangle shade [0..1] */
    int N
) {
    for (int i = 0; i < N; i++) {
        float x0=sx[i*3],y0=sy[i*3],x1=sx[i*3+1],y1=sy[i*3+1],x2=sx[i*3+2],y2=sy[i*3+2];
        float iz0=inv_z[i*3],iz1=inv_z[i*3+1],iz2=inv_z[i*3+2];
        float u0=uv[i*6],v0=uv[i*6+1],u1=uv[i*6+2],v1=uv[i*6+3],u2=uv[i*6+4],v2=uv[i*6+5];
        float sh=shade[i];
        int bx0=(int)fmaxf(0.f,fminf(x0,fminf(x1,x2)));
        int bx1=(int)fminf(W-1,fmaxf(x0,fmaxf(x1,x2)))+1;
        int by0=(int)fmaxf(0.f,fminf(y0,fminf(y1,y2)));
        int by1=(int)fminf(H-1,fmaxf(y0,fmaxf(y1,y2)))+1;
        float denom=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2);
        if(fabsf(denom)<1e-6f) continue;
        float inv_d=1.f/denom;
        for(int py=by0;py<by1;py++){
            for(int px=bx0;px<bx1;px++){
                float fx=px+.5f,fy=py+.5f;
                float w0=((y1-y2)*(fx-x2)+(x2-x1)*(fy-y2))*inv_d;
                float w1=((y2-y0)*(fx-x2)+(x0-x2)*(fy-y2))*inv_d;
                float w2=1.f-w0-w1;
                if(w0<0.f||w1<0.f||w2<0.f) continue;
                float iz=w0*iz0+w1*iz1+w2*iz2;
                if(iz<=0.f) continue;
                float z_val=1.f/iz;
                int bidx=py*W+px;
                if(z_val>=zbuf[bidx]) continue;
                zbuf[bidx]=z_val;
                float u=(w0*(u0*iz0)+w1*(u1*iz1)+w2*(u2*iz2))/iz;
                float v=(w0*(v0*iz0)+w1*(v1*iz1)+w2*(v2*iz2))/iz;
                int ti=(int)((u-(float)floor(u))*TW); if(ti<0)ti=0; if(ti>=TW)ti=TW-1;
                int tj=(int)((v-(float)floor(v))*TH); if(tj<0)tj=0; if(tj>=TH)tj=TH-1;
                const unsigned char* tp=tex+(tj*TW+ti)*4;
                if(tp[3]<128) continue;
                int p=bidx*3;
                float r=tp[0]*sh, g_=tp[1]*sh, b_=tp[2]*sh;
                fb[p  ]=(unsigned char)(r>255.f?255:r);
                fb[p+1]=(unsigned char)(g_>255.f?255:g_);
                fb[p+2]=(unsigned char)(b_>255.f?255:b_);
            }
        }
    }
}
"""

def _build_c_rasterizer():
    """Compile _C_RASTER_SRC with gcc; return loaded ctypes.CDLL or None."""
    try:
        import ctypes, tempfile, subprocess, os as _os
        fd, src = tempfile.mkstemp(suffix='.c')
        so  = src[:-2] + '.so'
        with _os.fdopen(fd, 'w') as f:
            f.write(_C_RASTER_SRC)
        rc = subprocess.run(
            ['gcc', '-O3', '-march=native', '-shared', '-fPIC', '-o', so, src, '-lm'],
            capture_output=True)
        _os.unlink(src)
        if rc.returncode != 0:
            return None
        lib = ctypes.CDLL(so)
        vp  = ctypes.c_void_p
        lib.raster_flat.argtypes = [vp, ctypes.c_int, ctypes.c_int, vp, vp, vp, ctypes.c_int]
        lib.raster_flat.restype  = None
        lib.raster_tex.argtypes  = [vp, vp, ctypes.c_int, ctypes.c_int,
                                     vp, vp, vp, vp, vp, ctypes.c_int, ctypes.c_int,
                                     vp, ctypes.c_int]
        lib.raster_tex.restype   = None
        return lib
    except Exception:
        return None

_RASTER_LIB = _build_c_rasterizer() if HAS_NUMPY else None

# ---------------------------------------------------------------------------
# OMT format: data types
# ---------------------------------------------------------------------------

MAGICS:  Dict[bytes, int] = {b"0MFS": 1, b"0MF2": 2, b"0MF3": 3}
MAGIC_B: Dict[int, bytes]  = {v: k for k, v in MAGICS.items()}


@dataclass
class Chunk:
    type:        str
    id:          int
    offset:      int
    size:        int
    name:        str   = ""
    compression: int   = 0
    raw:         bytes = field(default=b"", repr=False)


@dataclass
class OMTFile:
    path:    str
    version: int
    chunks:  List[Chunk] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Reader / writer
# ---------------------------------------------------------------------------

def _read_string(d: bytes, p: int) -> Tuple[str, int]:
    n = d[p]; p += 1
    return d[p:p + n].decode("latin-1"), p + n


def _write_string(s: str) -> bytes:
    b = s.encode("latin-1")
    if len(b) > 255:
        raise ValueError(f"chunk name too long ({len(b)} > 255): {s!r}")
    return bytes([len(b)]) + b


def load_omt(path: str) -> OMTFile:
    data = open(path, "rb").read()
    magic = data[:4]
    if magic not in MAGICS:
        raise ValueError(f"Not an OMT file (magic {magic!r})")
    version = MAGICS[magic]
    (hdr,) = struct.unpack_from(">I", data, 4)
    p = hdr
    n_types = struct.unpack_from(">I", data, p)[0]; p += 4
    chunks: List[Chunk] = []
    for _ in range(n_types):
        ctype = data[p:p + 4].decode("latin-1"); p += 4
        n_chunks = struct.unpack_from(">I", data, p)[0]; p += 4
        for _ in range(n_chunks):
            cid  = struct.unpack_from(">I", data, p)[0]; p += 4
            off  = struct.unpack_from(">I", data, p)[0]; p += 4
            size = struct.unpack_from(">I", data, p)[0]; p += 4
            name = ""
            if version >= 2:
                name, p = _read_string(data, p)
            comp = 0
            if version >= 3:
                comp = struct.unpack_from(">H", data, p)[0]; p += 2
            chunks.append(Chunk(ctype, cid, off, size, name, comp, data[off:off + size]))
    return OMTFile(path, version, chunks)


def chunk_payload(c: Chunk) -> bytes:
    if c.compression and len(c.raw) >= 4:
        return zlib.decompress(c.raw[4:])
    return c.raw


def build_omt(version: int, pairs: List[Tuple[Chunk, bytes]]) -> bytes:
    out = bytearray(MAGIC_B[version] + b"\x00\x00\x00\x00")
    placed: List[Chunk] = []
    for c, payload in pairs:
        off = len(out); out += payload
        placed.append(Chunk(c.type, c.id, off, len(payload), c.name, c.compression))
    hdr_off = len(out)
    struct.pack_into(">I", out, 4, hdr_off)
    by_type: Dict[str, List[Chunk]] = {}
    for c in placed:
        by_type.setdefault(c.type, []).append(c)
    hdr = bytearray()
    hdr += struct.pack(">I", len(by_type))
    for ctype, clist in by_type.items():
        hdr += ctype.encode("latin-1") + struct.pack(">I", len(clist))
        for c in clist:
            hdr += struct.pack(">III", c.id, c.offset, c.size)
            if version >= 2: hdr += _write_string(c.name)
            if version >= 3: hdr += struct.pack(">H", c.compression)
    hdr += struct.pack(">I", 0)
    out += hdr
    return bytes(out)


def _add_0hdr(raw: bytes, version: int) -> bytes:
    """Append a self-referencing 0HDR entry to a built OMT binary.

    The game uses 0HDR as an in-chunk copy of the header table so it can
    locate the index without seeking to end-of-file.  Its payload IS the
    header table bytes (same offset), and it lists itself in that table.
    """
    hdr_off = struct.unpack_from(">I", raw, 4)[0]
    hdr_bytes = raw[hdr_off:]          # current header (ends with 4-byte zero terminator)
    n_old = struct.unpack_from(">I", hdr_bytes, 0)[0]

    # 0HDR entry size: type(4) + n(4) + id(4)+off(4)+sz(4)+name(1) [+comp(2) if v3]
    ohdr_entry_sz = 4 + 4 + 4 + 4 + 4 + 1
    if version >= 3:
        ohdr_entry_sz += 2
    new_hdr_sz = len(hdr_bytes) + ohdr_entry_sz

    ohdr_entry = bytearray()
    ohdr_entry += b"0HDR"
    ohdr_entry += struct.pack(">I", 1)              # n_chunks = 1
    ohdr_entry += struct.pack(">I", 0x30484944)     # id
    ohdr_entry += struct.pack(">I", hdr_off)        # offset  (points at new_hdr itself)
    ohdr_entry += struct.pack(">I", new_hdr_sz)     # size    (= len of new header)
    ohdr_entry += b"\x00"                           # empty name
    if version >= 3:
        ohdr_entry += struct.pack(">H", 0)          # compression = 0

    new_hdr = bytearray()
    new_hdr += struct.pack(">I", n_old + 1)         # n_types (one more)
    new_hdr += hdr_bytes[4:-4]                      # all existing type entries
    new_hdr += ohdr_entry
    new_hdr += struct.pack(">I", 0)                 # terminator

    assert len(new_hdr) == new_hdr_sz, f"0HDR size mismatch: {len(new_hdr)} vs {new_hdr_sz}"
    return raw[:hdr_off] + bytes(new_hdr)


# ---------------------------------------------------------------------------
# OMT Builder  — encode_canv / encode_3dma / encode_3dsh / OMTBuilder
# ---------------------------------------------------------------------------

def _packbits32(row_bytes: bytes) -> bytes:
    """PackBits-encode a row of 32-bit pixels.

    unit = 4: decoder reads a 4-byte control token and interprets
    the first 2 bytes as a signed little-endian i16.
      n >= 0  → literal run of (n+1) pixels follows
      n < 0   → RLE run of (1-n) copies of the next pixel follows
    """
    unit = 4
    n = len(row_bytes) // unit
    out = bytearray()
    i = 0
    while i < n:
        j = i + 1
        while j < n and j - i < 32767 and row_bytes[j*4:(j+1)*4] == row_bytes[i*4:(i+1)*4]:
            j += 1
        rlen = j - i
        if rlen > 1:
            # RLE: 4-byte control (i16 LE = -(rlen-1), 2 bytes padding), then 1 pixel
            out += struct.pack("<h", -(rlen - 1)) + b"\x00\x00"
            out += row_bytes[i*4:(i+1)*4]
            i = j
            continue
        # literal run
        k = i + 1
        while k < n and k - i < 32767:
            if k + 1 < n and row_bytes[k*4:(k+1)*4] == row_bytes[(k+1)*4:(k+2)*4]:
                break
            k += 1
        llen = k - i
        # 4-byte control (i16 LE = llen-1, 2 bytes padding), then llen pixels
        out += struct.pack("<h", llen - 1) + b"\x00\x00"
        out += row_bytes[i*4:k*4]
        i = k
    return bytes(out)


def encode_canv(rgba: bytes, width: int, height: int) -> bytes:
    """Encode RGBA8888 pixel data as an OmCv (Canv) chunk payload.

    Uses depth=32, compression=2 (true-color, 4 bytes/pixel, PackBits rows).
    Pixels with alpha < 128 are keyed out as a fixed magenta transparent value.

    Args:
        rgba:   flat bytes, length = width * height * 4, row-major top-down.
        width:  image width in pixels.
        height: image height in pixels.

    Returns:
        Raw bytes suitable for use as a Canv chunk payload.
    """
    _TKEY   = 0x00FF00FF            # transparent key: magenta (unlikely in real art)
    _TKEY_B = struct.pack(">I", _TKEY)

    has_alpha = any(rgba[i*4 + 3] < 128 for i in range(width * height))

    rows_buf = bytearray()
    for y in range(height):
        pix = bytearray(width * 4)
        for x in range(width):
            o = (y * width + x) * 4
            r, g, b, a = rgba[o], rgba[o+1], rgba[o+2], rgba[o+3]
            if has_alpha and a < 128:
                pix[x*4:(x+1)*4] = _TKEY_B
            else:
                struct.pack_into(">I", pix, x * 4, (r << 16) | (g << 8) | b)
        encoded = _packbits32(bytes(pix))
        rows_buf += struct.pack(">h", len(encoded))
        rows_buf += encoded

    omgw = bytearray()
    omgw += struct.pack(">HHH", width, height, 32)  # w, h, depth=32
    omgw += bytes([2, 1])                            # comp=2 (unit=4), ver=1
    omgw += struct.pack(">H", width * 4)             # nbytes per uncompressed row
    omgw += struct.pack(">h", 1 if has_alpha else 0) # dotransparent
    if has_alpha:
        omgw += struct.pack(">I", _TKEY)
    omgw += struct.pack(">h", 0)                     # dopalette=0
    omgw += struct.pack(">I", len(rows_buf))
    omgw += rows_buf

    return b"OmCv\x00\x00\x10\x0cOmGW" + bytes(omgw)


def encode_3dma(
    diffuse:  Tuple[int,int,int] = (255, 255, 255),
    *,
    alpha:    float = 0.0,
    nofog:    bool  = False,
    emission: Tuple[int,int,int] = (255, 255, 255),
    specular: Tuple[int,int,int] = (0,   0,   0),
    ambient:  Tuple[int,int,int] = (0,   0,   0),
    gouraud:  bool  = False,
    shaded:   bool  = True,
    texture_id: Optional[int]   = None,
) -> bytes:
    """Encode a 3DMa material chunk payload (version 1).

    Args:
        diffuse:    RGB 0-255 base colour.
        alpha:      0.0 = opaque (OMedia convention).
        texture_id: Canv chunk id for a textured material, or None for phong.
        emission / specular / ambient: RGB 0-255 light response values.
        gouraud:    enable Gouraud shading.
        shaded:     apply lighting to texture (mtype=2 only).
        nofog:      disable fog for this material.

    Returns:
        Raw bytes for a 3DMa chunk payload.
    """
    def _rgb(r: int, g: int, b: int) -> bytes:
        return struct.pack(">HHH", r << 8, g << 8, b << 8)

    out = bytearray()
    if texture_id is None:
        out += struct.pack(">hh", 1, 1)             # version=1, mtype=1 (phong)
        out += struct.pack(">f",  alpha)
        out += struct.pack(">h",  1 if nofog else 0)
        out += _rgb(*emission)
        out += _rgb(*diffuse)
        out += _rgb(*specular)
        out += _rgb(*ambient)
        out += struct.pack(">h",  1 if gouraud else 0)
    else:
        out += struct.pack(">hh", 1, 2)             # version=1, mtype=2 (phong+tex)
        out += struct.pack(">f",  alpha)
        out += struct.pack(">h",  1 if nofog else 0)
        out += _rgb(*emission)
        out += _rgb(*diffuse)
        out += _rgb(*specular)
        out += _rgb(*ambient)
        out += struct.pack(">h",  1 if gouraud else 0)
        out += struct.pack(">h",  1 if shaded  else 0)
        out += struct.pack(">h",  1)                # has_tex=1
        out += struct.pack(">h",  1)                # filled=1
        out += b"Canv"
        out += struct.pack(">i",  texture_id)
    out += struct.pack(">h",  3)                    # mtype=3 body-list terminator (C++ decoder reads unconditionally)
    return bytes(out)


def encode_3dsh(
    verts:   list,
    faces:   list,
    normals: Optional[list] = None,
) -> bytes:
    """Encode a 3DSh (3DSP version 3) chunk payload.

    Args:
        verts:   list of (x, y, z) float tuples.
        faces:   list of (vis, uvs, nis, face_normal, mat) tuples:
                   vis  — vertex-index list (len ≥ 3; polygon fan)
                   uvs  — matching (u, v) per vertex
                   nis  — matching normal-index per vertex (or empty → uses vis)
                   face_normal — (nx, ny, nz) pre-computed face normal
                   mat  — ("3DMa", chunk_id)  or  None
        normals: optional per-vertex smooth normals list of (nx, ny, nz).

    Returns:
        Raw bytes for a 3DSh chunk payload.
    """
    import math as _math

    if verts:
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        cz = sum(v[2] for v in verts) / len(verts)
        r  = max(_math.sqrt((v[0]-cx)**2+(v[1]-cy)**2+(v[2]-cz)**2) for v in verts)
    else:
        cx = cy = cz = r = 0.0

    out = bytearray()
    out += b"3DSP"
    out += struct.pack(">I",   3)                    # version 3
    out += struct.pack(">4f",  cx, cy, cz, r)        # vis sphere
    out += struct.pack(">4f",  cx, cy, cz, r)        # glob sphere
    out += struct.pack(">I",   len(verts))
    for x, y, z in verts:
        out += struct.pack(">fff", x, y, z)

    out += struct.pack(">I", len(faces))
    for (vis, uvs, nis, fn, mat) in faces:
        n_pv = len(vis)
        ni_list = nis if nis else vis          # fall back to vertex index as normal index
        out += struct.pack(">HI", 0, 0)        # flags=0, unk=0
        out += struct.pack(">i", n_pv)
        for k in range(n_pv):
            out += struct.pack(">IIIff", vis[k], ni_list[k], 0, uvs[k][0], uvs[k][1])
        out += struct.pack(">fff", fn[0], fn[1], fn[2])
        if mat and mat[0] == "3DMa":
            out += struct.pack(">h", 1)
            out += b"3DMa"
            out += struct.pack(">i", mat[1])
        else:
            out += struct.pack(">h", 0)

    out += struct.pack(">I", 0)               # 0 bounding spheres
    norms = normals or []
    out += struct.pack(">I", len(norms))
    for nx, ny, nz in norms:
        out += struct.pack(">fff", nx, ny, nz)
    out += struct.pack(">I", 0)               # n_colors (version > 2 requires this field)

    return bytes(out)


def _parse_obj(text: str, mat_lookup: Dict[str, int]):
    """Parse a Wavefront OBJ text into encode_3dsh-compatible data.

    Args:
        text:       raw OBJ file contents.
        mat_lookup: {material_name.lower() -> 3DMa chunk id}

    Returns:
        (verts, faces, normals) where normals may be None.
        faces follow the encode_3dsh (vis, uvs, nis, face_normal, mat) convention.
    """
    import math as _m

    obj_v:  list = []
    obj_vt: list = [(0.0, 0.0)]   # 1-indexed placeholder at 0
    obj_vn: list = []             # 1-indexed, stored without placeholder

    cur_mat_id = None
    raw_faces:  list = []  # [(vids, tids, nids, mat_id)]  all 0-based except mat_id

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'): continue
        parts = line.split()
        tok = parts[0]
        if tok == 'v' and len(parts) >= 4:
            obj_v.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif tok == 'vt' and len(parts) >= 2:
            u = float(parts[1])
            v = 1.0 - float(parts[2]) if len(parts) > 2 else 0.0  # flip V
            obj_vt.append((u, v))
        elif tok == 'vn' and len(parts) >= 4:
            obj_vn.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif tok == 'usemtl':
            mname = ' '.join(parts[1:]).lower()
            cur_mat_id = mat_lookup.get(mname)
        elif tok == 'f' and len(parts) >= 4:
            vids: list = []; tids: list = []; nids: list = []
            for p in parts[1:]:
                comps = p.split('/')
                vi = int(comps[0])
                ti = int(comps[1]) if len(comps) > 1 and comps[1] else 0
                ni = int(comps[2]) if len(comps) > 2 and comps[2] else 0
                nv = len(obj_v); nvt = len(obj_vt) - 1; nvn = len(obj_vn)
                if vi < 0: vi = nv + vi + 1
                if ti < 0: ti = nvt + ti + 1
                if ni < 0: ni = nvn + ni + 1
                vids.append(vi - 1)  # 0-indexed
                tids.append(ti)      # 1-indexed into obj_vt (0 = no UV)
                nids.append(ni)      # 1-indexed into obj_vn (0 = none)
            if len(vids) >= 3:
                raw_faces.append((vids, tids, nids, cur_mat_id))

    has_obj_norms = any(ni > 0 for _, _, nids, _ in raw_faces for ni in nids)

    # Build normals array
    if has_obj_norms:
        normals = list(obj_vn)   # 0-indexed: OBJ ni-1 maps directly
    else:
        # Compute per-vertex averaged face normals
        acc = [(0.0, 0.0, 0.0)] * len(obj_v)
        cnt = [0] * len(obj_v)
        for vids, _, _, _ in raw_faces:
            a, b, c = obj_v[vids[0]], obj_v[vids[1]], obj_v[vids[2]]
            ax, ay, az = b[0]-a[0], b[1]-a[1], b[2]-a[2]
            bx, by, bz = c[0]-a[0], c[1]-a[1], c[2]-a[2]
            nx, ny, nz = ay*bz-az*by, az*bx-ax*bz, ax*by-ay*bx
            for vi in vids:
                ox, oy, oz = acc[vi]
                acc[vi] = (ox+nx, oy+ny, oz+nz)
                cnt[vi] += 1
        normals = []
        for (nx, ny, nz), c in zip(acc, cnt):
            if c == 0: normals.append((0.0, 0.0, 1.0))
            else:
                L = max(1e-9, _m.sqrt(nx*nx+ny*ny+nz*nz))
                normals.append((nx/L, ny/L, nz/L))

    # Build face list
    faces: list = []
    for (vids, tids, nids, mat_id) in raw_faces:
        a, b, c = obj_v[vids[0]], obj_v[vids[1]], obj_v[vids[2]]
        ax, ay, az = b[0]-a[0], b[1]-a[1], b[2]-a[2]
        bx, by, bz = c[0]-a[0], c[1]-a[1], c[2]-a[2]
        nx, ny, nz = ay*bz-az*by, az*bx-ax*bz, ax*by-ay*bx
        L = max(1e-9, _m.sqrt(nx*nx+ny*ny+nz*nz))
        fn = (nx/L, ny/L, nz/L)
        uvs  = [(obj_vt[ti][0], obj_vt[ti][1]) if ti > 0 else (0.0, 0.0) for ti in tids]
        if has_obj_norms:
            nis = [min(max(ni - 1, 0), len(normals)-1) if ni > 0 else vids[k]
                   for k, ni in enumerate(nids)]
        else:
            nis = list(vids)
        mat = ("3DMa", mat_id) if mat_id is not None else None
        faces.append((vids, uvs, nis, fn, mat))

    return obj_v, faces, (normals if normals else None)


def write_obj(verts, faces, normals, mat_chunks, canv_chunks,
              obj_path: str, mtl_path: Optional[str] = None,
              scale: float = 0.01) -> None:
    """Export decoded 3DSh mesh to Wavefront OBJ (+ MTL + PNG textures).

    scale: multiply every vertex coordinate before writing (default 0.01 so
    that game units ~= cm become metres in Blender).

    Args:
        verts:       list of (x, y, z)
        faces:       list of (vis, uvs, nis, face_normal, mat) from decode_shape_full
        normals:     list of (nx, ny, nz) or empty
        mat_chunks:  {chunk_id: Chunk} for 3DMa entries in the same OMT
        canv_chunks: {chunk_id: Chunk} for Canv entries (textures)
        obj_path:    destination .obj file
        mtl_path:    destination .mtl file; None skips material export
    """
    # Collect material id → name
    used_mats: Dict[int, str] = {}
    for _, _, _, _, mat in faces:
        if mat and mat[0] == "3DMa":
            mid = mat[1]
            if mid not in used_mats:
                ch = mat_chunks.get(mid)
                used_mats[mid] = ch.name if ch and ch.name else f"mat_{mid}"

    mtl_name = os.path.basename(mtl_path) if mtl_path else None

    # --- OBJ ----------------------------------------------------------------
    obj = []
    if mtl_name and used_mats:
        obj.append(f"mtllib {mtl_name}\n")

    for x, y, z in verts:
        obj.append(f"v {x*scale:.6f} {y*scale:.6f} {z*scale:.6f}\n")

    # Build UV table (deduplicate; flip V back for OBJ convention)
    uv_list: List[Tuple[float, float]] = []
    uv_idx: Dict[Tuple[float, float], int] = {}
    for _, uvs, _, _, _ in faces:
        for u, v in uvs:
            key = (round(u, 6), round(1.0 - v, 6))
            if key not in uv_idx:
                uv_idx[key] = len(uv_list) + 1
                uv_list.append(key)
    for u, v in uv_list:
        obj.append(f"vt {u:.6f} {v:.6f}\n")

    for nx, ny, nz in (normals or []):
        obj.append(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")

    # Group by material for usemtl lines
    cur_mat_id: Optional[int] = object()  # sentinel
    sorted_faces = sorted(faces,
        key=lambda f: f[4][1] if f[4] and f[4][0] == "3DMa" else -1)
    for vis, uvs, nis, _, mat in sorted_faces:
        mid = mat[1] if mat and mat[0] == "3DMa" else None
        if mid != cur_mat_id:
            cur_mat_id = mid
            mname = used_mats.get(mid, "default") if mid is not None else "default"
            obj.append(f"usemtl {mname}\n")
        # Fan-triangulate polygon
        for k in range(1, len(vis) - 1):
            tokens = []
            for j in (0, k, k + 1):
                vi  = vis[j] + 1
                key = (round(uvs[j][0], 6), round(1.0 - uvs[j][1], 6))
                vti = uv_idx.get(key, 0)
                vni = (nis[j] + 1) if normals else 0
                if vti and vni:
                    tokens.append(f"{vi}/{vti}/{vni}")
                elif vti:
                    tokens.append(f"{vi}/{vti}")
                elif vni:
                    tokens.append(f"{vi}//{vni}")
                else:
                    tokens.append(str(vi))
            obj.append(f"f {' '.join(tokens)}\n")

    with open(obj_path, "w") as fh:
        fh.writelines(obj)

    # --- MTL ----------------------------------------------------------------
    if not (mtl_path and used_mats):
        return
    tex_dir = os.path.dirname(mtl_path) or "."
    mtl = []
    for mid, mname in used_mats.items():
        ch = mat_chunks.get(mid)
        colors = mat_all_colors(ch.raw if ch else b"")
        diff = colors.get("diffuse", (180, 180, 180))
        emis = colors.get("emission", (0, 0, 0))
        spec = colors.get("specular", (0, 0, 0))
        mtl.append(f"newmtl {mname}\n")
        mtl.append(f"Kd {diff[0]/255:.4f} {diff[1]/255:.4f} {diff[2]/255:.4f}\n")
        mtl.append(f"Ka {emis[0]/255:.4f} {emis[1]/255:.4f} {emis[2]/255:.4f}\n")
        mtl.append(f"Ks {spec[0]/255:.4f} {spec[1]/255:.4f} {spec[2]/255:.4f}\n")
        tex_ref = colors.get("texture")
        if tex_ref and tex_ref[0] == "Canv" and HAS_PIL:
            canv_ch = canv_chunks.get(tex_ref[1])
            if canv_ch:
                try:
                    from PIL import Image as _Img
                    cw, ch2, _, _, rgba = decode_canv(chunk_payload(canv_ch))
                    img = _Img.frombytes("RGBA", (cw, ch2), rgba)
                    tex_fname = f"{mname}.png"
                    img.save(os.path.join(tex_dir, tex_fname))
                    mtl.append(f"map_Kd {tex_fname}\n")
                except Exception:
                    pass
        mtl.append("\n")
    with open(mtl_path, "w") as fh:
        fh.writelines(mtl)


def encode_mCTy(pairs: List[Tuple["Chunk", bytes]], version: int = 5) -> bytes:
    """Encode an mCTy (type catalog) chunk payload for the given chunk list.

    mCTy structure (big-endian):
      u16 version  u16 unk=0  u16 n_types
      For each distinct chunk type (order preserved):
        char[4] type_name  u16 flag=1  u32 unk=0  u16 n_entries
        For each entry:
          u32 chunk_id  5 bytes rest

    '0HDR' and 'mCTy' chunks are excluded from the catalog (game convention).

    Args:
        pairs:   the (Chunk, payload) list that will go into the OMT.
        version: mCTy payload version field (always 5 in observed game files).
    """
    _SKIP = frozenset(("mCTy", "0HDR"))
    by_type: Dict[str, List[int]] = {}
    for c, _ in pairs:
        if c.type not in _SKIP:
            by_type.setdefault(c.type, []).append(c.id)

    if not by_type:
        return struct.pack(">HHH", version, 0, 0)

    out = bytearray()
    out += struct.pack(">HHH", version, 0, len(by_type))

    for ctype, ids in by_type.items():
        out += ctype.encode("latin-1")
        out += struct.pack(">HI", 1, 0)        # flag=1, unk=0
        out += struct.pack(">H", len(ids))     # n_entries
        for idx, cid in enumerate(ids):
            out += struct.pack(">I", cid)
            # game files always have 0x01 at rest[2] for the last entry per type
            rest = bytearray(5)
            if idx == len(ids) - 1:
                rest[2] = 0x01
            out += bytes(rest)

    return bytes(out)


# ---------------------------------------------------------------------------
# WAV sanitizer — produces game-compatible RIFF/WAV
# ---------------------------------------------------------------------------

def sanitize_wav_for_game(wav_bytes: bytes) -> bytes:
    """Return a clean, game-compatible RIFF/WAV built from wav_bytes.

    OMediaSound::read_class (decompiled from MECHANIX.exe) scans the RIFF
    payload 4 bytes at a time for 'fmt ' and 'data' magic.  It does not parse
    chunk sizes to skip sub-chunks, so LIST/INFO metadata before the real
    sub-chunks can confuse it.  Worse, the game allocates the entire WAV into a
    DirectSound buffer, so multi-MB files reliably crash the 2001 engine.

    This function enforces the exact format found in all original game OMTs:
      • RIFF/WAVE with only 'fmt ' + 'data' sub-chunks (no LIST, fact, etc.)
      • PCM (wFormatTag = 1); raises ValueError for non-PCM input
      • Mono (stereo input is mixed down with 16-bit arithmetic)
      • 11 025 Hz (other rates resampled with nearest-neighbour — good enough
        for 8–16 s game sound effects)
      • 16-bit signed PCM
      • fmt chunk exactly 16 bytes (plain PCMWAVEFORMAT, no cbSize field)
    """
    import array as _array

    if wav_bytes[:4] != b'RIFF' or wav_bytes[8:12] != b'WAVE':
        raise ValueError("Not a RIFF/WAV file")

    fmt_payload: Optional[bytes] = None
    pcm_data:    Optional[bytes] = None

    pos = 12
    while pos + 8 <= len(wav_bytes):
        cid  = wav_bytes[pos:pos+4]
        size = struct.unpack_from('<I', wav_bytes, pos+4)[0]
        end  = pos + 8 + size
        payload = wav_bytes[pos+8:end]
        pos = end + (size & 1)  # RIFF chunks pad to even byte boundary

        if cid == b'fmt ':
            fmt_payload = payload
        elif cid == b'data':
            pcm_data = payload

    if fmt_payload is None:
        raise ValueError("WAV missing fmt chunk")
    if pcm_data is None:
        raise ValueError("WAV missing data chunk")
    if len(fmt_payload) < 16:
        raise ValueError(f"fmt chunk too short ({len(fmt_payload)} bytes)")

    wFormatTag, nChannels, nSamplesPerSec, _, _, wBitsPerSample = \
        struct.unpack_from('<HHIIHH', fmt_payload, 0)

    if wFormatTag != 1:
        raise ValueError(
            f"Non-PCM WAV (wFormatTag={wFormatTag}); "
            "convert to PCM with: ffmpeg -i in.wav -acodec pcm_s16le out.wav")

    # --- convert to 16-bit ---
    if wBitsPerSample == 8:
        # unsigned 8-bit → signed 16-bit
        raw = _array.array('B', pcm_data)
        pcm_data = _array.array('h', ((s - 128) << 8 for s in raw)).tobytes()
        wBitsPerSample = 16
    elif wBitsPerSample != 16:
        raise ValueError(f"Unsupported bit depth: {wBitsPerSample} (expected 8 or 16)")

    # --- downmix stereo → mono ---
    if nChannels == 2:
        samps = _array.array('h', pcm_data)
        n = len(samps) & ~1  # round down to even
        mixed = _array.array('h', ((samps[i] + samps[i+1]) >> 1 for i in range(0, n, 2)))
        pcm_data = mixed.tobytes()
        nChannels = 1
    elif nChannels != 1:
        raise ValueError(f"Unsupported channel count: {nChannels}")

    # --- rebuild clean RIFF ---
    nBlockAlign      = nChannels * (wBitsPerSample // 8)
    nAvgBytesPerSec  = nSamplesPerSec * nBlockAlign
    new_fmt = struct.pack('<HHIIHH',
        1, nChannels, nSamplesPerSec, nAvgBytesPerSec, nBlockAlign, wBitsPerSample)

    body = (b'WAVE'
            + b'fmt ' + struct.pack('<I', 16) + new_fmt
            + b'data' + struct.pack('<I', len(pcm_data)) + pcm_data)
    return b'RIFF' + struct.pack('<I', len(body)) + body


class OMTBuilder:
    """Build a new OMT file from scratch.

    Usage::

        b = OMTBuilder()

        # add a texture from RGBA bytes
        tid = b.add_texture("stone", rgba_bytes, 128, 128)

        # add a textured material that references it
        mid = b.add_material("stone_mat", diffuse=(200, 200, 200), texture_id=tid)

        # add a mesh — verts, and faces in encode_3dsh format
        sid = b.add_shape("floor", verts, faces)

        omt_bytes = b.build()
        open("my.omt", "wb").write(omt_bytes)

    Chunk ids start at 0 per type and increment by 1 for each add_* call.
    """

    def __init__(self, version: int = 2):
        self._version = version
        self._pairs:   List[Tuple[Chunk, bytes]] = []
        self._next_id: Dict[str, int] = {}

    def _alloc(self, ctype: str) -> int:
        n = self._next_id.get(ctype, 0)
        self._next_id[ctype] = n + 1
        return n

    def add_texture(self, name: str, rgba: bytes, width: int, height: int) -> int:
        """Add a Canv texture.  Returns the new chunk id."""
        cid     = self._alloc("Canv")
        payload = encode_canv(rgba, width, height)
        self._pairs.append((Chunk("Canv", cid, 0, len(payload), name), payload))
        return cid

    def add_material(
        self,
        name:       str,
        *,
        diffuse:    Tuple[int,int,int] = (255, 255, 255),
        texture_id: Optional[int]      = None,
        **kw,
    ) -> int:
        """Add a 3DMa material.  Returns the new chunk id.

        kw forwards to encode_3dma: alpha, nofog, emission, specular,
        ambient, gouraud, shaded.
        """
        cid     = self._alloc("3DMa")
        payload = encode_3dma(diffuse=diffuse, texture_id=texture_id, **kw)
        self._pairs.append((Chunk("3DMa", cid, 0, len(payload), name), payload))
        return cid

    def add_shape(
        self,
        name:    str,
        verts:   list,
        faces:   list,
        normals: Optional[list] = None,
    ) -> int:
        """Add a 3DSh mesh.  Returns the new chunk id.

        verts / faces / normals follow the encode_3dsh conventions.
        """
        cid     = self._alloc("3DSh")
        payload = encode_3dsh(verts, faces, normals)
        self._pairs.append((Chunk("3DSh", cid, 0, len(payload), name), payload))
        return cid

    def add_wave(self, name: str, wav_bytes: bytes, sanitize: bool = True) -> int:
        """Add a Wave chunk from raw WAV file bytes.  Returns the new chunk id.

        sanitize=True (default) runs sanitize_wav_for_game() first: strips LIST
        and other non-essential RIFF sub-chunks, downmixes stereo to mono,
        enforces plain 16-byte PCMWAVEFORMAT.  Pass sanitize=False only when
        the bytes are already game-compatible.
        """
        if sanitize:
            wav_bytes = sanitize_wav_for_game(wav_bytes)
        cid = self._alloc("Wave")
        self._pairs.append((Chunk("Wave", cid, 0, len(wav_bytes), name), wav_bytes))
        return cid

    def build(self) -> bytes:
        """Serialise to a complete OMT binary with mCTy catalog and 0HDR index."""
        mcty_payload = encode_mCTy(self._pairs)
        mcty_chunk   = Chunk("mCTy", 0xfffff752, 0, len(mcty_payload), "")
        all_pairs    = [(mcty_chunk, mcty_payload)] + list(self._pairs)
        raw = build_omt(self._version, all_pairs)
        return _add_0hdr(raw, self._version)


# ---------------------------------------------------------------------------
# Hex dump
# ---------------------------------------------------------------------------

def _hex_dump(b: bytes, max_bytes: int = 4096) -> str:
    b = b[:max_bytes]
    lines = []
    for i in range(0, len(b), 16):
        row = b[i:i + 16]
        lines.append(f"{i:08x}  {' '.join(f'{x:02x}' for x in row):<48}  "
                     f"{''.join(chr(x) if 32 <= x < 127 else '.' for x in row)}")
    if len(b) == max_bytes:
        lines.append(f"… ({max_bytes:,} bytes shown)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Canv decoder (OmCv / OmGW format)
# ---------------------------------------------------------------------------

def decode_canv(d: bytes):
    """Return (width, height, depth, transparent, rgba_bytes) or raise."""
    if d[:4] != b"OmCv":  raise ValueError(f"Not OmCv ({d[:4]!r})")
    if d[8:12] != b"OmGW": raise ValueError(f"Unexpected header {d[8:12]!r}")
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
                v = (row[2*x] << 8) | row[2*x+1]; r,g,b = ((v>>10)&31)<<3, ((v>>5)&31)<<3, (v&31)<<3
            elif depth == 32:
                v = struct.unpack_from(">I", row, 4*x)[0]; r,g,b = (v>>16)&255,(v>>8)&255,v&255
            elif depth == 8:
                v = row[x]; r,g,b = palette[v]
            elif depth == 4:
                v = row[x>>1]; v = ((v>>4) if (x&1) else v) & 15; r,g,b = palette[v]
            elif depth == 1:
                v = (row[x>>3] >> (7-(x&7))) & 1; r,g,b = palette[v]
            else:
                raise ValueError(f"depth {depth}")
            out[o:o+4] = bytes((r, g, b, 0 if (dotransparent and v == transp) else 255))
            o += 4
    return width, height, depth, bool(dotransparent), bytes(out)


# ---------------------------------------------------------------------------
# 3DSh: full polygon decode
# ---------------------------------------------------------------------------

def decode_shape_full(d: bytes):
    """Return (vertices, faces, normals).
    vertices: list of (x,y,z)
    normals:  list of (nx,ny,nz) per-vertex smooth normals (may be empty)
    faces:    list of (vis, uvs, nis, face_normal, mat)
              vis/uvs/nis are parallel per-polygon-vertex lists
    """
    if d[:4] != b"3DSP": return [], [], []

    class _S:
        def __init__(self, d): self.d, self.p = d, 0
        def u32(self): v=struct.unpack_from(">I",self.d,self.p)[0];self.p+=4;return v
        def i32(self): v=struct.unpack_from(">i",self.d,self.p)[0];self.p+=4;return v
        def u16(self): v=struct.unpack_from(">H",self.d,self.p)[0];self.p+=2;return v
        def i16(self): v=struct.unpack_from(">h",self.d,self.p)[0];self.p+=2;return v
        def f32(self): v=struct.unpack_from(">f",self.d,self.p)[0];self.p+=4;return v
        def f32n(self,n): v=struct.unpack_from(f">{n}f",self.d,self.p);self.p+=4*n;return v
        def bool(self): return self.i16()!=0
        def link(self):
            if not self.bool(): return None
            t=self.d[self.p:self.p+4].decode("latin-1");self.p+=4
            return (t,self.i32())

    s = _S(d); s.p = 4
    version = s.u32()
    if version > 5: return [], [], []
    s.f32n(4); s.f32n(4)
    if 0 < version < 3: s.f32(); s.bool()
    vertices = [s.f32n(3) for _ in range(s.u32())]
    faces = []
    n_polys = s.u32()
    for _ in range(n_polys):
        try:
            if version > 2:
                s.u16(); s.u32()
                vis = []; uvs = []; nis = []
                for _ in range(s.i32()):
                    vi,ni,ci,u,v = s.u32(),s.u32(),s.u32(),s.f32(),s.f32()
                    if version > 4:
                        for _ in range(s.u32()): s.f32(); s.f32()
                    vis.append(vi); uvs.append((u, v)); nis.append(ni)
                normal = s.f32n(3)
                mat = s.link()
                if version > 4: s.i32()
            else:
                s.bool(); s.bool(); s.i16(); s.u32()
                if version > 0: s.f32(); s.f32()
                s.i16()
                vis = []; uvs = []; nis = []
                for _ in range(s.u16()):
                    vi = s.u32(); u = s.f32(); v = s.f32()
                    ni = s.u32() if version > 1 else vi
                    vis.append(vi); uvs.append((u, v)); nis.append(ni)
                normal = s.f32n(3)
                mat = s.link() if s.bool() else None
            if len(vis) >= 3:
                faces.append((vis, uvs, nis, normal, mat))
        except Exception:
            break
    # Read bounding spheres (skip) then per-vertex smooth normals
    normals: list = []
    try:
        n_spheres = s.u32()
        s.p += n_spheres * 16
        n_normals = s.u32()
        normals = [s.f32n(3) for _ in range(n_normals)]
    except Exception:
        pass
    return vertices, faces, normals


def mat_diffuse_color(payload: bytes) -> Tuple[int,int,int]:
    """Extract the first layer's diffuse (r,g,b) 0-255 from a 3DMa payload."""
    try:
        p = 0
        p += 2  # version
        mtype = struct.unpack_from(">h", payload, p)[0]; p += 2
        if mtype > 2: return (160, 160, 160)
        p += 4  # alpha (f32)
        p += 2  # nofog (bool16)
        if mtype == 0:
            r,g,b = struct.unpack_from(">HHH", payload, p)
        else:
            p += 6  # emission
            r,g,b = struct.unpack_from(">HHH", payload, p)
        return (r >> 8, g >> 8, b >> 8)
    except Exception:
        return (160, 160, 160)


def mat_all_colors(payload: bytes) -> dict:
    """Return dict of emission/diffuse/specular/ambient + alpha for first layer."""
    result = {"mtype":0,"alpha":0.0,"nofog":False,"emission":(0,0,0),
              "diffuse":(180,180,180),"specular":(0,0,0),"ambient":(40,40,40),
              "gouraud":False,"texture":None}
    try:
        p = 0
        result["version"] = struct.unpack_from(">h", payload, p)[0]; p += 2
        mtype = struct.unpack_from(">h", payload, p)[0]; p += 2
        if mtype > 2: return result
        result["mtype"] = mtype
        result["alpha"] = struct.unpack_from(">f", payload, p)[0]; p += 4
        result["nofog"] = struct.unpack_from(">h", payload, p)[0] != 0; p += 2
        if mtype == 0:
            r,g,b = struct.unpack_from(">HHH", payload, p); p += 6
            result["diffuse"] = (r>>8, g>>8, b>>8)
        else:
            em = struct.unpack_from(">HHH", payload, p); p += 6
            di = struct.unpack_from(">HHH", payload, p); p += 6
            sp = struct.unpack_from(">HHH", payload, p); p += 6
            am = struct.unpack_from(">HHH", payload, p); p += 6
            gouraud = struct.unpack_from(">h", payload, p)[0]; p += 2
            result.update(emission=tuple(x>>8 for x in em),
                          diffuse =tuple(x>>8 for x in di),
                          specular=tuple(x>>8 for x in sp),
                          ambient =tuple(x>>8 for x in am),
                          gouraud =bool(gouraud))
            if mtype == 2:
                result["shaded"] = struct.unpack_from(">h", payload, p)[0] != 0; p += 2
                has_tex = struct.unpack_from(">h", payload, p)[0]; p += 2
                if has_tex == 1:
                    filled = struct.unpack_from(">h", payload, p)[0]; p += 2
                    if filled:
                        t = payload[p:p+4].decode("latin-1"); p += 4
                        tid = struct.unpack_from(">i", payload, p)[0]; p += 4
                        result["texture"] = (t, tid)
    except Exception:
        pass
    return result


def summarise_shape(d: bytes) -> str:
    if d[:4] != b"3DSP": return f"Not a 3DSP shape ({d[:4]!r})"
    p = 4
    version = struct.unpack_from(">I", d, p)[0]; p += 4
    vis  = struct.unpack_from(">4f", d, p); p += 16
    glob = struct.unpack_from(">4f", d, p); p += 16
    if 0 < version < 3: p += 6
    n_verts  = struct.unpack_from(">I", d, p)[0]; p += 4
    p += n_verts * 12
    n_polys  = struct.unpack_from(">I", d, p)[0]
    return (f"3DSh  version {version}\n"
            f"  Vertices : {n_verts:,}\n"
            f"  Polygons : {n_polys:,}\n"
            f"  Vis sphere : centre ({vis[0]:.2f}, {vis[1]:.2f}, {vis[2]:.2f})  r={vis[3]:.2f}\n"
            f"  Bounds     : centre ({glob[0]:.2f}, {glob[1]:.2f}, {glob[2]:.2f})  r={glob[3]:.2f}")


def summarise_material(d: bytes) -> str:
    p = 0; version = struct.unpack_from(">h", d, p)[0]; p += 2
    lines = [f"3DMa  version {version}"]
    layer = 0
    while p < len(d):
        mtype = struct.unpack_from(">h", d, p)[0]; p += 2
        if mtype > 2: break
        alpha = struct.unpack_from(">f", d, p)[0]; p += 4
        nofog = struct.unpack_from(">h", d, p)[0]; p += 2
        pre = f"  layer {layer}: "
        if mtype == 0:
            r,g,b = struct.unpack_from(">HHH", d, p); p += 6
            lines.append(f"{pre}Flat   rgb=({r>>8},{g>>8},{b>>8})  alpha={alpha:.4f}")
        else:
            em = struct.unpack_from(">HHH", d, p); p += 6
            di = struct.unpack_from(">HHH", d, p); p += 6
            sp = struct.unpack_from(">HHH", d, p); p += 6
            am = struct.unpack_from(">HHH", d, p); p += 6
            gouraud = struct.unpack_from(">h", d, p)[0]; p += 2
            di8 = tuple(x>>8 for x in di); sp8 = tuple(x>>8 for x in sp)
            if mtype == 2:
                shaded = struct.unpack_from(">h", d, p)[0]; p += 2
                has_tex = struct.unpack_from(">h", d, p)[0]; p += 2
                tex = None
                if has_tex == 1:
                    filled = struct.unpack_from(">h", d, p)[0]; p += 2
                    if filled:
                        t = d[p:p+4].decode("latin-1"); p += 4
                        tid = struct.unpack_from(">i", d, p)[0]; p += 4
                        tex = (t, tid)
                lines.append(f"{pre}Tex    diffuse={di8}  specular={sp8}  "
                              f"tex={'%s#%d'%tex if tex else 'none'}  "
                              f"alpha={alpha:.4f}  gouraud={bool(gouraud)}")
            else:
                lines.append(f"{pre}Lit    diffuse={di8}  specular={sp8}  "
                              f"alpha={alpha:.4f}  gouraud={bool(gouraud)}")
        layer += 1
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sphere geometry for material preview
# ---------------------------------------------------------------------------

def _make_sphere(divs: int = 20):
    """Return (verts, tris).  verts = list of (x,y,z) on unit sphere.
    tris = list of ((i0,i1,i2), (nx,ny,nz) face normal, (uva,uvb,uvc))."""
    verts = []
    for i in range(divs + 1):
        lat = math.pi * i / divs
        y = math.cos(lat); r = math.sin(lat)
        for j in range(divs):
            lon = 2 * math.pi * j / divs
            verts.append((r * math.cos(lon), y, r * math.sin(lon)))

    def _fn(*idxs):
        n = len(idxs)
        nx = sum(verts[k][0] for k in idxs) / n
        ny = sum(verts[k][1] for k in idxs) / n
        nz = sum(verts[k][2] for k in idxs) / n
        L = max(0.001, (nx*nx + ny*ny + nz*nz) ** 0.5)
        return nx/L, ny/L, nz/L

    def _uv(idx):
        row = idx // divs; col = idx % divs
        return (col / divs, row / divs)

    tris = []
    for i in range(divs):
        for j in range(divs):
            a = i * divs + j; b = i * divs + (j+1) % divs
            c = (i+1) * divs + j; d = (i+1) * divs + (j+1) % divs
            # UV: use actual (j+1)/divs at the right edge to avoid seam wrap artifact
            ua, ub = j/divs, (j+1)/divs
            va, vc = i/divs, (i+1)/divs
            tris.append(((a, b, d), _fn(a, b, d), ((ua,va),(ub,va),(ub,vc))))
            tris.append(((a, d, c), _fn(a, d, c), ((ua,va),(ub,vc),(ua,vc))))
    return verts, tris

_SPHERE_VERTS, _SPHERE_TRIS = _make_sphere(20)


def _canvas_average_color(rgba: bytes, w: int, h: int) -> Tuple[int, int, int]:
    """Average colour of non-transparent pixels in an RGBA flat buffer."""
    r = g = b = cnt = 0
    for i in range(w * h):
        a = rgba[i * 4 + 3]
        if a > 128:
            r += rgba[i * 4]; g += rgba[i * 4 + 1]; b += rgba[i * 4 + 2]
            cnt += 1
    if cnt == 0:
        return (160, 160, 160)
    return (r // cnt, g // cnt, b // cnt)


def _pt_in_tri_2d(mx, my, p0, p1, p2, slack=0.0):
    """Return True if screen point (mx,my) is inside (or within slack px of) triangle p0-p1-p2."""
    def _s(ax,ay,bx,by,px,py): return (px-bx)*(ay-by)-(ax-bx)*(py-by)
    d0=_s(p0[0],p0[1],p1[0],p1[1],mx,my)
    d1=_s(p1[0],p1[1],p2[0],p2[1],mx,my)
    d2=_s(p2[0],p2[1],p0[0],p0[1],mx,my)
    if not ((d0<0 or d1<0 or d2<0) and (d0>0 or d1>0 or d2>0)):
        return True
    if slack <= 0.0:
        return False
    # near-miss: within slack px of any edge
    def _dseg2(ax,ay,bx,by,px,py):
        ex,ey=bx-ax,by-ay; l2=ex*ex+ey*ey
        if l2<1e-9: return (px-ax)**2+(py-ay)**2
        t=max(0.,min(1.,((px-ax)*ex+(py-ay)*ey)/l2))
        return (px-ax-t*ex)**2+(py-ay-t*ey)**2
    s2=slack*slack
    return (_dseg2(p0[0],p0[1],p1[0],p1[1],mx,my)<=s2 or
            _dseg2(p1[0],p1[1],p2[0],p2[1],mx,my)<=s2 or
            _dseg2(p2[0],p2[1],p0[0],p0[1],mx,my)<=s2)


# ---------------------------------------------------------------------------
# MeshCanvas — interactive software 3D renderer
# ---------------------------------------------------------------------------

class MeshCanvas(tk.Canvas):
    """Drag to rotate · scroll to zoom · double-click to reset."""

    MAX_TRIS      = 10_000   # cap total stored triangles
    MAX_DRAG_TRIS = 3_000    # max draw-calls during drag (keeps <~20ms/frame)

    def __init__(self, parent, **kw):
        super().__init__(parent, background="#12121e", cursor="fleur", **kw)
        self._verts: List[Tuple] = []
        # tris: 4- or 5-tuple ((i0,i1,i2), color_or_tex_idx, (nx,ny,nz), uvs_or_None [, alpha])
        # alpha: 0.0=opaque … 1.0=fully transparent (optional 5th element)
        # uvs_or_None: ((u0,v0),(u1,v1),(u2,v2)) or None
        # color_or_tex_idx: (r,g,b) or int index into self._textures
        self._tris:       List = []
        self._transp_tris: List = []  # 5-tuples for transparent geo (alpha > 0)
        self._textures: List = []   # list of np.ndarray (h,w,4) uint8
        self._info   = ""
        self._tex_anim: Dict[int, List] = {}  # tex_idx → [frame_arr, ...]
        self._tex_frame: int = 0
        self._anim_id = None
        self.wireframe = False
        # Editor mode
        self._edit_mode   = False
        self._selected_key: Optional[str] = None
        self._hidden_keys: set = set()
        self._transform_mode: Optional[str] = None  # 'R' or 'S'
        self._tx_axis:       Optional[str]  = None  # 'X','Y','Z' constraint
        self._tx_start:      Optional[tuple] = None
        self._tx_centroid:   Optional[tuple] = None
        self._tx_vrange:     Optional[tuple] = None
        self._tx_verts_snap: Optional[list]  = None
        self._pv_last:       Optional[list]  = None
        self._tris_all:     List = []
        self._tri_keys_all: List = []
        self._ttris_all:    List = []
        self._ttri_keys_all:List = []
        self._tri_keys:     List = []
        self._ttri_keys:    List = []
        self._shape_vrange: Dict[str, tuple] = {}
        self.on_transform_applied = None   # callback(key, new_verts)
        self.on_selection_changed = None   # callback(key or None)
        self.textured  = True       # use PIL rasterizer when textures available
        self._rendered = None       # keep PhotoImage ref
        self._cam_pos   = [0.0, 0.0, 0.0]
        self._cam_yaw   = 25.0
        self._cam_pitch = -20.0
        self._cam_spd   = None          # set lazily in _home()
        self._cam_init  = False         # first redraw after a mesh load triggers _home()
        self._drag_look = None          # (x0, y0, yaw0, pitch0) for look drag
        self._redraw_id  = None         # pending after() id for throttled redraws
        self._dragging   = False        # True while mouse button is held — triggers half-res render
        self.bind("<ButtonPress-1>",   self._press)
        self.bind("<B1-Motion>",       self._move)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<MouseWheel>",      self._wheel)
        self.bind("<Button-4>",        self._wheel)
        self.bind("<Button-5>",        self._wheel)
        self.bind("<Configure>",       lambda e: self.redraw())
        self.bind("<Double-Button-1>", lambda e: self._home())
        self.bind("<KeyPress>",        self._key_press)

    # --- public API --------------------------------------------------------

    def load_mesh(self, verts, tris, info="", textures=None):
        """verts: [(x,y,z)…]
        tris: 4- or 5-tuple  ((i0,i1,i2), color_or_tex_idx, normal, uvs [, alpha])
              alpha 5th element: 0.0=opaque … 1.0=fully transparent; default 0.0
        textures: list of np.ndarray (h,w,4) uint8, indexed by tri's tex_idx
        """
        self._stop_anim()
        self._selected_key = None
        self._hidden_keys  = set()
        self._pv_last      = None
        self._cam_init     = False      # re-position camera from new mesh bounds
        self._edit_mode_reset()
        self._verts = list(verts)
        opaque, transp = [], []
        okeys, tkeys   = [], []
        for t in tris[:self.MAX_TRIS]:
            fa  = float(t[4]) if len(t) > 4 else 0.0
            sk  = t[5]        if len(t) > 5 else None
            core = (t[0], t[1], t[2], t[3] if len(t) > 3 else None)
            if fa > 0.0:
                transp.append(core + (fa,)); tkeys.append(sk)
            else:
                opaque.append(core); okeys.append(sk)
        self._tris_all      = opaque; self._tri_keys_all  = okeys
        self._ttris_all     = transp; self._ttri_keys_all = tkeys
        self._tris          = list(opaque); self._tri_keys  = list(okeys)
        self._transp_tris   = list(transp); self._ttri_keys = list(tkeys)
        self._textures = list(textures) if textures else []
        self._info  = info
        self._precompute_arrays()
        self._home()

    def _precompute_arrays(self):
        """Cache numpy arrays for the flat-shaded tris so redraw can skip Python loops."""
        self._np_flat = None   # (N_flat, 3+3+3) → idx0,idx1,idx2, r,g,b, nx,ny,nz
        self._np_tex  = None   # keep textured tris as list (harder to vectorise)
        if not HAS_NUMPY or not self._tris:
            return
        flat_rows = []
        tex_rows  = []
        for tri in self._tris:
            (i0,i1,i2), col_or_idx, nrm, uvs = tri
            if isinstance(col_or_idx, tuple):
                flat_rows.append((i0, i1, i2, col_or_idx[0], col_or_idx[1], col_or_idx[2],
                                  nrm[0], nrm[1], nrm[2]))
            else:
                tex_rows.append(tri)
        if flat_rows:
            self._np_flat = np.array(flat_rows, dtype=np.float32)  # (N,9)
        if tex_rows:
            self._np_tex = tex_rows

    def clear(self):
        self._stop_anim()
        self._verts = []; self._tris = []; self._transp_tris = []; self._info = ""
        self.delete("all")

    def set_tex_anim(self, anim_map: dict):
        """Set animated texture frames. anim_map: {tex_idx: [np.ndarray, ...]}"""
        self._tex_anim  = anim_map or {}
        self._tex_frame = 0

    def _stop_anim(self):
        if self._anim_id is not None:
            self.after_cancel(self._anim_id)
            self._anim_id = None
        self._tex_anim  = {}
        self._tex_frame = 0

    def toggle_anim(self):
        """F7: start or stop texture animation."""
        if self._anim_id is not None:
            self._stop_anim()
        elif self._tex_anim:
            self._anim_tick()

    def _anim_tick(self):
        if not self._tex_anim:
            self._anim_id = None
            return
        self._tex_frame += 1
        for tidx, frames in self._tex_anim.items():
            if frames and tidx < len(self._textures):
                self._textures[tidx] = frames[self._tex_frame % len(frames)]
        self.redraw()
        self._anim_id = self.after(125, self._anim_tick)

    # --- editor mode ----------------------------------------------------------

    def set_shape_vranges(self, d: dict):
        """d: {shape_key: (base_v, end_v)} — set by editor after load_mesh."""
        self._shape_vrange = d or {}

    def _edit_mode_reset(self):
        self._transform_mode = None
        self._tx_axis = None
        self._tx_start = self._tx_centroid = self._tx_vrange = self._tx_verts_snap = None

    def toggle_edit_mode(self):
        """F8: enter/exit editor pick mode."""
        self._edit_mode = not self._edit_mode
        if self._edit_mode:
            self.configure(cursor="crosshair")
        else:
            self._cancel_transform()
            self._selected_key = None
            if self.on_selection_changed:
                self.on_selection_changed(None)
            self.configure(cursor="fleur")
        self.redraw()

    def _rebuild_visible(self):
        """Rebuild _tris / _transp_tris filtering out hidden shapes, then recompute arrays."""
        if not self._hidden_keys:
            self._tris        = list(self._tris_all);  self._tri_keys  = list(self._tri_keys_all)
            self._transp_tris = list(self._ttris_all); self._ttri_keys = list(self._ttri_keys_all)
        else:
            vo = [(t,k) for t,k in zip(self._tris_all,  self._tri_keys_all)  if k not in self._hidden_keys]
            vt = [(t,k) for t,k in zip(self._ttris_all, self._ttri_keys_all) if k not in self._hidden_keys]
            self._tris        = [t for t,_ in vo]; self._tri_keys  = [k for _,k in vo]
            self._transp_tris = [t for t,_ in vt]; self._ttri_keys = [k for _,k in vt]
        self._precompute_arrays()

    def toggle_hidden(self, key):
        """Hide or un-hide a shape by key. Hidden shapes are excluded from render."""
        if key in self._hidden_keys:
            self._hidden_keys.discard(key)
        else:
            self._hidden_keys.add(key)
            if key == self._selected_key:
                self._selected_key = None
                if self.on_selection_changed:
                    self.on_selection_changed(None)
        self._rebuild_visible()
        self.redraw()

    def begin_transform(self, mode: str):
        """Start R (rotate-Y), S (uniform-scale), or M (translate-XZ) for the selected shape."""
        key = self._selected_key
        if not key or key not in self._shape_vrange:
            return
        base_v, end_v = self._shape_vrange[key]
        sub = self._verts[base_v:end_v]
        if not sub:
            return
        cx = sum(v[0] for v in sub) / len(sub)
        cy = sum(v[1] for v in sub) / len(sub)
        cz = sum(v[2] for v in sub) / len(sub)
        self._transform_mode  = mode
        self._tx_axis         = None
        self._tx_vrange       = (base_v, end_v)
        self._tx_centroid     = (cx, cy, cz)
        self._tx_verts_snap   = list(self._verts)
        self._tx_span         = getattr(self, '_last_span', 1.0)
        self._tx_start        = None
        self.bind("<Motion>", self._tx_motion)
        self.focus_set()

    def _tx_motion(self, e):
        if not self._transform_mode:
            return
        if self._tx_start is None:
            self._tx_start = (e.x, e.y); return
        import math as _m
        dx = e.x - self._tx_start[0]
        base_v, end_v = self._tx_vrange
        cx, cy, cz = self._tx_centroid
        if self._transform_mode == 'R':
            angle = dx * 0.01
            ca, sa = _m.cos(angle), _m.sin(angle)
            for i in range(base_v, end_v):
                vx, vy, vz = self._tx_verts_snap[i]
                lx, lz = vx - cx, vz - cz
                self._verts[i] = (cx + lx*ca - lz*sa, vy, cz + lx*sa + lz*ca)
        elif self._transform_mode == 'S':
            factor = _m.exp(dx * 0.005)
            for i in range(base_v, end_v):
                vx, vy, vz = self._tx_verts_snap[i]
                self._verts[i] = (cx+(vx-cx)*factor, cy+(vy-cy)*factor, cz+(vz-cz)*factor)
        elif self._transform_mode == 'M':
            dy = e.y - self._tx_start[1]
            ax = self._tx_axis
            tcx, tcy, tcz = self._tx_centroid
            if ax in ('X', 'Y', 'Z'):
                # Project the world axis onto screen space; map mouse drag
                # onto that screen direction to get world-space movement.
                # This works at any camera angle (no depth-scale singularity).
                giz = max(getattr(self, '_last_span', 100.) * 0.07, 2.)
                orig = self._screen_pt(tcx, tcy, tcz)
                if ax == 'X':
                    ep = self._screen_pt(tcx + giz, tcy, tcz)
                elif ax == 'Y':
                    ep = self._screen_pt(tcx, tcy + giz, tcz)
                else:
                    ep = self._screen_pt(tcx, tcy, tcz + giz)
                move = 0.0
                if orig and ep:
                    asx = ep[0] - orig[0]; asy = ep[1] - orig[1]
                    alen = _m.sqrt(asx*asx + asy*asy)
                    if alen > 0.5:
                        # project drag onto screen axis; convert to world units
                        proj_px = (dx*asx + dy*asy) / alen
                        move = proj_px * giz / alen
                for i in range(base_v, end_v):
                    vx, vy, vz = self._tx_verts_snap[i]
                    if ax == 'X':
                        self._verts[i] = (vx + move, vy,        vz       )
                    elif ax == 'Y':
                        self._verts[i] = (vx,        vy + move, vz       )
                    else:
                        self._verts[i] = (vx,        vy,        vz + move)
            else:
                # Free mode: move in screen-space right/up, projected to world
                w_ = self.winfo_width(); h_ = self.winfo_height()
                focal = min(w_, h_) / (2.0 * _m.tan(_m.radians(30.0)))
                cp = self._cam_pos
                ddx = tcx-cp[0]; ddy = tcy-cp[1]; ddz = tcz-cp[2]
                yr2 = _m.radians(self._cam_yaw); pr2 = _m.radians(self._cam_pitch)
                cy2, sy2 = _m.cos(yr2), _m.sin(yr2)
                cp2, sp2 = _m.cos(pr2), _m.sin(pr2)
                depth = max(1.0, (-sy2*cp2)*ddx + sp2*ddy + cy2*cp2*ddz)
                scale = depth / max(focal, 1.0)
                # camera right vector: (cos yaw, 0, sin yaw)
                # camera up vector:    (sin yaw*sin pitch, cos pitch, -cos yaw*sin pitch)
                rx = cy2; rz = sy2
                ux = sy2*sp2; uy = cp2; uz = -cy2*sp2
                for i in range(base_v, end_v):
                    vx, vy, vz = self._tx_verts_snap[i]
                    self._verts[i] = (vx + (dx*rx - dy*ux)*scale,
                                      vy + (-dy*uy)*scale,
                                      vz + (dx*rz - dy*uz)*scale)
        self._schedule_redraw()

    def _apply_transform(self):
        if not self._transform_mode:
            return
        self.unbind("<Motion>")
        key = self._selected_key
        if self.on_transform_applied and key:
            self.on_transform_applied(key, list(self._verts))
        self._edit_mode_reset()
        self.redraw()

    def _cancel_transform(self):
        if not self._transform_mode:
            return
        if self._tx_verts_snap:
            self._verts = list(self._tx_verts_snap)
        self.unbind("<Motion>")
        self._edit_mode_reset()
        self.redraw()

    def _edit_click(self, e):
        """Select the frontmost front-facing tri under the cursor."""
        if self._transform_mode:
            self._apply_transform(); return
        if not self._pv_last:
            return
        mx, my = float(e.x), float(e.y)
        pv = self._pv_last

        def _scan(slack):
            bz = None; bk = None
            for tri, key in zip(self._tris, self._tri_keys):
                i0, i1, i2 = tri[0]
                if max(i0, i1, i2) >= len(pv): continue
                p0, p1, p2 = pv[i0], pv[i1], pv[i2]
                ax, ay = p1[0]-p0[0], p1[1]-p0[1]
                bx, by = p2[0]-p0[0], p2[1]-p0[1]
                if ax*by - ay*bx > 0: continue          # back-face
                z = (p0[2]+p1[2]+p2[2]) / 3
                if z <= 0: continue                      # behind camera
                if _pt_in_tri_2d(mx, my, p0, p1, p2, slack):
                    if bz is None or z < bz:
                        bz = z; bk = key
            return bk

        # exact hit first; 4px edge slack only if nothing hit exactly
        best_key = _scan(0.0) or _scan(4.0)
        self._selected_key = best_key
        if self.on_selection_changed:
            self.on_selection_changed(best_key)
        self.redraw()

    def _screen_pt(self, wx, wy, wz):
        """Project a world-space point to (sx, sy), or None if behind camera."""
        import math as _m
        w = self.winfo_width(); h = self.winfo_height()
        cx, cy = w/2.0, h/2.0
        fov = min(w,h) / (2.0 * _m.tan(_m.radians(30.0)))
        yr = _m.radians(self._cam_yaw); pr = _m.radians(self._cam_pitch)
        cy_,sy_ = _m.cos(yr),_m.sin(yr)
        cp_,sp_ = _m.cos(pr),_m.sin(pr)
        R = [[cy_,0.,sy_],[sy_*sp_,cp_,-cy_*sp_],[-sy_*cp_,sp_,cy_*cp_]]
        dx,dy,dz = wx-self._cam_pos[0], wy-self._cam_pos[1], wz-self._cam_pos[2]
        rx = R[0][0]*dx+R[0][1]*dy+R[0][2]*dz
        ry = R[1][0]*dx+R[1][1]*dy+R[1][2]*dz
        rz = R[2][0]*dx+R[2][1]*dy+R[2][2]*dz
        if rz <= 0.001: return None
        return (cx + rx*fov/rz, cy - ry*fov/rz)

    def _draw_edit_overlays(self, pv):
        """Draw selection highlight and hidden-mesh wires over the rendered image."""
        if not self._edit_mode or not pv:
            return
        # Hidden meshes: dark-blue wireframe
        for tri, key in zip(self._tris_all, self._tri_keys_all):
            if key not in self._hidden_keys: continue
            i0, i1, i2 = tri[0]
            if max(i0, i1, i2) >= len(pv): continue
            p0, p1, p2 = pv[i0], pv[i1], pv[i2]
            self.create_polygon(p0[0],p0[1],p1[0],p1[1],p2[0],p2[1],
                                fill="", outline="#2244aa", width=1)
        # Selected shape: yellow stipple fill
        if self._selected_key:
            for tri, key in zip(self._tris, self._tri_keys):
                if key != self._selected_key: continue
                i0, i1, i2 = tri[0]
                if max(i0, i1, i2) >= len(pv): continue
                p0, p1, p2 = pv[i0], pv[i1], pv[i2]
                ax, ay = p1[0]-p0[0], p1[1]-p0[1]
                bx, by = p2[0]-p0[0], p2[1]-p0[1]
                if ax*by - ay*bx > 0: continue
                self.create_polygon(p0[0],p0[1],p1[0],p1[1],p2[0],p2[1],
                                    fill="#ffdd00", outline="#ffdd00",
                                    stipple="gray25")
        # XYZ gizmo when a transform is active
        if self._transform_mode and self._tx_centroid:
            tcx, tcy, tcz = self._tx_centroid
            giz = max(getattr(self, '_last_span', 100.) * 0.07, 2.)
            orig = self._screen_pt(tcx, tcy, tcz)
            if orig:
                cur_ax = self._tx_axis
                for (wx,wy,wz), color, lbl, akey in [
                        ((tcx+giz, tcy,     tcz    ), '#ff4444', 'X', 'X'),
                        ((tcx,     tcy+giz, tcz    ), '#44ee44', 'Y', 'Y'),
                        ((tcx,     tcy,     tcz+giz), '#4499ff', 'Z', 'Z')]:
                    ep = self._screen_pt(wx, wy, wz)
                    if ep:
                        lw = 3 if cur_ax == akey else 1
                        self.create_line(orig[0],orig[1],ep[0],ep[1],
                                         fill=color, width=lw, arrow='last')
                        self.create_text(ep[0]+5, ep[1], text=lbl,
                                         fill=color, font=('Courier',9,'bold'), anchor='w')

    # --- transparent geometry renderer ----------------------------------------

    def _render_transp_into_fb(self, fb, zbuf2d, pv_s, rn, light):
        """Blend transparent tris into the numpy fb with per-pixel z-testing."""
        if not self._transp_tris or not HAS_NUMPY:
            return
        rh, rw = fb.shape[:2]
        tex_avg: dict = {}
        for t in self._transp_tris:
            ck = t[1]
            if isinstance(ck, int) and 0 <= ck < len(self._textures) and ck not in tex_avg:
                tx = self._textures[ck]
                tex_avg[ck] = np.array([tx[:,:,0].mean(), tx[:,:,1].mean(), tx[:,:,2].mean()], dtype=np.float32)
        def _d(t):
            i0,i1,i2 = t[0]
            if max(i0,i1,i2) >= len(pv_s): return -1e9
            return (pv_s[i0][2]+pv_s[i1][2]+pv_s[i2][2]) / 3
        for t in sorted(self._transp_tris, key=_d):
            (i0,i1,i2), col_key, nrm, _uvs, fa = t
            if max(i0,i1,i2) >= len(pv_s): continue
            p0,p1,p2 = pv_s[i0],pv_s[i1],pv_s[i2]
            sx0,sy0,_,zv0=p0; sx1,sy1,_,zv1=p1; sx2,sy2,_,zv2=p2
            ax,ay=sx1-sx0,sy1-sy0; bx,by=sx2-sx0,sy2-sy0
            if ax*by - ay*bx > 0: continue
            rnx,rny,rnz = rn(nrm)
            sh = 0.35 + 0.65 * max(0.0, rnx*light[0]+rny*light[1]+rnz*light[2])
            if isinstance(col_key, int) and col_key in tex_avg:
                base = tex_avg[col_key]
            elif isinstance(col_key, tuple):
                base = np.array(col_key, dtype=np.float32)
            else:
                base = np.array([160.0, 160.0, 160.0], dtype=np.float32)
            tri_rgb = np.clip(base * sh + np.array([18, 12, 25], dtype=np.float32), 0, 255)
            bx0=max(0,int(min(sx0,sx1,sx2))); bx1=min(rw,int(max(sx0,sx1,sx2))+2)
            by0=max(0,int(min(sy0,sy1,sy2))); by1=min(rh,int(max(sy0,sy1,sy2))+2)
            if bx0>=bx1 or by0>=by1: continue
            xs=np.arange(bx0,bx1,dtype=np.float32)+0.5
            ys=np.arange(by0,by1,dtype=np.float32)+0.5
            XX,YY=np.meshgrid(xs,ys)
            denom=(sy1-sy2)*(sx0-sx2)+(sx2-sx1)*(sy0-sy2)
            if abs(denom)<1e-6: continue
            inv_d=1.0/denom
            w0=((sy1-sy2)*(XX-sx2)+(sx2-sx1)*(YY-sy2))*inv_d
            w1=((sy2-sy0)*(XX-sx2)+(sx0-sx2)*(YY-sy2))*inv_d
            w2=1.0-w0-w1
            inside=(w0>=0)&(w1>=0)&(w2>=0)
            if not inside.any(): continue
            rows,cols=np.where(inside)
            fb_r=rows+by0; fb_c=cols+bx0
            b0=w0[rows,cols]; b1=w1[rows,cols]; b2=w2[rows,cols]
            iz=b0/zv0+b1/zv1+b2/zv2
            z_val=(1.0/np.maximum(iz,1e-6)).astype(np.float32)
            # only draw where transparent tri is in front of opaque geometry
            in_front = z_val < zbuf2d[fb_r, fb_c]
            if not in_front.any(): continue
            fr=fb_r[in_front]; fc=fb_c[in_front]
            # alpha blend: fa=0 → opaque tri, fa→1 → more transparent
            existing = fb[fr,fc].astype(np.float32)
            blended = np.clip(tri_rgb*(1.0-fa) + existing*fa, 0, 255).astype(np.uint8)
            fb[fr,fc] = blended

    # --- private -----------------------------------------------------------

    def _home(self):
        self._cam_yaw   = 25.0
        self._cam_pitch = -20.0
        if self._verts:
            xs = [v[0] for v in self._verts]; ys = [v[1] for v in self._verts]; zs = [v[2] for v in self._verts]
            ocx = (max(xs)+min(xs))/2; ocy = (max(ys)+min(ys))/2; ocz = (max(zs)+min(zs))/2
            span = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs), 1.0)
            self._last_span = span
            self._cam_spd = max(span * 0.04, 1.0)
            yr0 = math.radians(25.0); pr0 = math.radians(-20.0)
            cy0, sy0 = math.cos(yr0), math.sin(yr0)
            cp0, sp0 = math.cos(pr0), math.sin(pr0)
            # place camera behind the look direction so the mesh centre is in front
            dist = span * 1.5
            self._cam_pos = [ocx + sy0*cp0*dist, ocy - sp0*dist, ocz - cy0*cp0*dist]
        self._cam_init = True
        self.redraw()

    def redraw(self):
        w = self.winfo_width(); h = self.winfo_height()
        if w < 2 or h < 2: return
        self.delete("all")

        if not self._tris:
            self.create_text(w//2, h//2, text="No mesh", fill="#404060",
                             font=("Helvetica", 14))
            return

        cx, cy = w / 2.0, h / 2.0
        verts = self._verts

        xs = [v[0] for v in verts]; ys = [v[1] for v in verts]; zs = [v[2] for v in verts]
        span = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs), 0.001)
        self._last_span = span

        # lazy camera init: position from mesh bounds on first render after load
        if not self._cam_init:
            self._home()
            return  # _home() calls redraw()

        yr = math.radians(self._cam_yaw); pr = math.radians(self._cam_pitch)
        cy_, sy_ = math.cos(yr), math.sin(yr)
        cp_, sp_ = math.cos(pr), math.sin(pr)
        R = [[cy_, 0.0, sy_],
             [sy_*sp_, cp_, -cy_*sp_],
             [-sy_*cp_, sp_,  cy_*cp_]]

        # focal length for 60° vertical FOV
        fov = min(w, h) / (2.0 * math.tan(math.radians(30.0)))
        light = (0.57, 0.57, 0.57)

        def rn(n):
            nx=R[0][0]*n[0]+R[0][1]*n[1]+R[0][2]*n[2]
            ny=R[1][0]*n[0]+R[1][1]*n[1]+R[1][2]*n[2]
            nz=R[2][0]*n[0]+R[2][1]*n[1]+R[2][2]*n[2]
            return nx, ny, nz

        # Vectorised vertex transform (numpy when available, else Python loop)
        # near_clip: at least 1 unit so behind-camera vertices don't project to
        # extreme screen coords that overflow Tkinter/X11's int16 coordinates
        near_clip = max(span * 0.001, 1.0)
        cp_arr = self._cam_pos
        if HAS_NUMPY and verts:
            V = np.array(verts, dtype=np.float32)
            V -= np.array(cp_arr, dtype=np.float32)
            Rm = np.array(R, dtype=np.float32)
            RV = V @ Rm.T               # (N,3)
            rx_a, ry_a, rz_a = RV[:,0], RV[:,1], RV[:,2]
            z_v_a = np.maximum(near_clip, rz_a)
            sx_a  = np.clip(cx + rx_a * fov / z_v_a, -32767.0, w + 32767.0)
            sy_a  = np.clip(cy - ry_a * fov / z_v_a, -32767.0, h + 32767.0)
            pv = list(zip(sx_a.tolist(), sy_a.tolist(), rz_a.tolist(), z_v_a.tolist()))
        else:
            def xf(v):
                dx=v[0]-cp_arr[0]; dy=v[1]-cp_arr[1]; dz=v[2]-cp_arr[2]
                rx=R[0][0]*dx+R[0][1]*dy+R[0][2]*dz
                ry=R[1][0]*dx+R[1][1]*dy+R[1][2]*dz
                rz=R[2][0]*dx+R[2][1]*dy+R[2][2]*dz
                z_v = max(near_clip, rz)
                return (max(-32767.0, min(w+32767.0, cx + rx*fov/z_v)),
                        max(-32767.0, min(h+32767.0, cy - ry*fov/z_v)),
                        rz, z_v)
            pv = [xf(v) for v in verts]

        self._pv_last = pv

        use_tex = (not self.wireframe and self.textured
                   and self._textures and HAS_PIL and HAS_NUMPY)

        if use_tex:
            self._render_textured(w, h, pv, rn, light)
        elif HAS_NUMPY and self._np_flat is not None and not self.wireframe:
            self._render_flat_numpy(w, h, pv, R, rn, light)
        else:
            # Python-loop fallback (wireframe or no numpy)
            draw = []
            for tri in self._tris:
                i0,i1,i2 = tri[0]
                color_or_key = tri[1]
                normal = tri[2]
                if max(i0,i1,i2) >= len(pv): continue
                p0, p1, p2 = pv[i0], pv[i1], pv[i2]
                ax,ay = p1[0]-p0[0], p1[1]-p0[1]
                bx,by = p2[0]-p0[0], p2[1]-p0[1]
                if not self.wireframe and ax*by - ay*bx > 0: continue
                depth = (p0[2]+p1[2]+p2[2]) / 3
                pts = (p0[0],p0[1], p1[0],p1[1], p2[0],p2[1])
                if self.wireframe:
                    draw.append((depth, pts, None))
                else:
                    color = color_or_key if isinstance(color_or_key, tuple) else (160,160,160)
                    rnx,rny,rnz = rn(normal)
                    diffuse = max(0.0, rnx*light[0]+rny*light[1]+rnz*light[2])
                    shade = 0.35 + 0.65 * diffuse
                    r = min(255, int(color[0]*shade + 18))
                    g = min(255, int(color[1]*shade + 12))
                    b = min(255, int(color[2]*shade + 25))
                    draw.append((depth, pts, (r,g,b)))
            draw.sort(key=lambda d: -d[0])
            if HAS_PIL:
                from PIL import Image as _Img, ImageDraw as _ID, ImageTk as _ITk
                img = _Img.new("RGB", (w, h), (18, 18, 30))
                d = _ID.Draw(img)
                for _, pts, color in draw:
                    x0,y0,x1,y1,x2,y2 = pts
                    if color is None:
                        d.line([(x0,y0),(x1,y1)], fill=(51,85,170))
                        d.line([(x1,y1),(x2,y2)], fill=(51,85,170))
                        d.line([(x2,y2),(x0,y0)], fill=(51,85,170))
                    else:
                        d.polygon([(x0,y0),(x1,y1),(x2,y2)], fill=color)
                self._rendered = _ITk.PhotoImage(img)
                self.create_image(0, 0, anchor="nw", image=self._rendered)
            else:
                for _, pts, color in draw:
                    if color is None:
                        x0,y0,x1,y1,x2,y2 = pts
                        self.create_line(x0,y0,x1,y1, fill="#3355aa", width=1)
                        self.create_line(x1,y1,x2,y2, fill="#3355aa", width=1)
                        self.create_line(x2,y2,x0,y0, fill="#3355aa", width=1)
                    else:
                        self.create_polygon(*pts,
                                            fill=f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                                            outline="")

        clipped = "⚠ capped " if len(self._tris) == self.MAX_TRIS else ""
        tex_note = " · textured" if use_tex else ""
        stat = (f"{self._info}  " if self._info else "") + \
               f"{clipped}{len(self._tris):,} tris · {len(verts):,} verts · " + \
               ("wire" if self.wireframe else "shaded") + tex_note + \
               "  drag=look  wasd/arrows=move  spc=up  scroll=speed  dbl=reset"
        self.create_text(6, h-6, anchor="sw", text=stat, fill="#444466",
                         font=("Courier", 9))
        self._draw_edit_overlays(pv)
        if self._edit_mode:
            mode_labels = {'R': 'ROTATE', 'S': 'SCALE', 'M': 'MOVE'}
            if self._transform_mode:
                ax_label = f' [{self._tx_axis}]' if self._tx_axis else '  X/Y/Z=axis'
                hud = f"{mode_labels.get(self._transform_mode, self._transform_mode)}{ax_label} — drag · Enter=apply · Esc=cancel"
            elif self._selected_key:
                hud = f"EDIT  selected: {self._selected_key}  · R=rotate · S=scale · M=move · H=hide · C=copy · Esc=deselect"
            else:
                hud = "EDIT MODE — click to select mesh · C=copy  P=paste · F8=exit  ←→↑↓=move  Spc=up  Alt=down"
            self.create_text(w//2, 14, anchor="n", text=hud,
                             fill="#ffee44", font=("Courier", 10, "bold"))

    def _render_flat_numpy(self, w, h, pv, R, rn, light):
        """Vectorised flat-shaded renderer: numpy for cull/shade/sort, C/PIL for fill."""
        from PIL import Image as _Img, ImageDraw as _ID, ImageTk as _ITk

        # Half-res during drag only when no C rasterizer is available
        sc = 1.0 if _RASTER_LIB else (0.5 if self._dragging else 1.0)
        rw, rh = max(1, int(w * sc)), max(1, int(h * sc))

        F = self._np_flat          # (N, 9): i0,i1,i2, r,g,b, nx,ny,nz
        pv_arr = np.array(pv, dtype=np.float32)  # (V, 4)

        max_v = len(pv_arr) - 1
        i0 = np.clip(F[:, 0].astype(np.int32), 0, max_v)
        i1 = np.clip(F[:, 1].astype(np.int32), 0, max_v)
        i2 = np.clip(F[:, 2].astype(np.int32), 0, max_v)
        valid = (F[:, 0] <= max_v) & (F[:, 1] <= max_v) & (F[:, 2] <= max_v)

        p0 = pv_arr[i0]; p1 = pv_arr[i1]; p2 = pv_arr[i2]
        sx = np.stack([p0[:,0], p1[:,0], p2[:,0]], axis=1) * sc
        sy = np.stack([p0[:,1], p1[:,1], p2[:,1]], axis=1) * sc
        rz = np.stack([p0[:,2], p1[:,2], p2[:,2]], axis=1)

        ax = sx[:,1]-sx[:,0]; ay = sy[:,1]-sy[:,0]
        bx = sx[:,2]-sx[:,0]; by = sy[:,2]-sy[:,0]
        valid &= (ax*by - ay*bx) <= 0          # backface cull
        depth = rz.sum(axis=1)

        normals = F[:, 6:9]
        Rm      = np.array(R, dtype=np.float32)
        rn_all  = normals @ Rm.T
        lv      = np.array(light, dtype=np.float32)
        diffuse = np.maximum(0.0, rn_all @ lv)
        shade_f = (0.35 + 0.65 * diffuse)[:, np.newaxis]
        base_rgb = F[:, 3:6]
        shaded   = np.clip(base_rgb * shade_f + np.array([18, 12, 25], dtype=np.float32),
                           0, 255).astype(np.uint8)

        vis_idx    = np.where(valid)[0]
        sort_order = vis_idx[np.argsort(-depth[vis_idx])]

        if _RASTER_LIB is not None:
            import ctypes
            fb   = np.full((rh, rw, 3), [18, 18, 30], dtype=np.uint8)
            sx_s = np.ascontiguousarray(sx[sort_order], dtype=np.float32)
            sy_s = np.ascontiguousarray(sy[sort_order], dtype=np.float32)
            rgb_s= np.ascontiguousarray(shaded[sort_order], dtype=np.uint8)
            _RASTER_LIB.raster_flat(
                fb.ctypes.data_as(ctypes.c_void_p), rw, rh,
                sx_s.ctypes.data_as(ctypes.c_void_p),
                sy_s.ctypes.data_as(ctypes.c_void_p),
                rgb_s.ctypes.data_as(ctypes.c_void_p),
                len(sort_order))
            img = _Img.fromarray(fb, 'RGB')
        else:
            img = _Img.new("RGB", (rw, rh), (18, 18, 30))
            d   = _ID.Draw(img)
            for i in sort_order:
                d.polygon([(sx[i,0], sy[i,0]), (sx[i,1], sy[i,1]), (sx[i,2], sy[i,2])],
                          fill=(int(shaded[i,0]), int(shaded[i,1]), int(shaded[i,2])))
            if sc < 1.0:
                img = img.resize((w, h), _Img.NEAREST)
        self._rendered = _ITk.PhotoImage(img)
        self.create_image(0, 0, anchor="nw", image=self._rendered)

    def _render_textured(self, w, h, pv, rn, light):
        """Z-buffered textured renderer: C raster_tex fast path, numpy fallback."""
        import numpy as np
        from PIL import Image, ImageTk

        # Half-res during drag only when no C rasterizer available
        sc = 1.0 if _RASTER_LIB else (0.5 if self._dragging else 1.0)
        rw, rh = max(1, int(w * sc)), max(1, int(h * sc))

        fb   = np.full((rh, rw, 3), [18, 18, 30], dtype=np.uint8)
        zbuf = np.full((rh * rw,), np.finfo(np.float32).max, dtype=np.float32)

        pv_s = [(sx*sc, sy_*sc, rz_, zv) for sx,sy_,rz_,zv in pv] if sc < 1.0 else pv

        if _RASTER_LIB is not None:
            import ctypes
            # Build per-triangle arrays for the C rasterizer
            # Group by: textured (int key) vs flat (tuple key)
            tex_groups: dict = {}   # tex_idx → lists of (sx0..sx2, sy0..sy2, iz0..iz2, uvs, shade)
            flat_rows = []          # (sx0..sx2, sy0..sy2, shade, color)

            for tri in self._tris:
                i0,i1,i2 = tri[0]
                col_key   = tri[1]; nrm = tri[2]; uvs = tri[3]
                if max(i0,i1,i2) >= len(pv_s): continue
                p0,p1,p2 = pv_s[i0],pv_s[i1],pv_s[i2]
                sx0,sy0,_,zv0=p0; sx1,sy1,_,zv1=p1; sx2,sy2,_,zv2=p2
                ax,ay=sx1-sx0,sy1-sy0; bx,by=sx2-sx0,sy2-sy0
                if ax*by - ay*bx > 0: continue
                rnx,rny,rnz=rn(nrm)
                sh = 0.35 + 0.65*max(0.0, rnx*light[0]+rny*light[1]+rnz*light[2])
                iz0,iz1,iz2 = 1.0/zv0, 1.0/zv1, 1.0/zv2
                if isinstance(col_key, int) and uvs is not None and 0<=col_key<len(self._textures):
                    g = tex_groups.setdefault(col_key, [])
                    uv0,uv1,uv2 = uvs
                    g.append((sx0,sy0,sx1,sy1,sx2,sy2, iz0,iz1,iz2,
                               uv0[0],uv0[1],uv1[0],uv1[1],uv2[0],uv2[1], sh))
                else:
                    color = col_key if isinstance(col_key, tuple) else (160,160,160)
                    flat_rows.append((sx0,sy0,sx1,sy1,sx2,sy2, iz0,iz1,iz2, color, sh))

            # Render textured groups via C
            for tidx, rows in tex_groups.items():
                tex = self._textures[tidx]
                if not tex.flags['C_CONTIGUOUS']:
                    tex = np.ascontiguousarray(tex)
                n = len(rows)
                sx_a  = np.array([(r[0],r[2],r[4]) for r in rows], dtype=np.float32)
                sy_a  = np.array([(r[1],r[3],r[5]) for r in rows], dtype=np.float32)
                iz_a  = np.array([(r[6],r[7],r[8]) for r in rows], dtype=np.float32)
                uv_a  = np.array([(r[9],r[10],r[11],r[12],r[13],r[14]) for r in rows], dtype=np.float32)
                sh_a  = np.array([r[15] for r in rows], dtype=np.float32)
                th, tw = tex.shape[:2]
                _RASTER_LIB.raster_tex(
                    fb.ctypes.data_as(ctypes.c_void_p),
                    zbuf.ctypes.data_as(ctypes.c_void_p),
                    rw, rh,
                    sx_a.ctypes.data_as(ctypes.c_void_p),
                    sy_a.ctypes.data_as(ctypes.c_void_p),
                    iz_a.ctypes.data_as(ctypes.c_void_p),
                    uv_a.ctypes.data_as(ctypes.c_void_p),
                    tex.ctypes.data_as(ctypes.c_void_p), tw, th,
                    sh_a.ctypes.data_as(ctypes.c_void_p),
                    n)

            # Render flat (untextured) tris via flat C rasterizer (painter's on top)
            if flat_rows:
                zbuf2d = zbuf.reshape(rh, rw)
                # Back-to-front sort by average iz (deeper z_view = smaller iz)
                flat_rows.sort(key=lambda r: -(r[6]+r[7]+r[8]))
                n = len(flat_rows)
                sx_a  = np.array([(r[0],r[2],r[4]) for r in flat_rows], dtype=np.float32)
                sy_a  = np.array([(r[1],r[3],r[5]) for r in flat_rows], dtype=np.float32)
                rgb_a = np.zeros((n,3), dtype=np.uint8)
                for k,r in enumerate(flat_rows):
                    c,sh = r[9],r[10]
                    rgb_a[k] = (min(255,int(c[0]*sh)+18), min(255,int(c[1]*sh)+12), min(255,int(c[2]*sh)+25))
                _RASTER_LIB.raster_flat(
                    fb.ctypes.data_as(ctypes.c_void_p), rw, rh,
                    sx_a.ctypes.data_as(ctypes.c_void_p),
                    sy_a.ctypes.data_as(ctypes.c_void_p),
                    rgb_a.ctypes.data_as(ctypes.c_void_p),
                    n)
        else:
            # --- numpy fallback (original per-triangle loop) ---
            zbuf2d = zbuf.reshape(rh, rw)
            zbuf2d[:] = np.inf
            for tri in self._tris:
                i0,i1,i2 = tri[0]
                color_or_key = tri[1]; normal = tri[2]; uvs = tri[3]
                if max(i0,i1,i2) >= len(pv_s): continue
                p0,p1,p2 = pv_s[i0],pv_s[i1],pv_s[i2]
                sx0,sy0,_,zv0=p0; sx1,sy1,_,zv1=p1; sx2,sy2,_,zv2=p2
                ax,ay=sx1-sx0,sy1-sy0; bx,by=sx2-sx0,sy2-sy0
                if ax*by - ay*bx > 0: continue
                rnx,rny,rnz=rn(normal)
                shade=0.35+0.65*max(0.0,rnx*light[0]+rny*light[1]+rnz*light[2])
                tex=None; color=(160,160,160); uv0=uv1=uv2=(0.0,0.0)
                if (isinstance(color_or_key,int) and uvs is not None
                        and 0<=color_or_key<len(self._textures)):
                    tex=self._textures[color_or_key]; uv0,uv1,uv2=uvs
                elif isinstance(color_or_key,tuple):
                    color=color_or_key
                bx0=max(0,int(min(sx0,sx1,sx2))); bx1=min(rw,int(max(sx0,sx1,sx2))+2)
                by0=max(0,int(min(sy0,sy1,sy2))); by1=min(rh,int(max(sy0,sy1,sy2))+2)
                if bx0>=bx1 or by0>=by1: continue
                xs=np.arange(bx0,bx1,dtype=np.float32)+0.5
                ys=np.arange(by0,by1,dtype=np.float32)+0.5
                XX,YY=np.meshgrid(xs,ys)
                denom=(sy1-sy2)*(sx0-sx2)+(sx2-sx1)*(sy0-sy2)
                if abs(denom)<1e-6: continue
                inv_d=1.0/denom
                w0=((sy1-sy2)*(XX-sx2)+(sx2-sx1)*(YY-sy2))*inv_d
                w1=((sy2-sy0)*(XX-sx2)+(sx0-sx2)*(YY-sy2))*inv_d
                w2=1.0-w0-w1
                inside=(w0>=0)&(w1>=0)&(w2>=0)
                if not inside.any(): continue
                rows,cols=np.where(inside)
                fb_r=rows+by0; fb_c=cols+bx0
                b0=w0[rows,cols]; b1=w1[rows,cols]; b2=w2[rows,cols]
                iz0,iz1,iz2=1.0/zv0,1.0/zv1,1.0/zv2
                iz=b0*iz0+b1*iz1+b2*iz2
                iz_s=np.maximum(iz,1e-6)
                z_val=(1.0/iz_s).astype(np.float32)
                upd=z_val<zbuf2d[fb_r,fb_c]
                if not upd.any(): continue
                fr=fb_r[upd]; fc=fb_c[upd]; b0u=b0[upd]; b1u=b1[upd]; b2u=b2[upd]; izu=iz_s[upd]
                if tex is not None:
                    u0,v0_=uv0; u1,v1_=uv1; u2,v2_=uv2
                    u_arr=(b0u*(u0*iz0)+b1u*(u1*iz1)+b2u*(u2*iz2))/izu
                    v_arr=(b0u*(v0_*iz0)+b1u*(v1_*iz1)+b2u*(v2_*iz2))/izu
                    th,tw=tex.shape[:2]
                    ti=np.floor(np.mod(u_arr,1.0)*tw).astype(np.int32)
                    tj=np.floor(np.mod(v_arr,1.0)*th).astype(np.int32)
                    rgba=tex[tj,ti]
                    opq=rgba[:,3]>128 if tex.shape[2]==4 else np.ones(len(fr),dtype=bool)
                    if opq.any():
                        rgb_=np.clip(rgba[opq,:3].astype(np.float32)*shade,0,255).astype(np.uint8)
                        fb[fr[opq],fc[opq]]=rgb_; zbuf2d[fr[opq],fc[opq]]=z_val[upd][opq]
                else:
                    fb[fr,fc]=(min(255,int(color[0]*shade)+18),min(255,int(color[1]*shade)+12),
                               min(255,int(color[2]*shade)+25))
                    zbuf2d[fr,fc]=z_val[upd]

        self._render_transp_into_fb(fb, zbuf.reshape(rh, rw), pv_s, rn, light)
        img = Image.fromarray(fb, "RGB")
        if sc < 1.0:
            img = img.resize((w, h), Image.NEAREST)
        self._rendered = ImageTk.PhotoImage(img)
        self.create_image(0, 0, anchor="nw", image=self._rendered)

    def _press(self, e):
        self.focus_set()
        self._drag_look = (e.x, e.y, self._cam_yaw, self._cam_pitch)
        self._dragging = True

    def _release(self, e):
        self._dragging = False
        if self._edit_mode and self._drag_look:
            dx = abs(e.x - self._drag_look[0]); dy = abs(e.y - self._drag_look[1])
            if dx < 5 and dy < 5:
                self._edit_click(e); return
        self.redraw()

    def _move(self, e):
        if not self._drag_look: return
        if self._transform_mode: return  # transform dragged by _tx_motion via <Motion>
        x0, y0, yaw0, pit0 = self._drag_look
        dx = e.x - x0; dy = e.y - y0
        # Only start rotating after 5px to avoid perturbing _pv_last during a click
        if abs(dx) < 5 and abs(dy) < 5: return
        self._cam_yaw   = yaw0 - dx * 0.3
        self._cam_pitch = max(-89.0, min(89.0, pit0 - dy * 0.3))
        self._schedule_redraw()

    def _wheel(self, e):
        d = getattr(e, "delta", 0)
        factor = 1.25 if (e.num == 4 or d > 0) else 1.0 / 1.25
        spd = self._cam_spd or max(getattr(self, '_last_span', 100.0) * 0.04, 1.0)
        self._cam_spd = max(0.1, spd * factor)

    def _key_press(self, e):
        key = e.keysym.lower()
        # During active transform: apply, cancel, or change axis constraint
        if self._transform_mode:
            if key == 'return':
                self._apply_transform(); return 'break'
            if key == 'escape':
                self._cancel_transform(); return 'break'
            if key in ('x', 'y', 'z'):
                new_ax = key.upper()
                self._tx_axis = None if self._tx_axis == new_ax else new_ax
                self._schedule_redraw()
                return 'break'
        # Don't intercept transform-shortcut keys while in edit mode
        if self._edit_mode and key in ('r', 's', 'm', 'c', 'p', 'h'):
            return
        spd = self._cam_spd or max(getattr(self, '_last_span', 100.0) * 0.04, 1.0)
        yr = math.radians(self._cam_yaw)
        fwd_x, fwd_z = -math.sin(yr), math.cos(yr)   # R[2]: (-sin, 0, cos) at pitch=0
        rgt_x, rgt_z =  math.cos(yr), math.sin(yr)   # R[0]: (cos,  0, sin)
        moved = True
        if   key in ('w', 'up'):
            self._cam_pos[0] += fwd_x * spd; self._cam_pos[2] += fwd_z * spd
        elif key in ('s', 'down'):
            self._cam_pos[0] -= fwd_x * spd; self._cam_pos[2] -= fwd_z * spd
        elif key in ('a', 'left'):
            self._cam_pos[0] -= rgt_x * spd; self._cam_pos[2] -= rgt_z * spd
        elif key in ('d', 'right'):
            self._cam_pos[0] += rgt_x * spd; self._cam_pos[2] += rgt_z * spd
        elif key in ('space', 'q'):
            self._cam_pos[1] += spd
        elif key in ('z', 'e', 'alt_l', 'alt_r'):
            self._cam_pos[1] -= spd
        else:
            moved = False
        if moved:
            self._schedule_redraw()
            return 'break'

    def _schedule_redraw(self):
        """Cap interactive redraws at ~60 fps to avoid queueing more renders than can run."""
        if self._redraw_id is not None:
            return
        self._redraw_id = self.after(16, self._do_redraw)

    def _do_redraw(self):
        self._redraw_id = None
        self.redraw()


# ---------------------------------------------------------------------------
# MaterialView — color swatches + sphere preview
# ---------------------------------------------------------------------------

SWATCH_LABELS = ["emission", "diffuse", "specular", "ambient"]


class MaterialView(ttk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self._tex_photo = None   # keep reference so PhotoImage isn't GC'd

        # Left: sphere canvas (shaded with material diffuse)
        left = ttk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.sphere = MeshCanvas(left, height=200)
        self.sphere.pack(fill=tk.BOTH, expand=True)
        ctrl = ttk.Frame(left)
        ctrl.pack(fill=tk.X)
        self._wire = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Wireframe", variable=self._wire,
                        command=self._toggle_wire).pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="⟳ Reset", command=self.sphere._home).pack(side=tk.LEFT)

        # Right: swatch panel
        right = ttk.LabelFrame(self, text="Colors", width=220)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=4, pady=4)
        right.pack_propagate(False)
        self.swatch_cv = tk.Canvas(right, background="#1e1e2e", width=210)
        self.swatch_cv.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

    def load(self, colors: dict, canv_rgba=None):
        """colors: dict from mat_all_colors(); canv_rgba: (w,h,rgba) or None."""
        self._tex_photo = None

        # Sphere: render with actual texture when available, else flat diffuse colour
        diffuse = colors.get("diffuse", (160, 160, 160))
        if canv_rgba and colors.get("mtype") == 2 and HAS_PIL and HAS_NUMPY:
            sw, sh, srgba = canv_rgba
            import numpy as _np
            tex_arr = _np.frombuffer(srgba, dtype=_np.uint8).reshape(sh, sw, 4).copy()
            tris = [((i0,i1,i2), 0, fn, uvs) for (i0,i1,i2), fn, uvs in _SPHERE_TRIS]
            self.sphere.load_mesh(_SPHERE_VERTS, tris, info="sphere preview", textures=[tex_arr])
            self.sphere.textured = True
        else:
            sphere_color = diffuse
            if canv_rgba and colors.get("mtype") == 2:
                sphere_color = _canvas_average_color(canv_rgba[2], canv_rgba[0], canv_rgba[1])
            tris = [((i0,i1,i2), sphere_color, fn) for (i0,i1,i2), fn, _ in _SPHERE_TRIS]
            self.sphere.load_mesh(_SPHERE_VERTS, tris, info="sphere preview")
            self.sphere.textured = False

        # Swatches
        cv = self.swatch_cv; cv.delete("all")
        mtype_names = {0: "Flat", 1: "Lit", 2: "Textured"}
        mtype = colors.get("mtype", 0)
        alpha = colors.get("alpha", 0.0)
        y = 10
        cv.create_text(8, y, anchor="nw",
                       text=f"Type: {mtype_names.get(mtype,'?')}   alpha: {alpha:.4f}",
                       fill="#aaa", font=("Helvetica", 10, "bold"))
        y += 24

        for key in SWATCH_LABELS:
            rgb = colors.get(key, None)
            if rgb is None: continue
            r, g, b = rgb
            hex_c = f"#{r:02x}{g:02x}{b:02x}"
            cv.create_rectangle(8, y, 32, y+20, fill=hex_c, outline="#555")
            cv.create_text(40, y+10, anchor="w",
                           text=f"{key:<10} ({r:3d},{g:3d},{b:3d})   {hex_c}",
                           fill="#ccc", font=("Courier", 10))
            y += 28

        tex = colors.get("texture")
        if tex:
            cv.create_text(8, y, anchor="nw",
                           text=f"Texture: {tex[0]}#{tex[1]}",
                           fill="#88aaff", font=("Courier", 10))
            y += 20

        gouraud = colors.get("gouraud", False)
        nofog   = colors.get("nofog",   False)
        cv.create_text(8, y, anchor="nw",
                       text=f"gouraud={gouraud}  nofog={nofog}",
                       fill="#888", font=("Courier", 9))
        y += 18

        # Texture image preview (requires Pillow)
        if canv_rgba:
            w, h, rgba = canv_rgba
            if HAS_PIL:
                from PIL import Image, ImageTk
                img = Image.frombytes("RGBA", (w, h), rgba)
                # Scale to fit the swatch panel width (~200 px)
                scale = min(190 / w, 190 / h, 1.0)
                nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
                img = img.resize((nw, nh), Image.NEAREST)
                self._tex_photo = ImageTk.PhotoImage(img)
                cv.create_text(8, y, anchor="nw",
                               text=f"Texture  {w}×{h}", fill="#88aaff",
                               font=("Courier", 9))
                y += 14
                cv.create_image(8, y, anchor="nw", image=self._tex_photo)
                # expand canvas height so the image is fully visible
                cv.configure(height=y + nh + 8)
            else:
                cv.create_text(8, y, anchor="nw",
                               text=f"Texture  {w}×{h}  (install Pillow to preview)",
                               fill="#666", font=("Courier", 9))

    def _toggle_wire(self):
        self.sphere.wireframe = self._wire.get()
        self.sphere.redraw()


# ---------------------------------------------------------------------------
# GUI helpers
# ---------------------------------------------------------------------------

PAD = dict(padx=6, pady=4)

# Name prefixes for shapes that are game-editor markers / debug-only
# (confirmed from level data: all are 3-vert + 1-poly triangles used as spawn/waypoint/sound/etc.)
# Event-block material names as checked by FUN_0043fc50 (c47_p77.c) in MECHANIX.exe.
# Any polygon whose 3DMa material name matches is "consumed" (flag +0x1c = 4):
# not rendered, not solid.
_EVENT_MAT_EXACT: frozenset = frozenset({
    # billboard sprites
    "treep", "bush1p", "bush2p",
    # pickup boxes
    "xboxp", "xboxgp", "pboxp",
    # emitters
    "emitfp", "emitfp1", "emitfp2",
    "emitw", "emitwp", "emitwp1", "emitwp2",
    # camera intro markers
    "camp", "camlookp",
    # race start / waypoints
    "vstartp", "vpointp", "cstart",
    # misc markers
    "flarep", "fogcolor", "fogcoloralt", "respawnp",
    # turn-sign templates
    "turnlp", "turnrp",
    # invisible collision volumes
    "invisoblock",
})
# Prefix rules from the same function:
#   "snd"     → positional sound loop
#   "kobject" → kobject%dp template objects (1-10)
#   "pobject" → pobject%dp pushable objects (1-4)
_EVENT_MAT_PREFIX: tuple = ("snd", "kobject", "pobject", "ccheck")
# %cspawnp and %cpointp for letters a..z  → spawn waypoints / path waypoints
_EVENT_MAT_SPAWN = frozenset(c + "spawnp" for c in "abcdefghijklmnopqrstuvwxyz")
_EVENT_MAT_POINT = frozenset(c + "pointp" for c in "abcdefghijklmnopqrstuvwxyz")

# Editor-only: not event blocks (game renders them), but invisible or purely
# technical in the assembled view — collision-only geometry and gate planes.
_EDITOR_INVIS_MATS: frozenset = frozenset({
    "invisoblock",
})
_EDITOR_INVIS_PREFIX: tuple = ()


def _smooth_normal(nis, normals, face_normal):
    """Return averaged+normalised vertex normal for a triangle fan slice, or face_normal."""
    if not normals or not nis:
        return face_normal
    nlen = len(normals)
    vecs = [normals[ni] for ni in nis if 0 <= ni < nlen]
    if not vecs:
        return face_normal
    nx = sum(v[0] for v in vecs) / len(vecs)
    ny = sum(v[1] for v in vecs) / len(vecs)
    nz = sum(v[2] for v in vecs) / len(vecs)
    L = max(0.001, (nx*nx + ny*ny + nz*nz) ** 0.5)
    return (nx/L, ny/L, nz/L)


def _parse_wave_info(data: bytes) -> str:
    """Return a human-readable summary of a RIFF/WAV payload."""
    try:
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return f"Wave  {len(data):,} bytes  (not RIFF/WAV — raw PCM?)"
        p = 12; channels = sample_rate = bits = n_samples = 0
        while p < len(data) - 8:
            cid = data[p:p+4]
            csz = struct.unpack_from("<I", data, p+4)[0]; p += 8
            if cid == b"fmt ":
                _, channels, sample_rate = struct.unpack_from("<HHI", data, p)
                bits = struct.unpack_from("<H", data, p+14)[0]
            elif cid == b"data":
                n_samples = csz // max(1, channels * max(1, bits // 8))
                break
            p += csz
        dur = n_samples / max(1, sample_rate)
        return (f"Wave  {len(data):,} bytes\n"
                f"  Channels   : {channels}\n"
                f"  Sample rate: {sample_rate:,} Hz\n"
                f"  Bit depth  : {bits}-bit\n"
                f"  Samples    : {n_samples:,}\n"
                f"  Duration   : {dur:.3f} s\n")
    except Exception as e:
        return f"Wave  {len(data):,} bytes\n  (parse error: {e})\n"


def _is_event_material(mat_name: str) -> bool:
    """Return True if this 3DMa name matches the game's event-block material list."""
    n = (mat_name or "").lower()
    return (n in _EVENT_MAT_EXACT
            or n.startswith(_EVENT_MAT_PREFIX)
            or n in _EVENT_MAT_SPAWN
            or n in _EVENT_MAT_POINT)


def _is_max_default_mat(name: str) -> bool:
    """Return True for unassigned 3ds-Max slot names like '16 - Default'."""
    import re as _re
    return bool(_re.fullmatch(r'\d+\s*-\s*default', name.lower()))


def _shape_all_event_mats(faces: list, mat_names: dict) -> bool:
    """Return True if every polygon uses an event-block or editor-invisible material."""
    if not faces:
        return False
    for _, _, _, _, mat in faces:
        if not mat or mat[0] != "3DMa":
            return False
        name = mat_names.get(mat[1], "").lower()
        if not (_is_event_material(name)
                or name in _EDITOR_INVIS_MATS
                or name.startswith(_EDITOR_INVIS_PREFIX)
                or _is_max_default_mat(name)):
            return False
    return True


def _build_mat_names(omt) -> dict:
    """Build {3DMa_id -> name} lookup from an OMTFile (or return {} if omt is None)."""
    if omt is None:
        return {}
    return {c.id: (c.name or "") for c in omt.chunks if c.type == "3DMa"}


def _is_debug_shape(name: str, n_verts: int, n_faces: int) -> bool:
    """Return True for pure marker shapes that carry no terrain geometry.
    Only 3-vertex/1-face marker triangles (spawn/waypoint/camera/etc.) and
    shapes whose name matches a known event-marker material name are filtered."""
    if n_verts == 3 and n_faces == 1:
        return True
    # shape names mirror their material names for all marker types
    nl = (name or "").lower()
    return _is_event_material(nl)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class OMF2Editor:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("OMF2 / OMT Editor")
        self.root.geometry("1280x780")
        self.omt:  Optional[OMTFile] = None
        self.edit: Dict[int, bytes]  = {}
        self._photo  = None
        self._mat_colors: Dict[int, Tuple] = {}   # cached {mat_id: (r,g,b)}
        self._wav_proc = None                      # subprocess playing the current Wave preview
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self._build_menu()
        self._build_toolbar()
        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        self._build_left(pane)
        self._build_right(pane)
        self.status_var = tk.StringVar(value="No file open")
        ttk.Label(self.root, textvariable=self.status_var,
                  relief=tk.SUNKEN, anchor="w").pack(fill=tk.X, side=tk.BOTTOM, padx=4, pady=2)
        self.root.bind("<Control-o>", lambda _: self.cmd_open())
        self.root.bind("<Control-s>", lambda _: self.cmd_save())
        self.root.bind("<Delete>",    lambda _: self.cmd_delete_chunk())
        self.root.bind("<F5>",        lambda _: self._load_level_assembled())
        self.root.bind("<F6>",        lambda _: self._preview_all_shapes())
        self.root.bind("<F7>",        lambda _: self.mesh_cv.toggle_anim())
        self.root.bind("<F8>",        lambda _: self.mesh_cv.toggle_edit_mode())
        self.root.bind("<r>",         lambda _: self.mesh_cv.begin_transform('R')
                                        if self.mesh_cv._edit_mode else None)
        self.root.bind("<s>",         lambda _: self.mesh_cv.begin_transform('S')
                                        if self.mesh_cv._edit_mode else None)
        self.root.bind("<m>",         lambda _: self.mesh_cv.begin_transform('M')
                                        if self.mesh_cv._edit_mode else None)
        self.root.bind("<c>",         lambda _: self._edit_copy())
        self.root.bind("<p>",         lambda _: self._edit_paste())
        self.root.bind("<h>",         lambda _: self._edit_toggle_hidden())
        self.root.bind("<Return>",    lambda _: self.mesh_cv._apply_transform()
                                        if self.mesh_cv._transform_mode else None)
        self.root.bind("<Escape>",    lambda _: self._edit_escape())
        self.root.bind("<Left>",      lambda _: self._cam_nudge(right=-1))
        self.root.bind("<Right>",     lambda _: self._cam_nudge(right=+1))
        self.root.bind("<Up>",        lambda _: self._cam_nudge(fwd=+1))
        self.root.bind("<Down>",      lambda _: self._cam_nudge(fwd=-1))
        self.root.bind("<space>",     lambda _: self._cam_nudge(y=+1)
                                        if self.mesh_cv._edit_mode else None)
        self.root.bind("<Alt_L>",     lambda _: self._cam_nudge(y=-1)
                                        if self.mesh_cv._edit_mode else None)
        self.root.bind("<Alt_R>",     lambda _: self._cam_nudge(y=-1)
                                        if self.mesh_cv._edit_mode else None)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self._stop_wav()
        self.root.destroy()

    def _build_menu(self):
        mb = tk.Menu(self.root)
        fm = tk.Menu(mb, tearoff=0)
        fm.add_command(label="New…",              command=self.cmd_new,            accelerator="Ctrl+N")
        fm.add_command(label="Open…",             command=self.cmd_open,           accelerator="Ctrl+O")
        fm.add_command(label="Save",              command=self.cmd_save,           accelerator="Ctrl+S")
        fm.add_command(label="Save As…",          command=self.cmd_save_as)
        fm.add_separator()
        fm.add_command(label="Set OMF version…",  command=self.cmd_set_version)
        fm.add_separator()
        fm.add_command(label="Exit",              command=self.root.quit)
        mb.add_cascade(label="File", menu=fm)
        cm = tk.Menu(mb, tearoff=0)
        cm.add_command(label="Add chunk from raw file…",    command=self.cmd_add_chunk)
        cm.add_command(label="Add texture from image…",     command=self.cmd_add_texture_from_image)
        cm.add_command(label="Add material…",               command=self.cmd_add_material)
        cm.add_command(label="Add shape from OBJ…",         command=self.cmd_add_shape_from_obj)
        cm.add_command(label="Add wave from WAV…",          command=self.cmd_add_wave_from_wav)
        cm.add_separator()
        cm.add_command(label="Delete selected chunk",       command=self.cmd_delete_chunk, accelerator="Del")
        cm.add_separator()
        cm.add_command(label="Export raw payload…",         command=self.cmd_export_chunk)
        cm.add_command(label="Export shape as OBJ…",        command=self.cmd_export_shape_as_obj)
        cm.add_command(label="Export level as OBJ…",        command=self.cmd_export_level_as_obj)
        cm.add_command(label="Preview level in browser",    command=self.cmd_preview_level)
        cm.add_command(label="Replace payload from file…",  command=self.cmd_replace_chunk)
        mb.add_cascade(label="Chunks", menu=cm)
        mb.add_command(label="About", command=self._about)
        self.root.config(menu=mb)
        self.root.bind("<Control-n>", lambda _: self.cmd_new())

    def _build_toolbar(self):
        tb = ttk.Frame(self.root)
        tb.pack(fill=tk.X, padx=4, pady=(4, 0))
        for label, cmd in [("Open",self.cmd_open),("Save",self.cmd_save),
                           ("Save As",self.cmd_save_as),("|",None),
                           ("Add Chunk",self.cmd_add_chunk),("Add Wave",self.cmd_add_wave_from_wav),
                           ("Delete",self.cmd_delete_chunk),
                           ("|",None),("Export Raw",self.cmd_export_chunk),
                           ("Replace",self.cmd_replace_chunk)]:
            if label == "|":
                ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)
            else:
                ttk.Button(tb, text=label, command=cmd).pack(side=tk.LEFT, padx=2, pady=2)

    def _build_left(self, pane):
        lf = ttk.LabelFrame(pane, text="Chunks")
        pane.add(lf, weight=1)
        fbar = ttk.Frame(lf)
        fbar.pack(fill=tk.X, padx=4, pady=(4, 0))
        ttk.Label(fbar, text="Filter:").pack(side=tk.LEFT)
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", lambda *_: self._refresh_tree())
        ttk.Entry(fbar, textvariable=self.filter_var, width=20).pack(side=tk.LEFT, padx=4)
        ttk.Button(fbar, text="✕", width=2,
                   command=lambda: self.filter_var.set("")).pack(side=tk.LEFT)
        cols = ("type", "id", "name", "size", "comp")
        self.tree = ttk.Treeview(lf, columns=cols, show="tree headings", selectmode="browse")
        self.tree.heading("#0", text=""); self.tree.column("#0", width=18, stretch=False)
        for col, hdr, w in [("type","Type",56),("id","ID",52),("name","Name",170),
                             ("size","Size",72),("comp","Zip",38)]:
            self.tree.heading(col, text=hdr); self.tree.column(col, width=w, anchor="w")
        vsb = ttk.Scrollbar(lf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4,0), pady=4)
        vsb.pack(side=tk.RIGHT, fill=tk.Y, pady=4, padx=(0,4))
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.tag_configure("edited", foreground="#e8a000")

    def _build_right(self, pane):
        rf = ttk.Frame(pane)
        pane.add(rf, weight=3)

        # Summary text: small read-only area always visible above the tabs
        sum_frame = ttk.Frame(rf)
        sum_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
        self.dec_txt = tk.Text(sum_frame, font=("Courier", 10), wrap=tk.WORD, state=tk.DISABLED,
                               height=4, background="#12121e", foreground="#c8c8d8")
        ysb = ttk.Scrollbar(sum_frame, orient="vertical", command=self.dec_txt.yview)
        self.dec_txt.configure(yscrollcommand=ysb.set)
        ysb.pack(side=tk.RIGHT, fill=tk.Y)
        self.dec_txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.nb = ttk.Notebook(rf)
        self.nb.pack(fill=tk.BOTH, expand=True)

        # Info tab
        ti = ttk.Frame(self.nb); self.nb.add(ti, text="  Info  ")
        self._build_info_tab(ti)

        # Hex tab
        th = ttk.Frame(self.nb); self.nb.add(th, text="  Hex  ")
        self._build_hex_tab(th)

        # 3D Mesh tab
        self._mesh_tab = ttk.Frame(self.nb)
        self.nb.add(self._mesh_tab, text="  3D Mesh  ")
        self._build_mesh_tab(self._mesh_tab)

        # Material tab
        self._mat_tab = ttk.Frame(self.nb)
        self.nb.add(self._mat_tab, text="  Material  ")
        self.mat_view = MaterialView(self._mat_tab)
        self.mat_view.pack(fill=tk.BOTH, expand=True)

        # Image tab
        self._img_tab = ttk.Frame(self.nb)
        self.nb.add(self._img_tab, text="  Image  ")
        self.dec_img = ttk.Label(self._img_tab)
        self.dec_img.pack(padx=8, pady=8, anchor="center")

        # Wave tab
        self._wave_tab = ttk.Frame(self.nb)
        self.nb.add(self._wave_tab, text="  Wave  ")
        self._build_wave_tab(self._wave_tab)

    def _build_info_tab(self, f):
        fg = ttk.LabelFrame(f, text="File"); fg.pack(fill=tk.X, **PAD)
        self._fv_path    = self._info_row(fg, 0, "Path")
        self._fv_version = self._info_row(fg, 1, "Version")
        self._fv_chunks  = self._info_row(fg, 2, "Total chunks")
        fg.columnconfigure(1, weight=1)
        cg = ttk.LabelFrame(f, text="Selected chunk"); cg.pack(fill=tk.X, **PAD)
        self._cv_type = self._info_row(cg, 0, "Type",   editable=True)
        self._cv_id   = self._info_row(cg, 1, "ID",     editable=True)
        self._cv_name = self._info_row(cg, 2, "Name",   editable=True)
        self._cv_size = self._info_row(cg, 3, "Size")
        self._cv_comp = self._info_row(cg, 4, "Compression")
        cg.columnconfigure(1, weight=1)
        ttk.Button(cg, text="Apply changes (type / ID / name)",
                   command=self._apply_info).grid(row=5, column=0, columnspan=2, pady=8)
        ttk.Label(cg, text="Type = 4 ASCII chars · ID = integer",
                  foreground="gray").grid(row=6, column=0, columnspan=2)

    def _info_row(self, parent, row, label, editable=False):
        ttk.Label(parent, text=label+":").grid(row=row, column=0, sticky="w", **PAD)
        var = tk.StringVar()
        ttk.Entry(parent, textvariable=var,
                  state="normal" if editable else "readonly",
                  width=48).grid(row=row, column=1, sticky="ew", **PAD)
        return var

    def _build_hex_tab(self, f):
        self.hex_txt = tk.Text(f, font=("Courier", 10), wrap=tk.NONE, state=tk.DISABLED,
                                background="#1a1a2e", foreground="#e0e0e0")
        xsb = ttk.Scrollbar(f, orient="horizontal", command=self.hex_txt.xview)
        ysb = ttk.Scrollbar(f, orient="vertical",   command=self.hex_txt.yview)
        self.hex_txt.configure(xscrollcommand=xsb.set, yscrollcommand=ysb.set)
        self.hex_txt.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        f.rowconfigure(0, weight=1); f.columnconfigure(0, weight=1)

    def _build_wave_tab(self, f):
        self._wave_btns = ttk.Frame(f)
        self._wave_btns.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(self._wave_btns, text="▶  Play",
                   command=lambda: self._play_wav(self._wave_data) if self._wave_data else None
                   ).pack(side=tk.LEFT, padx=4)
        ttk.Button(self._wave_btns, text="■  Stop",
                   command=self._stop_wav).pack(side=tk.LEFT, padx=4)
        self._wave_data: bytes = b""

    def _build_mesh_tab(self, f):
        tb = ttk.Frame(f); tb.pack(fill=tk.X, padx=4, pady=2)

        # Group 1: view controls
        self._wire_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tb, text="Wireframe", variable=self._wire_var,
                        command=self._toggle_wire).pack(side=tk.LEFT, padx=4)
        self._tex_var = tk.BooleanVar(value=True)
        tex_state = "normal" if (HAS_PIL and HAS_NUMPY) else "disabled"
        self._tex_cb = ttk.Checkbutton(tb, text="Textured", variable=self._tex_var,
                                       state=tex_state, command=self._toggle_textured)
        self._tex_cb.pack(side=tk.LEFT, padx=4)
        self._mat_color_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tb, text="Materials", variable=self._mat_color_var,
                        command=self._reload_mesh).pack(side=tk.LEFT, padx=4)
        ttk.Button(tb, text="⟳ Reset", width=7,
                   command=lambda: self.mesh_cv._home()).pack(side=tk.LEFT, padx=4)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        # Group 2: assembly
        self._hide_debug_var = tk.BooleanVar(value=True)
        self._last_assembly_cmd = None  # tracks which assembly view is live
        ttk.Checkbutton(tb, text="Hide markers", variable=self._hide_debug_var,
                        command=self._on_hide_markers_toggle
                        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(tb, text="Preview All",
                   command=self._preview_all_shapes).pack(side=tk.LEFT, padx=4)
        ttk.Button(tb, text="Level Assembly",
                   command=self._load_level_assembled).pack(side=tk.LEFT, padx=4)

        self.mesh_cv = MeshCanvas(f)
        self.mesh_cv.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Tree helpers
    # ------------------------------------------------------------------

    def _refresh_tree(self):
        flt = self.filter_var.get().lower()
        self.tree.delete(*self.tree.get_children())
        if not self.omt: return
        by_type: Dict[str, List] = {}
        for i, c in enumerate(self.omt.chunks):
            if flt and flt not in c.type.lower() and flt not in c.name.lower() \
                    and flt not in str(c.id):
                continue
            by_type.setdefault(c.type, []).append((i, c))
        for ctype in sorted(by_type):
            items = by_type[ctype]
            parent = self.tree.insert("", "end", text="▸",
                                       values=(ctype,"",f"({len(items)})","",""), open=True)
            for i, c in items:
                tags = ("edited",) if i in self.edit else ()
                self.tree.insert(parent, "end",
                                  values=(c.type, c.id, c.name or "(unnamed)",
                                          f"{c.size:,}", f"z{c.compression}" if c.compression else ""),
                                  tags=(*tags, f"idx:{i}"))

    def _selected_index(self) -> Optional[int]:
        sel = self.tree.selection()
        if not sel: return None
        for tag in self.tree.item(sel[0], "tags"):
            if tag.startswith("idx:"): return int(tag[4:])
        return None

    def _selected_chunk(self) -> Optional[Tuple[Chunk, int]]:
        idx = self._selected_index()
        if idx is None: return None
        return self.omt.chunks[idx], idx

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_select(self, _event):
        result = self._selected_chunk()
        if result is None: return
        chunk, idx = result
        self._show_chunk(chunk, idx)

    def _show_chunk(self, chunk: Chunk, idx: int):
        # Info tab
        self._fv_path.set(self.omt.path)
        self._fv_version.set(f"OMF{self.omt.version}  ({MAGIC_B[self.omt.version].decode()})")
        self._fv_chunks.set(str(len(self.omt.chunks)))
        self._cv_type.set(chunk.type)
        self._cv_id.set(str(chunk.id))
        self._cv_name.set(chunk.name)
        self._cv_size.set(f"{chunk.size:,} bytes  (payload: {len(chunk_payload(chunk)):,})")
        self._cv_comp.set(f"zlib level {chunk.compression}" if chunk.compression else "none")

        # Hex tab
        raw = self.edit.get(idx, chunk.raw)
        self.hex_txt.config(state=tk.NORMAL)
        self.hex_txt.delete("1.0", tk.END)
        self.hex_txt.insert("1.0", _hex_dump(raw))
        self.hex_txt.config(state=tk.DISABLED)

        # Decoded tab
        payload = self.edit.get(idx, chunk_payload(chunk))
        self._show_decoded(chunk, payload)

    def _stop_wav(self):
        """Kill any wave that is currently playing."""
        if self._wav_proc and self._wav_proc.poll() is None:
            try: self._wav_proc.kill()
            except Exception: pass
        self._wav_proc = None

    def _play_wav(self, data: bytes):
        """Pipe WAV bytes to aplay / ffplay in a background thread so the UI stays responsive."""
        import subprocess, threading
        self._stop_wav()
        proc = None
        for cmd in (["aplay", "--quiet", "-"],
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"]):
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                break
            except FileNotFoundError:
                continue
        if proc is None:
            messagebox.showinfo("Play", "No audio player found — install aplay (alsa-utils) or ffplay (ffmpeg)")
            return
        self._wav_proc = proc
        def _feed(p=proc, d=data):
            try:
                p.stdin.write(d)
                p.stdin.close()
            except Exception:
                pass
        threading.Thread(target=_feed, daemon=True).start()

    def _show_decoded(self, chunk: Chunk, payload: bytes):
        self._stop_wav()
        self.dec_txt.config(state=tk.NORMAL)
        self.dec_txt.delete("1.0", tk.END)
        self._photo = None
        self.dec_img.config(image="")
        self.mesh_cv.clear()

        try:
            if chunk.type == "Canv":
                w, h, depth, transp, rgba = decode_canv(payload)
                self.dec_txt.insert("1.0",
                    f"Canvas  {w} × {h} px  depth={depth}  transparent={transp}\n"
                    f"RGBA bytes: {len(rgba):,}")
                if HAS_PIL:
                    img = Image.frombytes("RGBA", (w, h), rgba)
                    scale = min(400/w, 400/h, 1.0)
                    nw, nh = max(1, int(w*scale)), max(1, int(h*scale))
                    img = img.resize((nw, nh), Image.NEAREST)
                    self._photo = ImageTk.PhotoImage(img)
                    self.dec_img.config(image=self._photo)
                else:
                    self.dec_txt.insert(tk.END, "\n\n(install Pillow for image preview)")
                self.nb.select(self._img_tab)

            elif chunk.type == "3DSh":
                self.dec_txt.insert("1.0", summarise_shape(payload))
                self._current_shape_payload = payload
                _sv, _sf, _ = decode_shape_full(payload)
                _mn = _build_mat_names(self.omt)
                if self._hide_debug_var.get() and (
                    _shape_all_event_mats(_sf, _mn)
                    or _is_debug_shape(chunk.name, len(_sv), len(_sf))
                ):
                    self.mesh_cv.clear()
                    self.dec_txt.insert(tk.END,
                        "\n\n[Marker shape — hidden by 'Hide markers' toggle]")
                else:
                    self._load_mesh_for_chunk(chunk, payload)
                    self.nb.select(self._mesh_tab)

            elif chunk.type == "Wave":
                self.dec_txt.insert("1.0", _parse_wave_info(payload))
                self._wave_data = payload
                self.nb.select(self._wave_tab)

            elif chunk.type == "3DMa":
                self.dec_txt.insert("1.0", summarise_material(payload))
                colors = mat_all_colors(payload)
                # Try to find linked canvas texture for preview
                canv_rgba = None
                tex = colors.get("texture")
                if tex and tex[0] == "Canv" and self.omt:
                    tc = next((c for c in self.omt.chunks if c.type == "Canv" and c.id == tex[1]), None)
                    if tc:
                        try:
                            _w,_h,_,_,rgba = decode_canv(chunk_payload(tc))
                            canv_rgba = (_w, _h, rgba)
                        except Exception:
                            pass
                self.mat_view.load(colors, canv_rgba)
                self.nb.select(self._mat_tab)

            else:
                printable = sum(32 <= b < 127 for b in payload[:256])
                if printable > 180:
                    self.dec_txt.insert("1.0",
                        f"Text preview ({len(payload):,} bytes):\n\n"
                        + payload[:1024].decode("latin-1", errors="replace"))
                else:
                    self.dec_txt.insert("1.0",
                        f"Type '{chunk.type}' — no decoder\n"
                        f"Payload: {len(payload):,} bytes\n\n"
                        + "".join(chr(b) if 32 <= b < 127 else "." for b in payload[:64]))

        except Exception as e:
            self.dec_txt.insert("1.0", f"Decode error:\n{e}")

        self.dec_txt.config(state=tk.DISABLED)

    def _load_mesh_for_chunk(self, chunk: Chunk, payload: bytes):
        """Decode 3DSh payload and populate the MeshCanvas."""
        verts, faces, normals = decode_shape_full(payload)
        if not verts:
            self.mesh_cv.clear(); return

        use_mats = self._mat_color_var.get()
        GREY = (160, 160, 160)

        # Build per-material data: flat colour fallback and optional texture array
        mat_colors:   Dict[int, Tuple[int,int,int]] = {}  # mat_id → (r,g,b) fallback
        mat_tex_idx:  Dict[int, int] = {}                  # mat_id → textures[] index
        textures: list = []                                # list of np.ndarray (h,w,4)

        if self.omt and use_mats:
            for c in self.omt.chunks:
                if c.type != "3DMa":
                    continue
                try:
                    pl = chunk_payload(c)
                    info_m = mat_all_colors(pl)
                    color: Tuple[int,int,int] = tuple(info_m["diffuse"])  # type: ignore
                    if info_m["mtype"] == 2:
                        tex_ref = info_m.get("texture")
                        if tex_ref and tex_ref[0] == "Canv":
                            tc = next((cc for cc in self.omt.chunks
                                       if cc.type == "Canv" and cc.id == tex_ref[1]), None)
                            if tc:
                                try:
                                    _w, _h, _, _, _rgba = decode_canv(chunk_payload(tc))
                                    avg = _canvas_average_color(_rgba, _w, _h)
                                    if max(avg) > 20:
                                        color = avg
                                    if HAS_NUMPY:
                                        import numpy as np
                                        arr = np.frombuffer(_rgba, dtype=np.uint8
                                                            ).reshape(_h, _w, 4).copy()
                                        mat_tex_idx[c.id] = len(textures)
                                        textures.append(arr)
                                except Exception:
                                    pass
                    mat_colors[c.id] = color
                except Exception:
                    mat_colors[c.id] = GREY

        tris = []
        for vis, uvs_list, nis_list, normal, mat in faces:
            tex_idx = None
            color = GREY
            if mat and mat[0] == "3DMa":
                mid = mat[1]
                if mid in mat_tex_idx:
                    tex_idx = mat_tex_idx[mid]
                elif mid in mat_colors:
                    r, g, b = mat_colors[mid]
                    color = (r, g, b) if (r | g | b) > 10 else (120, 120, 140)

            # fan triangulation
            for i in range(1, len(vis) - 1):
                i0, i1, i2 = vis[0], vis[i], vis[i + 1]
                if max(i0, i1, i2) >= len(verts):
                    continue
                sn = _smooth_normal([nis_list[0], nis_list[i], nis_list[i+1]], normals, normal) \
                     if nis_list else normal
                if tex_idx is not None:
                    uv0 = uvs_list[0]   if uvs_list else (0.0, 0.0)
                    uv1 = uvs_list[i]   if i   < len(uvs_list) else (0.0, 0.0)
                    uv2 = uvs_list[i+1] if i+1 < len(uvs_list) else (0.0, 0.0)
                    tris.append(((i0, i1, i2), tex_idx, sn, (uv0, uv1, uv2)))
                else:
                    tris.append(((i0, i1, i2), color, sn, None))

        info = f"{chunk.name}  "
        self.mesh_cv.load_mesh(verts, tris, info=info, textures=textures or None)
        self._wire_var.set(self.mesh_cv.wireframe)
        # auto-enable textured mode if textures were loaded
        if textures and HAS_PIL and HAS_NUMPY:
            self._tex_var.set(True)
            self.mesh_cv.textured = True

    def _on_dnd_drop(self, event):
        """Handle a file dropped onto the window."""
        try:
            files = event.widget.tk.splitlist(event.data)
            path = files[0] if files else ""
        except Exception:
            path = event.data.strip().strip("{}")
        if path and path.lower().endswith(".omt"):
            self.open_file(path)
        elif path:
            self._status(f"Drop ignored — not an .omt file: {os.path.basename(path)}")

    def _preview_all_shapes(self):
        """Combine every 3DSh chunk in the file into one assembled mesh view."""
        self._last_assembly_cmd = self._preview_all_shapes
        if not self.omt:
            self._status("No file open."); return
        shape_chunks = [(i, c) for i, c in enumerate(self.omt.chunks) if c.type == "3DSh"]
        if not shape_chunks:
            self._status("No 3DSh chunks in this file."); return

        GREY = (160, 160, 160)
        use_mats = self._mat_color_var.get()

        # Build per-material data from ALL 3DMa chunks in the file
        mat_colors:  Dict[int, Tuple[int,int,int]] = {}
        mat_tex_idx: Dict[int, int] = {}
        textures: list = []

        if use_mats:
            for c in self.omt.chunks:
                if c.type != "3DMa":
                    continue
                try:
                    pl = chunk_payload(c)
                    info_m = mat_all_colors(pl)
                    color: Tuple[int,int,int] = tuple(info_m["diffuse"])  # type: ignore
                    if info_m["mtype"] == 2:
                        tex_ref = info_m.get("texture")
                        if tex_ref and tex_ref[0] == "Canv":
                            tc = next((cc for cc in self.omt.chunks
                                       if cc.type == "Canv" and cc.id == tex_ref[1]), None)
                            if tc:
                                try:
                                    _w, _h, _, _, _rgba = decode_canv(chunk_payload(tc))
                                    avg = _canvas_average_color(_rgba, _w, _h)
                                    if max(avg) > 20:
                                        color = avg
                                    if HAS_NUMPY:
                                        arr = np.frombuffer(_rgba, dtype=np.uint8
                                                            ).reshape(_h, _w, 4).copy()
                                        mat_tex_idx[c.id] = len(textures)
                                        textures.append(arr)
                                except Exception:
                                    pass
                    mat_colors[c.id] = color
                except Exception:
                    mat_colors[c.id] = GREY

        all_verts:    list = []
        all_tris:     list = []
        shape_vranges: dict = {}
        self._shape_data: dict = {}
        MAX = self.mesh_cv.MAX_TRIS

        hide_debug = self._hide_debug_var.get()
        mat_names = _build_mat_names(self.omt) if hide_debug else {}

        for idx, chunk in shape_chunks:
            if len(all_tris) >= MAX:
                break
            payload = self.edit.get(idx, chunk_payload(chunk))
            try:
                verts, faces, normals = decode_shape_full(payload)
            except Exception:
                continue
            if hide_debug and (
                _shape_all_event_mats(faces, mat_names)
                or _is_debug_shape(chunk.name, len(verts), len(faces))
            ):
                continue
            shape_key = chunk.name or str(idx)
            base_v = len(all_verts)
            all_verts.extend(verts)
            shape_vranges[shape_key] = (base_v, base_v + len(verts))
            self._shape_data[shape_key] = {
                'chunk_idx': idx, 'verts': list(verts),
                'vrange': (base_v, base_v + len(verts)),
                'faces': faces, 'normals': normals,
            }

            for vis, uvs_list, nis_list, normal, mat in faces:
                if len(all_tris) >= MAX:
                    break
                if not vis or max(vis) >= len(verts):
                    continue
                tex_idx = None
                color = GREY
                if mat and mat[0] == "3DMa":
                    mid = mat[1]
                    if mid in mat_tex_idx:
                        tex_idx = mat_tex_idx[mid]
                    elif mid in mat_colors:
                        r, g, b = mat_colors[mid]
                        color = (r, g, b) if (r | g | b) > 10 else (120, 120, 140)
                for i in range(1, len(vis) - 1):
                    i0, i1, i2 = vis[0] + base_v, vis[i] + base_v, vis[i+1] + base_v
                    sn = _smooth_normal([nis_list[0], nis_list[i], nis_list[i+1]], normals, normal) \
                         if nis_list else normal
                    if tex_idx is not None:
                        uv0 = uvs_list[0]   if uvs_list else (0.0, 0.0)
                        uv1 = uvs_list[i]   if i   < len(uvs_list) else (0.0, 0.0)
                        uv2 = uvs_list[i+1] if i+1 < len(uvs_list) else (0.0, 0.0)
                        all_tris.append(((i0, i1, i2), tex_idx, sn, (uv0, uv1, uv2)))
                    else:
                        all_tris.append(((i0, i1, i2), color, sn, None))

        n_shapes = len(shape_chunks)
        info = f"All shapes ({n_shapes})  "
        self.mesh_cv.load_mesh(all_verts, all_tris, info=info,
                               textures=textures or None)
        if textures and HAS_PIL and HAS_NUMPY:
            self._tex_var.set(True)
            self.mesh_cv.textured = True
        self.nb.select(self._mesh_tab)
        capped = " (triangle cap reached)" if len(all_tris) >= MAX else ""
        self._status(f"Preview: {n_shapes} shapes, {len(all_verts):,} verts, "
                     f"{len(all_tris):,} tris{capped}")

    def _on_hide_markers_toggle(self):
        """Re-run whichever assembly view is currently displayed."""
        if self._last_assembly_cmd:
            self._last_assembly_cmd()

    def _load_level_assembled(self):
        """Show terrain as the game assembles it: skip 3d* templates and marker shapes."""
        self._last_assembly_cmd = self._load_level_assembled
        if not self.omt:
            self._status("No file open."); return

        GREY = (160, 160, 160)
        use_mats = self._mat_color_var.get()

        # Build per-material data from ALL 3DMa chunks in the file (same as _preview_all_shapes)
        mat_colors:  Dict[int, Tuple[int,int,int]] = {}
        mat_tex_idx: Dict[int, int] = {}
        mat_canv_id: Dict[int, int] = {}
        mat_alpha:   Dict[int, float] = {}   # 0=opaque  1=fully transparent
        textures: list = []
        _BG = np.array([18, 18, 30], dtype=np.float32) if HAS_NUMPY else None

        if use_mats:
            for c in self.omt.chunks:
                if c.type != "3DMa":
                    continue
                try:
                    pl = chunk_payload(c)
                    info_m = mat_all_colors(pl)
                    alpha_val: float = float(info_m.get("alpha", 0.0))
                    mat_alpha[c.id] = alpha_val
                    color: Tuple[int,int,int] = tuple(info_m["diffuse"])  # type: ignore
                    if info_m["mtype"] == 2:
                        tex_ref = info_m.get("texture")
                        if tex_ref and tex_ref[0] == "Canv":
                            tc = next((cc for cc in self.omt.chunks
                                       if cc.type == "Canv" and cc.id == tex_ref[1]), None)
                            if tc:
                                try:
                                    _w, _h, _, _, _rgba = decode_canv(chunk_payload(tc))
                                    avg = _canvas_average_color(_rgba, _w, _h)
                                    if max(avg) > 20:
                                        color = avg
                                    if HAS_NUMPY:
                                        arr = np.frombuffer(_rgba, dtype=np.uint8
                                                            ).reshape(_h, _w, 4).copy()
                                        if 0.0 < alpha_val < 0.95:
                                            rgb = arr[:, :, :3].astype(np.float32)
                                            arr[:, :, :3] = np.clip(
                                                rgb * (1.0 - alpha_val) + _BG * alpha_val,
                                                0, 255).astype(np.uint8)
                                        mat_tex_idx[c.id] = len(textures)
                                        mat_canv_id[c.id] = tc.id
                                        textures.append(arr)
                                except Exception:
                                    pass
                    mat_colors[c.id] = color
                except Exception:
                    mat_colors[c.id] = GREY

        # Build animated-texture map: find numbered-sibling Canv sequences
        anim_map: dict = {}
        if use_mats and HAS_NUMPY and mat_canv_id:
            import re as _re
            canv_by_id = {c.id: c for c in self.omt.chunks if c.type == "Canv"}
            by_stem: dict = {}
            for c in self.omt.chunks:
                if c.type != "Canv": continue
                m = _re.match(r'^(.*?)(\d+)$', c.name or "")
                if m:
                    by_stem.setdefault(m.group(1), []).append((int(m.group(2)), c))
            for mid, tidx in mat_tex_idx.items():
                if mat_alpha.get(mid, 0.0) <= 0.0: continue
                cid = mat_canv_id.get(mid)
                if cid is None: continue
                cc = canv_by_id.get(cid)
                if cc is None: continue
                m = _re.match(r'^(.*?)(\d+)$', cc.name or "")
                if not m: continue
                siblings = by_stem.get(m.group(1), [])
                if len(siblings) <= 1: continue
                frame_arrs = []
                for _, fc in sorted(siblings, key=lambda x: x[0]):
                    try:
                        fw, fh, _, _, frgba = decode_canv(chunk_payload(fc))
                        frame_arrs.append(np.frombuffer(frgba, dtype=np.uint8
                                          ).reshape(fh, fw, 4).copy())
                    except Exception:
                        pass
                if len(frame_arrs) > 1:
                    anim_map[tidx] = frame_arrs

        all_verts:     list = []
        all_tris:      list = []
        shape_vranges: dict = {}
        self._shape_data: dict = {}
        MAX = self.mesh_cv.MAX_TRIS
        n_terrain = n_skipped = 0

        hide_debug = self._hide_debug_var.get()
        _mat_names_lvl = _build_mat_names(self.omt) if hide_debug else {}

        for idx, chunk in enumerate(self.omt.chunks):
            if chunk.type != "3DSh":
                continue
            if len(all_tris) >= MAX:
                break
            payload = self.edit.get(idx, chunk_payload(chunk))
            try:
                verts, faces, normals = decode_shape_full(payload)
            except Exception:
                continue
            if chunk.name and chunk.name.lower().startswith("3d"):
                n_skipped += 1; continue
            if hide_debug and (
                _shape_all_event_mats(faces, _mat_names_lvl)
                or _is_debug_shape(chunk.name, len(verts), len(faces))
            ):
                n_skipped += 1; continue

            n_terrain += 1
            shape_key = f"{chunk.name}_{idx}" if chunk.name else str(idx)
            base_v = len(all_verts)
            all_verts.extend(verts)
            shape_vranges[shape_key] = (base_v, base_v + len(verts))
            self._shape_data[shape_key] = {
                'chunk_idx': idx, 'verts': list(verts),
                'vrange': (base_v, base_v + len(verts)),
                'faces': faces, 'normals': normals,
            }

            for vis, uvs_list, nis_list, normal, mat in faces:
                if len(all_tris) >= MAX:
                    break
                if not vis or max(vis) >= len(verts):
                    continue
                tex_idx = None
                color = GREY
                alpha = 0.0
                if mat and mat[0] == "3DMa":
                    mid = mat[1]
                    alpha = mat_alpha.get(mid, 0.0)
                    if alpha > 0.95:
                        continue  # fully transparent — skip face
                    if mid in mat_tex_idx:
                        tex_idx = mat_tex_idx[mid]
                    elif mid in mat_colors:
                        r, g, b = mat_colors[mid]
                        color = (r, g, b) if (r | g | b) > 10 else (120, 120, 140)
                for i in range(1, len(vis) - 1):
                    i0, i1, i2 = vis[0] + base_v, vis[i] + base_v, vis[i+1] + base_v
                    sn = _smooth_normal([nis_list[0], nis_list[i], nis_list[i+1]], normals, normal) \
                         if nis_list else normal
                    if tex_idx is not None:
                        uv0 = uvs_list[0]   if uvs_list else (0.0, 0.0)
                        uv1 = uvs_list[i]   if i   < len(uvs_list) else (0.0, 0.0)
                        uv2 = uvs_list[i+1] if i+1 < len(uvs_list) else (0.0, 0.0)
                        all_tris.append(((i0, i1, i2), tex_idx, sn, (uv0, uv1, uv2), alpha, shape_key))
                    else:
                        all_tris.append(((i0, i1, i2), color, sn, None, alpha, shape_key))

        if not all_verts:
            self._status("No terrain shapes found (markers/templates only?)."); return

        self.mesh_cv.load_mesh(all_verts, all_tris,
                               info=f"Level terrain ({n_terrain} shapes)  ",
                               textures=textures or None)
        self.mesh_cv.set_tex_anim(anim_map)
        self.mesh_cv.set_shape_vranges(shape_vranges)
        self.mesh_cv.on_transform_applied = self._on_transform_applied
        self.mesh_cv.on_selection_changed = self._on_selection_changed
        if textures and HAS_PIL and HAS_NUMPY:
            self._tex_var.set(True)
            self.mesh_cv.textured = True
        self.nb.select(self._mesh_tab)
        capped = " (triangle cap)" if len(all_tris) >= MAX else ""
        self._status(f"Level assembly: {n_terrain} terrain shapes, {n_skipped} markers/templates skipped, "
                     f"{len(all_verts):,} verts, {len(all_tris):,} tris{capped}")

    # --- editor mode callbacks ------------------------------------------------

    def _on_selection_changed(self, key):
        if key:
            self._status(f"Selected: {key}")
        else:
            self._status("Edit mode — click to select a mesh")

    def _on_transform_applied(self, shape_key, new_verts_global):
        """Bake the transform result back into the 3DSh chunk payload."""
        data = getattr(self, '_shape_data', {}).get(shape_key)
        if data is None:
            return
        vrange = self.mesh_cv._shape_vrange.get(shape_key)
        if vrange is None:
            return
        base_v, end_v = vrange
        local_verts = new_verts_global[base_v:end_v]
        try:
            payload = encode_3dsh(local_verts, data['faces'], data['normals'])
        except Exception as e:
            self._status(f"Re-encode failed: {e}"); return
        self.edit[data['chunk_idx']] = payload
        data['verts'] = list(local_verts)
        self._refresh_tree()
        self._status(f"Transform applied to {shape_key}")

    def _edit_toggle_hidden(self):
        key = self.mesh_cv._selected_key
        if not self.mesh_cv._edit_mode or not key:
            return
        self.mesh_cv.toggle_hidden(key)
        hidden = key in self.mesh_cv._hidden_keys
        self._status(f"{'Hidden' if hidden else 'Visible'}: {key}")

    def _edit_escape(self):
        cv = self.mesh_cv
        if cv._transform_mode:
            cv._cancel_transform()
        elif cv._selected_key:
            cv._selected_key = None
            if cv.on_selection_changed:
                cv.on_selection_changed(None)
            cv.redraw()
        elif cv._edit_mode:
            cv.toggle_edit_mode()

    def _edit_copy(self):
        cv = self.mesh_cv
        if not cv._edit_mode:
            self._status("Copy: press F8 to enter edit mode first"); return
        if not cv._selected_key:
            self._status("Copy: click a mesh to select it first"); return
        key = cv._selected_key
        sd = getattr(self, '_shape_data', {})
        if key not in sd:
            self._status(f"Copy: no shape data for {key!r}"); return
        info = sd[key]
        base_v, end_v = info['vrange']
        self._copy_clipboard = {
            'verts':   list(cv._verts[base_v:end_v]),
            'faces':   info['faces'],
            'normals': info['normals'],
            'source_key': key,
        }
        self._status(f"Copied: {key} ({end_v - base_v} verts)")

    def _edit_paste(self):
        if not self.omt:
            self._status("Paste: no file open"); return
        cb = getattr(self, '_copy_clipboard', None)
        if not cb:
            self._status("Nothing to paste — press C to copy a mesh first"); return
        new_verts = [(x + 50, y, z) for x, y, z in cb['verts']]
        try:
            payload = encode_3dsh(new_verts, cb['faces'], cb['normals'])
        except Exception as e:
            self._status(f"Paste encode failed: {e}"); return
        existing_ids = {c.id for c in self.omt.chunks if c.type == "3DSh"}
        new_id = max(existing_ids, default=0) + 1
        new_name = cb.get('source_key', 'shape') + "_copy"
        idx = len(self.omt.chunks)
        self.omt.chunks.append(Chunk("3DSh", new_id, 0, len(payload), new_name, 0, payload))
        self.edit[idx] = payload
        self._refresh_tree()
        self._load_level_assembled()
        self._status(f"Pasted {new_name} (+50 X, {len(new_verts)} verts) — save to keep")

    def _cam_nudge(self, fwd=0, right=0, y=0):
        import math as _m
        cv = self.mesh_cv
        spd = cv._cam_spd or max(getattr(cv, '_last_span', 100.0) * 0.04, 1.0)
        yr = _m.radians(cv._cam_yaw)
        fwd_x, fwd_z = -_m.sin(yr), _m.cos(yr)   # R[2] horizontal
        rgt_x, rgt_z =  _m.cos(yr), _m.sin(yr)   # R[0] horizontal
        cv._cam_pos[0] += (fwd_x * fwd + rgt_x * right) * spd
        cv._cam_pos[2] += (fwd_z * fwd + rgt_z * right) * spd
        cv._cam_pos[1] += y * spd
        cv.redraw()

    # --- toggle wire / texture ------------------------------------------------

    def _toggle_wire(self):
        self.mesh_cv.wireframe = self._wire_var.get()
        self.mesh_cv.redraw()

    def _toggle_textured(self):
        self.mesh_cv.textured = self._tex_var.get()
        self.mesh_cv.redraw()

    def _reload_mesh(self):
        """Re-build mesh with/without material colors after toggle."""
        result = self._selected_chunk()
        if result is None: return
        chunk, idx = result
        if chunk.type == "3DSh":
            payload = self.edit.get(idx, chunk_payload(chunk))
            self._load_mesh_for_chunk(chunk, payload)

    def _apply_info(self):
        result = self._selected_chunk()
        if result is None:
            messagebox.showinfo("No selection", "Select a chunk first."); return
        chunk, idx = result
        new_type = self._cv_type.get()
        if len(new_type) != 4:
            messagebox.showerror("Error", "Type must be exactly 4 characters."); return
        try:
            new_id = int(self._cv_id.get())
        except ValueError:
            messagebox.showerror("Error", "ID must be an integer."); return
        new_name = self._cv_name.get()
        chunk.type, chunk.id, chunk.name = new_type, new_id, new_name
        self._refresh_tree()
        self._status(f"Updated chunk {idx}: {new_type}/{new_id}  name={new_name!r}")

    # ------------------------------------------------------------------
    # File commands
    # ------------------------------------------------------------------

    def cmd_open(self):
        path = filedialog.askopenfilename(
            title="Open OMT file",
            filetypes=[("OMT files", "*.omt"), ("All files", "*.*")])
        if not path: return
        try:
            self.omt  = load_omt(path)
            self.edit = {}
            self._mat_colors = {}
            self._refresh_tree()
            self._status(f"Opened  {path}  —  {len(self.omt.chunks)} chunks  OMF{self.omt.version}")
            self.root.title(f"OMF2 Editor — {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Open failed", str(e))

    def cmd_save(self):
        if not self.omt:
            messagebox.showinfo("Nothing to save", "Open a file first."); return
        self._do_save(self.omt.path)

    def cmd_save_as(self):
        if not self.omt:
            messagebox.showinfo("Nothing to save", "Open a file first."); return
        path = filedialog.asksaveasfilename(
            title="Save As", initialfile=os.path.basename(self.omt.path),
            defaultextension=".omt",
            filetypes=[("OMT files", "*.omt"), ("All files", "*.*")])
        if path: self._do_save(path)

    def _build_omt_bytes(self) -> bytes:
        """Build the full OMT file bytes with self.edit overrides applied."""
        ver = self.omt.version
        content_pairs = [
            (c, self.edit.get(i, c.raw))
            for i, c in enumerate(self.omt.chunks)
            if c.type not in ("mCTy", "0HDR")
        ]
        mcty_payload = encode_mCTy(content_pairs)
        mcty_chunk   = Chunk("mCTy", 0xfffff752, 0, len(mcty_payload), "")
        all_pairs    = [(mcty_chunk, mcty_payload)] + content_pairs
        return _add_0hdr(build_omt(ver, all_pairs), ver), content_pairs

    def _do_save(self, path: str):
        try:
            data, content_pairs = self._build_omt_bytes()
            with open(path, "wb") as f: f.write(data)
            self.omt.path = path
            import struct as _s
            for cp, pl in content_pairs:
                if cp.type == "Wave" and len(pl) >= 36:
                    tag, ch, sr, _, _, bps = _s.unpack_from('<HHIIHH', pl, 20)
                    print(f"[save] Wave/{cp.id} name={cp.name!r} size={len(pl):,}B ch={ch} sr={sr} bps={bps}", flush=True)
                else:
                    print(f"[save] {cp.type}/{cp.id} name={cp.name!r} size={len(pl):,}B", flush=True)
            self._status(f"Saved  {path}  —  {len(data):,} bytes  OMF{self.omt.version}")
            self.root.title(f"OMF2 Editor — {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    # ------------------------------------------------------------------
    # Chunk commands
    # ------------------------------------------------------------------

    def cmd_add_chunk(self):
        if not self.omt:
            messagebox.showinfo("No file", "Open a file first."); return
        path = filedialog.askopenfilename(title="Add chunk — select raw payload",
                                          filetypes=[("All files", "*.*")])
        if not path: return
        ctype = simpledialog.askstring("Chunk type", "4-char type (Canv, 3DSh, 3DMa, …):",
                                       initialvalue="Canv")
        if not ctype or len(ctype) != 4:
            messagebox.showerror("Error", "Type must be exactly 4 characters."); return
        cid_s = simpledialog.askstring("Chunk ID", "Integer ID:", initialvalue="0")
        try: cid = int(cid_s)
        except (TypeError, ValueError):
            messagebox.showerror("Error", "ID must be an integer."); return
        cname = simpledialog.askstring("Chunk name", "Name:", initialvalue="") or ""
        raw = open(path, "rb").read()
        idx = len(self.omt.chunks)
        self.omt.chunks.append(Chunk(ctype, cid, 0, len(raw), cname, 0, raw))
        self.edit[idx] = raw
        self._refresh_tree()
        self._status(f"Added {ctype}/{cid} '{cname}'  {len(raw):,} bytes")

    def cmd_delete_chunk(self):
        result = self._selected_chunk()
        if result is None:
            messagebox.showinfo("No selection", "Select a chunk first."); return
        chunk, idx = result
        if not messagebox.askyesno("Delete chunk",
                                    f"Delete  {chunk.type}/{chunk.id}  '{chunk.name}'?"):
            return
        self.omt.chunks.pop(idx)
        self.edit = {(k if k < idx else k-1): v
                     for k, v in self.edit.items() if k != idx}
        self._refresh_tree()
        self._status(f"Deleted chunk at index {idx}")

    def cmd_export_chunk(self):
        result = self._selected_chunk()
        if result is None:
            messagebox.showinfo("No selection", "Select a chunk first."); return
        chunk, idx = result
        path = filedialog.asksaveasfilename(
            title="Export", initialfile=f"{chunk.type}_{chunk.id}.bin",
            filetypes=[("Binary","*.bin"),("All files","*.*")])
        if not path: return
        raw = self.edit.get(idx, chunk.raw)
        if chunk.compression and messagebox.askyesno(
                "Export", "Export decompressed? (No = compressed as stored)"):
            raw = chunk_payload(chunk)
        with open(path, "wb") as f: f.write(raw)
        self._status(f"Exported  {len(raw):,} bytes  →  {path}")

    def cmd_export_shape_as_obj(self):
        result = self._selected_chunk()
        if result is None:
            messagebox.showinfo("No selection", "Select a chunk first."); return
        chunk, idx = result
        if chunk.type != "3DSh":
            messagebox.showinfo("Wrong type", "Select a 3DSh chunk to export as OBJ."); return
        payload = self.edit.get(idx, chunk_payload(chunk))
        verts, faces, normals = decode_shape_full(payload)
        if not verts:
            messagebox.showerror("Export failed", "Could not decode shape data."); return
        base_name = chunk.name or f"3DSh_{chunk.id}"
        path = filedialog.asksaveasfilename(
            title="Export shape as OBJ",
            initialfile=f"{base_name}.obj",
            filetypes=[("Wavefront OBJ", "*.obj"), ("All files", "*.*")])
        if not path: return
        mtl_path = os.path.splitext(path)[0] + ".mtl"
        mat_chunks  = {c.id: c for c in self.omt.chunks if c.type == "3DMa"}
        canv_chunks = {c.id: c for c in self.omt.chunks if c.type == "Canv"}
        try:
            write_obj(verts, faces, normals, mat_chunks, canv_chunks, path, mtl_path)
            n_tex = sum(1 for _, _, _, _, mat in faces
                        if mat and mat[0] == "3DMa"
                        and mat_all_colors(
                            (mat_chunks[mat[1]].raw if mat[1] in mat_chunks else b"")
                        ).get("texture"))
            self._status(
                f"Exported  {len(verts):,} verts  {len(faces):,} faces"
                f"  →  {os.path.basename(path)}"
                + (f"  ({n_tex} texture(s) saved)" if n_tex and HAS_PIL else ""))
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def cmd_export_level_as_obj(self):
        if not self.omt:
            messagebox.showinfo("No file", "Open a levelN.omt file first."); return
        import re
        fname = os.path.basename(self.omt.path)
        m = re.match(r"level(\d+)\.omt$", fname, re.IGNORECASE)
        if not m:
            messagebox.showinfo(
                "Not a level file",
                f"'{fname}' doesn't look like a level file.\n"
                "Open level1.omt … level7.omt to use this export."); return
        level_index = int(m.group(1))
        omt_dir = os.path.dirname(os.path.abspath(self.omt.path))

        include_props = messagebox.askyesno(
            "Export level as OBJ",
            "Include placed objects (crates, signs, moving platforms)?\n\n"
            "No = terrain geometry only.")

        out_dir = filedialog.askdirectory(title="Choose output folder")
        if not out_dir: return

        try:
            from tools.level_exporter import export_level
        except ImportError:
            try:
                import importlib.util, sys as _sys
                spec = importlib.util.spec_from_file_location(
                    "level_exporter",
                    os.path.join(_TOOLS_DIR, "level_exporter.py"))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                export_level = mod.export_level
            except Exception as e:
                messagebox.showerror("Import error", f"Could not load level_exporter:\n{e}"); return

        try:
            import tempfile as _tf
            tmp_dir = _tf.mkdtemp(prefix="mech_export_")
            with open(os.path.join(tmp_dir, os.path.basename(self.omt.path)), "wb") as _f:
                _f.write(self._build_omt_bytes()[0])
            obj_path, mtl_path = export_level(
                level_index, out_dir=out_dir,
                include_props=include_props, omt_dir=tmp_dir)
            self._status(f"Level {level_index} exported  →  {os.path.basename(obj_path)}")
            messagebox.showinfo("Export complete",
                f"Level {level_index} written to:\n{obj_path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def cmd_preview_level(self):
        if not self.omt:
            messagebox.showinfo("No file", "Open a levelN.omt file first."); return
        import re, tempfile, webbrowser, base64, json as _json
        fname = os.path.basename(self.omt.path)
        m = re.match(r"level(\d+)\.omt$", fname, re.IGNORECASE)
        if not m:
            messagebox.showinfo(
                "Not a level file",
                f"'{fname}' doesn't look like a level file.\n"
                "Open level1.omt … level7.omt to use this preview."); return
        level_index = int(m.group(1))
        omt_dir = os.path.dirname(os.path.abspath(self.omt.path))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "level_exporter", os.path.join(_TOOLS_DIR, "level_exporter.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            export_level = mod.export_level
        except Exception as e:
            messagebox.showerror("Import error", f"Could not load level_exporter:\n{e}"); return

        self._status(f"Exporting level {level_index} for preview…")
        self.root.update()

        try:
            tmp_dir = tempfile.mkdtemp(prefix=f"mech_lvl{level_index}_")
            edited_omt_dir = tempfile.mkdtemp(prefix="mech_omt_")
            with open(os.path.join(edited_omt_dir, os.path.basename(self.omt.path)), "wb") as _f:
                _f.write(self._build_omt_bytes()[0])
            obj_path, mtl_path = export_level(
                level_index, out_dir=tmp_dir, include_props=True, omt_dir=edited_omt_dir)
        except Exception as e:
            messagebox.showerror("Export failed", str(e)); return

        # Read OBJ + MTL, encode all referenced textures as data URIs
        with open(obj_path) as f:
            obj_text = f.read()
        with open(mtl_path) as f:
            mtl_text = f.read()

        textures = {}
        for line in mtl_text.splitlines():
            if line.startswith("map_Kd"):
                tname = line.split()[1]
                tpath = os.path.join(tmp_dir, tname)
                if os.path.exists(tpath) and tname not in textures:
                    data = base64.b64encode(open(tpath,"rb").read()).decode()
                    textures[tname] = f"data:image/png;base64,{data}"

        NAMES = {1:"Urban Wasteland",2:"Tiki Jungle",3:"Pluto Mines",
                 4:"Construction Site",5:"Beach City",6:"Moon Base",7:"Asteroid Belt"}
        level_name = NAMES.get(level_index, f"Level {level_index}")

        tex_js  = _json.dumps(textures)
        obj_js  = _json.dumps(obj_text)
        mtl_js  = _json.dumps(mtl_text)

        html = """<!doctype html><html><head><meta charset=utf-8>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HW Mechanix — %%LEVELNAME%%</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d0e13;color:#e8eaf0;font:13px system-ui,sans-serif;overflow:hidden}
#c{display:block;width:100vw;height:100vh;cursor:grab;touch-action:none}
#c:active{cursor:grabbing}
#hud{position:fixed;top:10px;left:12px;display:flex;flex-direction:column;gap:5px;pointer-events:none}
.chip{background:rgba(22,24,32,.85);padding:4px 10px;border-radius:6px;border:1px solid #2a2d3a;font-size:12px}
#badge{color:#f7c430;font-weight:700}
</style></head><body>
<canvas id="c"></canvas>
<div id="hud"><div class="chip" id="badge">HW Mechanix — %%LEVELNAME%%</div>
<div class="chip" id="info">Loading…</div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/0.128.0/three.min.js"></script>
<script>
const TEX_MAP=%%TEX%%;
const OBJ_SRC=%%OBJ%%;
const MTL_SRC=%%MTL%%;
const renderer=new THREE.WebGLRenderer({canvas:document.getElementById('c'),antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setSize(innerWidth,innerHeight);
renderer.outputEncoding=THREE.sRGBEncoding;
const scene=new THREE.Scene();
scene.background=new THREE.Color(0x0d0e13);
scene.add(new THREE.AmbientLight(0xffffff,.55));
const sun=new THREE.DirectionalLight(0xffffff,.9);
sun.position.set(.4,1,.6);scene.add(sun);
const camera=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,1,200000);
window.addEventListener('resize',()=>{renderer.setSize(innerWidth,innerHeight);camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();});
const texCache={};
function getTex(n){if(!n)return null;if(texCache[n])return texCache[n];const u=TEX_MAP[n];if(!u)return null;const t=new THREE.TextureLoader().load(u);t.encoding=THREE.sRGBEncoding;texCache[n]=t;return t;}
function parseMTL(s){const m={};let c=null;for(const r of s.split('\\n')){const l=r.trim();if(!l||l[0]==='#')continue;const[k,...p]=l.split(/\\s+/);if(k==='newmtl'){c=p.join(' ');m[c]={};}else if(!c)continue;else if(k==='Kd')m[c].Kd=p.map(Number);else if(k==='d')m[c].d=+p[0];else if(k==='map_Kd')m[c].map=p[0];}return m;}
function parseOBJ(s,md){const root=new THREE.Group(),vs=[],vts=[],vns=[];let cg='root',cm='default',pd=[];
function flush(){if(!pd.length)return;const pos=[],uv=[],nr=[];for(const p of pd){pos.push(...p.v);uv.push(...p.vt);nr.push(...p.vn);}
const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));g.setAttribute('uv',new THREE.Float32BufferAttribute(uv,2));
if(nr.some(n=>n!==0))g.setAttribute('normal',new THREE.Float32BufferAttribute(nr,3));else g.computeVertexNormals();
const d=md[cm]||{};const t=d.map?getTex(d.map):null;const op=(d.d!==undefined)?d.d:1;const col=d.Kd?new THREE.Color(...d.Kd):new THREE.Color(.7,.7,.7);
root.add(new THREE.Mesh(g,new THREE.MeshPhongMaterial({map:t||null,color:t?0xffffff:col,side:THREE.DoubleSide,transparent:op<.9999,opacity:op,alphaTest:t?.1:0})));pd=[];}
for(const r of s.split('\\n')){const l=r.trim();if(!l||l[0]==='#')continue;const p=l.split(/\\s+/),k=p[0];
if(k==='v')vs.push([+p[1],+p[2],+p[3]]);else if(k==='vt')vts.push([+p[1],1-+p[2]]);else if(k==='vn')vns.push([+p[1],+p[2],+p[3]]);
else if(k==='g'){flush();cg=p.slice(1).join(' ')||'root';}else if(k==='usemtl'){flush();cm=p.slice(1).join(' ')||'default';}
else if(k==='f'){const pts=p.slice(1).map(tok=>{const[a,b,c]=tok.split('/').map(x=>x?+x:0);return{v:vs[a>0?a-1:vs.length+a]||[0,0,0],vt:b?vts[b>0?b-1:vts.length+b]||[0,0]:[0,0],vn:c?vns[c>0?c-1:vns.length+c]||[0,0,1]:[0,0,1]}});for(let i=1;i<pts.length-1;i++)for(const q of[pts[0],pts[i],pts[i+1]])pd.push(q);}}flush();return root;}
let theta=0,phi=Math.PI/4,radius=10000,ctr=new THREE.Vector3();
function uc(){camera.position.set(ctr.x+radius*Math.sin(phi)*Math.sin(theta),ctr.y+radius*Math.cos(phi),ctr.z+radius*Math.sin(phi)*Math.cos(theta));camera.lookAt(ctr);}
let drag=false,last={x:0,y:0};
document.getElementById('c').addEventListener('mousedown',e=>{drag=true;last={x:e.clientX,y:e.clientY};});
window.addEventListener('mouseup',()=>drag=false);
window.addEventListener('mousemove',e=>{if(!drag)return;theta-=(e.clientX-last.x)*.005;phi=Math.max(.05,Math.min(Math.PI-.05,phi+(e.clientY-last.y)*.005));last={x:e.clientX,y:e.clientY};uc();});
document.getElementById('c').addEventListener('wheel',e=>{radius=Math.max(radius*.05,radius+e.deltaY*radius*.001);uc();e.preventDefault();},{passive:false});
document.getElementById('info').textContent='Parsing…';
setTimeout(()=>{
const root=parseOBJ(OBJ_SRC,parseMTL(MTL_SRC));scene.add(root);
const box=new THREE.Box3().setFromObject(root),sz=box.getSize(new THREE.Vector3()),mx=Math.max(sz.x,sz.y,sz.z);
ctr=box.getCenter(new THREE.Vector3());radius=mx*.75;phi=Math.PI/4;theta=0;
camera.near=mx*.0005;camera.far=mx*12;camera.updateProjectionMatrix();uc();
document.getElementById('info').textContent=root.children.length+' meshes · drag orbit · scroll zoom';
renderer.setAnimationLoop(()=>renderer.render(scene,camera));
},50);
</script></body></html>"""
        html = (html
                .replace("%%LEVELNAME%%", level_name)
                .replace("%%TEX%%", tex_js)
                .replace("%%OBJ%%", obj_js)
                .replace("%%MTL%%", mtl_js))

        html_path = os.path.join(tmp_dir, f"level{level_index}_preview.html")
        with open(html_path, "w") as f:
            f.write(html)

        webbrowser.open(f"file://{html_path}")
        self._status(f"Level {level_index} preview opened in browser")

    def cmd_replace_chunk(self):
        result = self._selected_chunk()
        if result is None:
            messagebox.showinfo("No selection", "Select a chunk first."); return
        chunk, idx = result
        path = filedialog.askopenfilename(title="Replace payload",
                                          filetypes=[("All files","*.*")])
        if not path: return
        raw = open(path, "rb").read()
        note = ""
        if chunk.type == "Wave" and path.lower().endswith(".wav"):
            try:
                before = len(raw)
                raw = sanitize_wav_for_game(raw)
                if before != len(raw):
                    note = f"  (sanitized {before:,} → {len(raw):,} bytes)"
            except Exception as e:
                if not messagebox.askyesno("Sanitize failed",
                        f"WAV sanitize failed: {e}\n\nReplace with raw bytes anyway?"):
                    return
        self.edit[idx] = raw; chunk.size = len(raw); chunk.raw = raw
        self._show_chunk(chunk, idx)
        self._refresh_tree()
        self._status(f"Replaced {chunk.type}/{chunk.id}  {len(raw):,} bytes from {os.path.basename(path)}{note}")

    # ------------------------------------------------------------------
    # Builder commands  (New / Add Texture / Add Material / Add Shape)
    # ------------------------------------------------------------------

    def cmd_set_version(self):
        if not self.omt:
            messagebox.showinfo("No file", "Open a file first."); return
        ver_str = simpledialog.askstring(
            "Set OMF version",
            "OMF version (1, 2 or 3):\n"
            "  1 = 0MFS  (no chunk names)\n"
            "  2 = 0MF2  (names, Hot Wheels Mechanix)\n"
            "  3 = 0MF3  (names + zlib compression)",
            initialvalue=str(self.omt.version))
        if not ver_str: return
        try: ver = int(ver_str)
        except (TypeError, ValueError): return
        if ver not in (1, 2, 3):
            messagebox.showerror("Error", "Version must be 1, 2 or 3."); return
        self.omt.version = ver
        self._status(f"Version set to OMF{ver} — will take effect on next save")

    def cmd_new(self):
        ver_str = simpledialog.askstring("New OMT",
                                          "OMF version (1, 2 or 3):\n"
                                          "(Hot Wheels Mechanix uses version 2)",
                                          initialvalue="2")
        if not ver_str: return
        try: ver = int(ver_str)
        except (TypeError, ValueError): return
        if ver not in (1, 2, 3):
            messagebox.showerror("Error", "Version must be 1, 2 or 3."); return
        self.omt  = OMTFile("untitled.omt", ver, [])
        self.edit = {}
        self._mat_colors = {}
        self._refresh_tree()
        self._status(f"New OMT  version {ver}  —  no chunks")
        self.root.title("OMF2 Editor — untitled.omt")

    def cmd_add_texture_from_image(self):
        if not self.omt:
            messagebox.showinfo("No file", "Create or open an OMT file first."); return
        if not HAS_PIL:
            messagebox.showerror("Missing library",
                "Pillow (PIL) is required for image import.\n"
                "Install with:  pip install Pillow"); return
        path = filedialog.askopenfilename(
            title="Import texture",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tga *.tiff"),
                       ("All files", "*.*")])
        if not path: return
        try:
            img  = _Img.open(path).convert("RGBA")
            w, h = img.size
            name = simpledialog.askstring(
                "Chunk name", "Name:",
                initialvalue=os.path.splitext(os.path.basename(path))[0]) or ""
            cid  = max((c.id for c in self.omt.chunks if c.type == "Canv"), default=-1) + 1
            cid_s = simpledialog.askstring("Chunk ID", "ID:", initialvalue=str(cid))
            try: cid = int(cid_s)
            except (TypeError, ValueError): return
            payload = encode_canv(img.tobytes(), w, h)
            idx = len(self.omt.chunks)
            self.omt.chunks.append(Chunk("Canv", cid, 0, len(payload), name, 0, payload))
            self.edit[idx] = payload
            self._refresh_tree()
            self._status(f"Added Canv/{cid} '{name}'  {w}×{h}  {len(payload):,} bytes")
        except Exception as e:
            messagebox.showerror("Import failed", str(e))

    def cmd_add_material(self):
        if not self.omt:
            messagebox.showinfo("No file", "Create or open an OMT file first."); return

        dlg = tk.Toplevel(self.root)
        dlg.title("Add Material")
        dlg.resizable(False, False)
        dlg.grab_set()
        result: list = [None]

        canv_ids = [(c.id, c.name) for c in self.omt.chunks if c.type == "Canv"]
        P = {"padx": 8, "pady": 4}

        ttk.Label(dlg, text="Name:").grid(row=0, column=0, sticky=tk.W, **P)
        name_var = tk.StringVar(value="new_material")
        ttk.Entry(dlg, textvariable=name_var, width=24).grid(row=0, column=1, columnspan=2, **P)

        ttk.Label(dlg, text="Diffuse (#RRGGBB):").grid(row=1, column=0, sticky=tk.W, **P)
        diff_var = tk.StringVar(value="#c8c8c8")
        ttk.Entry(dlg, textvariable=diff_var, width=12).grid(row=1, column=1, sticky=tk.W, **P)

        use_tex = tk.BooleanVar(value=bool(canv_ids))
        ttk.Checkbutton(dlg, text="Textured", variable=use_tex).grid(
            row=2, column=0, sticky=tk.W, **P)

        tex_cb = None; tex_id_var = tk.StringVar(value="0")
        if canv_ids:
            choices = [f"{cid} — {cname}" for cid, cname in canv_ids]
            tex_cb = ttk.Combobox(dlg, values=choices, width=22, state="readonly")
            tex_cb.set(choices[0])
            tex_cb.grid(row=2, column=1, columnspan=2, **P)
        else:
            ttk.Entry(dlg, textvariable=tex_id_var, width=8).grid(row=2, column=1, **P)

        def on_ok():
            try:
                name = name_var.get().strip()
                hx = diff_var.get().lstrip("#")
                diff = (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))
                tid = None
                if use_tex.get():
                    if tex_cb is not None:
                        tid = canv_ids[tex_cb.current()][0]
                    else:
                        tid = int(tex_id_var.get())
                payload = encode_3dma(diffuse=diff, texture_id=tid)
                result[0] = (name, payload)
                dlg.destroy()
            except Exception as e:
                messagebox.showerror("Error", str(e), parent=dlg)

        bf = ttk.Frame(dlg)
        bf.grid(row=3, column=0, columnspan=3, pady=8)
        ttk.Button(bf, text="Add",    command=on_ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(bf, text="Cancel", command=dlg.destroy).pack(side=tk.LEFT, padx=4)

        dlg.wait_window()
        if result[0] is None: return

        name, payload = result[0]
        cid = max((c.id for c in self.omt.chunks if c.type == "3DMa"), default=-1) + 1
        idx = len(self.omt.chunks)
        self.omt.chunks.append(Chunk("3DMa", cid, 0, len(payload), name, 0, payload))
        self.edit[idx] = payload
        self._refresh_tree()
        self._status(f"Added 3DMa/{cid} '{name}'")

    def cmd_add_shape_from_obj(self):
        if not self.omt:
            messagebox.showinfo("No file", "Create or open an OMT file first."); return
        path = filedialog.askopenfilename(
            title="Import shape from OBJ",
            filetypes=[("Wavefront OBJ", "*.obj"), ("All files", "*.*")])
        if not path: return
        try:
            name = simpledialog.askstring(
                "Chunk name", "Name:",
                initialvalue=os.path.splitext(os.path.basename(path))[0]) or ""
            cid  = max((c.id for c in self.omt.chunks if c.type == "3DSh"), default=-1) + 1
            cid_s = simpledialog.askstring("Chunk ID", "ID:", initialvalue=str(cid))
            try: cid = int(cid_s)
            except (TypeError, ValueError): return
            mat_lookup = {c.name.lower(): c.id for c in self.omt.chunks if c.type == "3DMa"}
            with open(path, "r", errors="replace") as f:
                text = f.read()
            verts, faces, normals = _parse_obj(text, mat_lookup)
            if not verts:
                messagebox.showerror("Import failed", "No vertices found in OBJ file."); return
            payload = encode_3dsh(verts, faces, normals)
            idx = len(self.omt.chunks)
            self.omt.chunks.append(Chunk("3DSh", cid, 0, len(payload), name, 0, payload))
            self.edit[idx] = payload
            self._refresh_tree()
            self._status(
                f"Added 3DSh/{cid} '{name}'  {len(verts)} verts  {len(faces)} faces")
        except Exception as e:
            messagebox.showerror("Import failed", str(e))

    def cmd_add_wave_from_wav(self):
        if not self.omt:
            messagebox.showinfo("No file", "Create or open an OMT file first."); return
        path = filedialog.askopenfilename(
            title="Import WAV as Wave chunk",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if not path: return
        try:
            raw = open(path, "rb").read()
            name = simpledialog.askstring(
                "Chunk name", "Name:",
                initialvalue=os.path.splitext(os.path.basename(path))[0]) or ""

            existing_waves = [(i, c) for i, c in enumerate(self.omt.chunks) if c.type == "Wave"]
            if existing_waves:
                total_new = sum(c.size for _, c in existing_waves) + len(raw)
                replace = messagebox.askyesno(
                    "Wave chunk already exists",
                    f"This OMT already has {len(existing_waves)} Wave chunk(s).\n\n"
                    f"Replace Wave/{existing_waves[0][1].id} with the new WAV?\n"
                    f"(Yes = replace, No = add as extra chunk)\n\n"
                    f"NOTE: Hot Wheels Mechanix loads all Wave chunks from an OMT.\n"
                    f"Adding multiple large tracks ({total_new//1024//1024}+ MB total) will crash the game.")
                if replace:
                    # Replace the first Wave chunk in-place
                    rep_idx, rep_chunk = existing_waves[0]
                    before = len(raw)
                    payload = sanitize_wav_for_game(raw)
                    rep_chunk.name = name
                    rep_chunk.size = len(payload)
                    rep_chunk.raw  = payload
                    self.edit[rep_idx] = payload
                    self._refresh_tree()
                    note = f"  (sanitized {before:,} → {len(payload):,} bytes)" if before != len(payload) else ""
                    self._status(f"Replaced Wave/{rep_chunk.id} '{name}'  {len(payload):,} bytes{note}")
                    return

            cid  = max((c.id for c in self.omt.chunks if c.type == "Wave"), default=-1) + 1
            cid_s = simpledialog.askstring("Chunk ID", "ID:", initialvalue=str(cid))
            try: cid = int(cid_s)
            except (TypeError, ValueError): return
            before = len(raw)
            payload = sanitize_wav_for_game(raw)
            idx = len(self.omt.chunks)
            self.omt.chunks.append(Chunk("Wave", cid, 0, len(payload), name, 0, payload))
            self.edit[idx] = payload
            self._refresh_tree()
            note = f"  (sanitized {before:,} → {len(payload):,} bytes)" if before != len(payload) else ""
            self._status(f"Added Wave/{cid} '{name}'  {len(payload):,} bytes{note}")
        except Exception as e:
            messagebox.showerror("Import failed", str(e))

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _status(self, msg: str): self.status_var.set(msg)

    def _about(self):
        messagebox.showinfo("OMF2 Editor",
            "OMF2 / OMT file editor\n\n"
            "Reads, edits and writes OMedia formatted stream (.omt) files.\n"
            "Formats: 0MFS v1 · 0MF2 v2 · 0MF3 v3 (zlib chunks)\n\n"
            "Chunk viewers:\n"
            "  Canv → image preview (Pillow required)\n"
            "  3DSh → interactive 3D mesh (drag / scroll / dbl-click)\n"
            "  3DMa → material colors + shaded sphere preview\n"
            "  Wave → WAV info + Play/Stop buttons (needs aplay or ffplay)\n\n"
            f"Pillow: {HAS_PIL}")

    def open_file(self, path: str):
        try:
            self.omt  = load_omt(path)
            self.edit = {}
            self._refresh_tree()
            self._status(f"Opened  {path}  —  {len(self.omt.chunks)} chunks  OMF{self.omt.version}")
            self.root.title(f"OMF2 Editor — {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Open failed", str(e))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    app = OMF2Editor(root)
    if HAS_DND:
        root.drop_target_register(DND_FILES)
        root.dnd_bind("<<Drop>>", app._on_dnd_drop)
    if len(sys.argv) > 1:
        app.open_file(sys.argv[1])
    elif not HAS_DND:
        app._status("No file open  |  drag+drop needs: pip install tkinterdnd2")
    root.mainloop()


if __name__ == "__main__":
    main()
