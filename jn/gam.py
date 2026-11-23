"""Lossless reader/writer for Jimmy Neutron .gam entity files (big-endian).

Written from the format description in UnixNight/Neutron docs/gam-format.md:
  file:     char[4] level_code, u32 entity_count, entity[]
  entity:   char[4] class_code, u32 property_count, property[], byte[4] trailer
  property: u8 name_len, name, u32 type, u32 payload_len, [u8 str_len if type==1], payload
Types: 1 text, 3 float32, 6 int32; anything else is kept as raw bytes.
Unchanged properties are written back byte for byte.
"""
import struct

TEXT, FLOAT, INT = 1, 3, 6


class Prop:
    __slots__ = ("name", "type", "raw")

    def __init__(self, name, typ, raw):
        self.name, self.type, self.raw = name, typ, raw

    @property
    def value(self):
        if self.type == TEXT: return self.raw.decode("latin-1")
        if self.type == FLOAT and len(self.raw) == 4: return struct.unpack(">f", self.raw)[0]
        if self.type == INT and len(self.raw) == 4: return struct.unpack(">i", self.raw)[0]
        return self.raw

    @value.setter
    def value(self, v):
        if self.type == TEXT:
            b = str(v).encode("latin-1")
            if len(b) > 255: raise ValueError("text longer than 255 bytes")
            self.raw = b
        elif self.type == FLOAT: self.raw = struct.pack(">f", float(v))
        elif self.type == INT: self.raw = struct.pack(">i", int(v))
        else: raise ValueError(f"property type {self.type} is opaque; not editable")

    @property
    def editable(self):
        return self.type in (TEXT, FLOAT, INT)


class Entity:
    def __init__(self, cls, props, trailer):
        self.cls, self.props, self.trailer = cls, props, trailer

    def get(self, name, default=None):
        for p in self.props:
            if p.name == name: return p.value
        return default

    def set(self, name, value):
        for p in self.props:
            if p.name == name:
                p.value = value; return
        raise KeyError(name)

    @property
    def position(self):
        x, y, z = (self.get("Position" + a) for a in "XYZ")
        return None if x is None else (x, y, z)

    @property
    def tag(self):
        return self.get("ObjectTag", "")


class GamFile:
    def __init__(self, level, entities, path=None):
        self.level, self.entities, self.path = level, entities, path

    @classmethod
    def load(cls, path):
        d = open(path, "rb").read(); p = 0

        def take(n):
            nonlocal p
            if p + n > len(d): raise ValueError(f"truncated at 0x{p:x}")
            b = d[p:p + n]; p += n; return b

        level = take(4).decode("latin-1"); (n,) = struct.unpack(">I", take(4))
        ents = []
        for _ in range(n):
            ec = take(4).decode("latin-1"); (np_,) = struct.unpack(">I", take(4))
            props = []
            for _ in range(np_):
                name = take(take(1)[0]).decode("latin-1")
                typ, ln = struct.unpack(">II", take(8))
                if typ == TEXT:
                    if take(1)[0] != ln: raise ValueError(f"text length prefix mismatch at 0x{p:x}")
                props.append(Prop(name, typ, take(ln)))
            ents.append(Entity(ec, props, take(4)))
        if p != len(d): raise ValueError(f"{len(d) - p} trailing bytes")
        return cls(level, ents, path)

    def to_bytes(self):
        out = bytearray(self.level.encode("latin-1") + struct.pack(">I", len(self.entities)))
        for e in self.entities:
            out += e.cls.encode("latin-1") + struct.pack(">I", len(e.props))
            for pr in e.props:
                nb = pr.name.encode("latin-1")
                out += bytes([len(nb)]) + nb + struct.pack(">II", pr.type, len(pr.raw))
                if pr.type == TEXT: out += bytes([len(pr.raw)])
                out += pr.raw
            out += e.trailer
        return bytes(out)

    def save(self, path=None):
        path = path or self.path
        open(path, "wb").write(self.to_bytes())
        self.path = path


def read_gam(path):
    """Compatibility helper: (level_code, [(class_code, {name: value})])."""
    g = GamFile.load(path)
    return g.level, [(e.cls, {p.name: p.value for p in e.props}) for e in g.entities]
