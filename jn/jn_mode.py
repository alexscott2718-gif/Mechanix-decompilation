"""Jimmy Neutron mode for awefan's OMF2 editor.

Extends tools/omf2_editor.py (unmodified) with:
  * a "Jimmy Neutron" menu: open the level's .gam entity file, show the entity list
  * entity markers drawn into "Level Assembly" (coloured by class)
  * select a marker in edit mode (F8, click) and move it (M, then X/Y/Z to lock an
    axis, Enter to apply): the entity's PositionX/Y/Z is updated
  * an entity panel: filter by class, edit any text/float/int property, save the .gam

Level .omt geometry in JN is already in world coordinates, and .gam positions share
that space (checked: e.g. all 320 positioned entities in level1 fall inside level1.omt).
"""
import math
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
sys.path.insert(0, os.path.dirname(HERE))  # omf2_editor.py at the repo root
sys.path.insert(0, HERE)
import omf2_editor as ed  # noqa: E402
import gam  # noqa: E402

ENT_PREFIX = "ENT:"

# marker colours by class code (RGB); anything else is red
CLASS_COLOURS = {
    "3JIM": (60, 230, 80),     # Jimmy / player start
    "STRT": (60, 230, 80),
    "3PIC": (255, 215, 0),     # pickups
    "3PAT": (110, 140, 255),   # patrol points
    "3CAM": (0, 220, 230),     # cameras
    "3MCA": (0, 170, 190),
    "3SOU": (200, 90, 255),    # sounds
    "3TRE": (255, 140, 0),     # triggers
    "3NEU": (255, 255, 255),   # neutron (generic) objects
    "3AIT": (255, 70, 70),     # AI
    "LOAD": (255, 105, 180),   # level loads
}
CLASS_NAMES = {
    "3JIM": "Jimmy", "STRT": "start", "3PIC": "pickup", "3PAT": "patrol point",
    "3CAM": "camera", "3MCA": "camera", "3SOU": "sound", "3TRE": "trigger",
    "3NEU": "object", "3AIT": "AI", "LOAD": "level load", "3BUT": "button",
    "3DOR": "door", "3CHK": "checkpoint", "3ROK": "rock", "3RED": "red",
}


def _octahedron(cx, cy, cz, r):
    v = [(cx + r, cy, cz), (cx - r, cy, cz), (cx, cy + r * 1.6, cz),
         (cx, cy - r * 0.4, cz), (cx, cy, cz + r), (cx, cy, cz - r)]
    f = [(2, 4, 0), (2, 0, 5), (2, 5, 1), (2, 1, 4),
         (3, 0, 4), (3, 5, 0), (3, 1, 5), (3, 4, 1)]
    return v, f


def _find_gam_for(omt_path):
    """level1.omt -> ../gam/Level1.gam or ./Level1.gam (case-insensitive)."""
    stem = os.path.splitext(os.path.basename(omt_path))[0].lower()
    d = os.path.dirname(os.path.abspath(omt_path))
    for cand_dir in (d, os.path.join(os.path.dirname(d), "gam"), os.path.join(d, "gam")):
        if not os.path.isdir(cand_dir):
            continue
        for fn in os.listdir(cand_dir):
            if os.path.splitext(fn)[0].lower() == stem and fn.lower().endswith(".gam"):
                return os.path.join(cand_dir, fn)
    return None


class JNEditor(ed.OMF2Editor):
    def __init__(self, root):
        self.gam = None
        self._gam_dirty = False
        self._ent_panel = None
        self._show_ents = None
        super().__init__(root)
        self.root.title("OMF2 Editor — Jimmy Neutron mode")
        self.mesh_cv.MAX_TRIS = 20_000

    # ---- menu ---------------------------------------------------------------
    def _build_menu(self):
        super()._build_menu()
        mb = self.root.nametowidget(self.root["menu"])
        jm = tk.Menu(mb, tearoff=0)
        jm.add_command(label="Open level's .gam (auto)", command=self.jn_open_gam_auto)
        jm.add_command(label="Open .gam…", command=self.jn_open_gam)
        jm.add_command(label="Entity panel", command=self.jn_panel)
        jm.add_separator()
        jm.add_command(label="Save .gam", command=self.jn_save_gam)
        jm.add_command(label="Save .gam as…", command=lambda: self.jn_save_gam(ask=True))
        jm.add_separator()
        jm.add_command(label="Help", command=self.jn_help)
        mb.insert_cascade(2, label="Jimmy Neutron", menu=jm)

    def jn_help(self):
        messagebox.showinfo("Jimmy Neutron mode", (
            "1. Open a level .omt (e.g. jn-work\\omt\\level1.omt).\n"
            "2. Jimmy Neutron > Open level's .gam (auto) loads its entities.\n"
            "3. 3D Mesh tab > Level Assembly shows the level with entity markers.\n"
            "   Green = Jimmy/start, yellow = pickups, blue = patrol points,\n"
            "   cyan = cameras, purple = sounds, orange = triggers, white = objects.\n"
            "4. F8 = edit mode. Click a marker to select it (the panel follows).\n"
            "   M = move, X/Y/Z = lock axis, Enter = apply, Esc = cancel.\n"
            "5. Entity panel: double-click a property to edit it.\n"
            "6. Jimmy Neutron > Save .gam writes the entity file.\n\n"
            "Work on copies (jn-work), never the installed game."))

    # ---- gam loading/saving -------------------------------------------------------
    def jn_open_gam_auto(self):
        if not self.omt:
            messagebox.showinfo("No level", "Open a level .omt first."); return
        p = _find_gam_for(self.omt.path)
        if not p:
            messagebox.showinfo("No .gam found",
                                "No matching .gam next to the level or in ../gam.\n"
                                "Use Open .gam… to pick one."); return
        self._load_gam(p)

    def jn_open_gam(self):
        p = filedialog.askopenfilename(title="Open .gam", filetypes=[("JN entities", "*.gam"), ("All", "*.*")])
        if p:
            self._load_gam(p)

    def _load_gam(self, path):
        if self._gam_dirty and not messagebox.askyesno("Unsaved entities", "Discard unsaved .gam changes?"):
            return
        try:
            self.gam = gam.GamFile.load(path)
        except Exception as e:
            messagebox.showerror("Could not read .gam", str(e)); return
        self._gam_dirty = False
        n = sum(1 for e in self.gam.entities if e.position)
        self._status(f"Loaded {os.path.basename(path)}: {len(self.gam.entities)} entities ({n} positioned). "
                     "Press Level Assembly to see them.")
        self.jn_panel()
        if self.omt:
            self._load_level_assembled()

    def jn_save_gam(self, ask=False):
        if not self.gam:
            messagebox.showinfo("No .gam", "No entity file is open."); return
        path = self.gam.path
        if ask or not path:
            path = filedialog.asksaveasfilename(title="Save .gam as", defaultextension=".gam",
                                                initialfile=os.path.basename(path or "level.gam"))
            if not path:
                return
        try:
            self.gam.save(path)
        except Exception as e:
            messagebox.showerror("Save failed", str(e)); return
        self._gam_dirty = False
        self._status(f"Saved {path}")
        self._panel_title()

    # ---- markers in the level view ---------------------------------------------------
    def _load_level_assembled(self):
        cv = self.mesh_cv
        orig_load, orig_ranges = cv.load_mesh, cv.set_shape_vranges
        extra_ranges = {}

        def load_with_markers(verts, tris, info="", textures=None):
            verts, tris = list(verts), list(tris)
            if self.gam and verts:
                xs = [v[0] for v in verts]; zs = [v[2] for v in verts]
                r = max(max(xs) - min(xs), max(zs) - min(zs)) * 0.0035
                r = min(max(r, 12.0), 120.0)
                shown = self._shown_classes()
                for i, e in enumerate(self.gam.entities):
                    pos = e.position
                    if not pos or (shown is not None and e.cls not in shown):
                        continue
                    key = f"{ENT_PREFIX}{i}"
                    mv, mf = _octahedron(*pos, r)
                    base = len(verts); verts.extend(mv)
                    extra_ranges[key] = (base, base + len(mv))
                    col = CLASS_COLOURS.get(e.cls, (230, 40, 40))
                    for a, b, c in mf:
                        pa, pb, pc = mv[a], mv[b], mv[c]
                        ux, uy, uz = (pb[k] - pa[k] for k in range(3))
                        vx, vy, vz = (pc[k] - pa[k] for k in range(3))
                        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
                        ln = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
                        tris.append(((base + a, base + b, base + c), col, (nx / ln, ny / ln, nz / ln), None, 0.0, key))
                info = info + f"+ {len(extra_ranges)} entities  "
            return orig_load(verts, tris, info=info, textures=textures)

        def ranges_with_markers(d):
            d = dict(d); d.update(extra_ranges)
            return orig_ranges(d)

        # JN levels are boxed in by invisible collision volumes (BLOCKTop, BLOCKING_07, ...):
        # shapes whose polygons carry no material at all. The game never draws them, so
        # treat them like markers (hidden while "Hide markers" is ticked).
        orig_event = ed._shape_all_event_mats

        def event_or_invisible(faces, mat_names):
            if faces and all(not f[4] for f in faces):
                return True
            return orig_event(faces, mat_names)

        cv.load_mesh, cv.set_shape_vranges = load_with_markers, ranges_with_markers
        ed._shape_all_event_mats = event_or_invisible
        try:
            super()._load_level_assembled()
        finally:
            cv.load_mesh, cv.set_shape_vranges = orig_load, orig_ranges
            ed._shape_all_event_mats = orig_event

    def _shown_classes(self):
        if not self._ent_panel or not self._ent_panel.winfo_exists():
            return None
        v = self._class_var.get()
        return None if v == "All" else {v.split()[0]}

    # ---- selection / move callbacks ------------------------------------------------
    def _on_selection_changed(self, key):
        if key and key.startswith(ENT_PREFIX):
            i = int(key[len(ENT_PREFIX):])
            e = self.gam.entities[i]
            self._status(f"Entity {i}: {e.cls} {e.tag!r} at {tuple(round(c, 1) for c in e.position)}  "
                         "(M = move, X/Y/Z lock axis, Enter apply)")
            self._panel_select(i)
            return
        super()._on_selection_changed(key)

    def _on_transform_applied(self, shape_key, new_verts_global):
        if not shape_key.startswith(ENT_PREFIX):
            return super()._on_transform_applied(shape_key, new_verts_global)
        i = int(shape_key[len(ENT_PREFIX):])
        e = self.gam.entities[i]
        a, b = self.mesh_cv._shape_vrange[shape_key]
        mv = new_verts_global[a:b]
        old = e.position
        # marker vertex 0/1 are +-r on X, 4/5 +-r on Z, so their midpoints give the centre;
        # Y from the X pair (the marker is taller above than below the point).
        nx = (mv[0][0] + mv[1][0]) / 2; nz = (mv[4][2] + mv[5][2]) / 2; ny = (mv[0][1] + mv[1][1]) / 2
        e.set("PositionX", nx); e.set("PositionY", ny); e.set("PositionZ", nz)
        self._gam_dirty = True
        self._status(f"Moved {e.cls} {e.tag!r}: {tuple(round(c, 1) for c in old)} -> "
                     f"{tuple(round(c, 1) for c in e.position)}  (unsaved: Jimmy Neutron > Save .gam)")
        self._panel_refresh_row(i)
        self._panel_title()

    # ---- entity panel -----------------------------------------------------------------
    def jn_panel(self):
        if self._ent_panel and self._ent_panel.winfo_exists():
            self._ent_panel.lift(); self._panel_fill(); return
        w = tk.Toplevel(self.root); w.geometry("560x720"); self._ent_panel = w
        top = ttk.Frame(w); top.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(top, text="Class:").pack(side=tk.LEFT)
        self._class_var = tk.StringVar(value="All")
        self._class_cb = ttk.Combobox(top, textvariable=self._class_var, state="readonly", width=24)
        self._class_cb.pack(side=tk.LEFT, padx=4)
        self._class_cb.bind("<<ComboboxSelected>>", lambda _: (self._panel_fill(), self._reassemble()))
        ttk.Button(top, text="Go to", command=self._panel_goto).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Save .gam", command=self.jn_save_gam).pack(side=tk.RIGHT)

        pw = ttk.PanedWindow(w, orient=tk.VERTICAL); pw.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        f1 = ttk.Frame(pw); f2 = ttk.Frame(pw); pw.add(f1, weight=3); pw.add(f2, weight=2)
        cols = ("i", "class", "tag", "x", "y", "z")
        self._ent_tree = ttk.Treeview(f1, columns=cols, show="headings", selectmode="browse")
        for c, wd in zip(cols, (40, 90, 140, 80, 80, 80)):
            self._ent_tree.heading(c, text=c); self._ent_tree.column(c, width=wd, anchor=tk.W)
        sb = ttk.Scrollbar(f1, command=self._ent_tree.yview); self._ent_tree.configure(yscrollcommand=sb.set)
        self._ent_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._ent_tree.bind("<<TreeviewSelect>>", self._panel_on_pick)
        self._ent_tree.bind("<Double-1>", lambda _: self._panel_goto())

        self._prop_tree = ttk.Treeview(f2, columns=("prop", "type", "value"), show="headings")
        for c, wd in (("prop", 150), ("type", 50), ("value", 300)):
            self._prop_tree.heading(c, text=c); self._prop_tree.column(c, width=wd, anchor=tk.W)
        sb2 = ttk.Scrollbar(f2, command=self._prop_tree.yview); self._prop_tree.configure(yscrollcommand=sb2.set)
        self._prop_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); sb2.pack(side=tk.RIGHT, fill=tk.Y)
        self._prop_tree.bind("<Double-1>", self._panel_edit_prop)
        self._panel_fill()

    def _panel_title(self):
        if self._ent_panel and self._ent_panel.winfo_exists() and self.gam:
            star = " *" if self._gam_dirty else ""
            self._ent_panel.title(f"Entities — {os.path.basename(self.gam.path or '?')}{star}")

    def _panel_fill(self):
        if not self.gam:
            return
        self._panel_title()
        counts = {}
        for e in self.gam.entities:
            counts[e.cls] = counts.get(e.cls, 0) + 1
        opts = ["All"] + [f"{c} ({CLASS_NAMES.get(c, '?')}, {n})" for c, n in sorted(counts.items(), key=lambda t: -t[1])]
        self._class_cb["values"] = opts
        want = self._shown_classes()
        self._ent_tree.delete(*self._ent_tree.get_children())
        for i, e in enumerate(self.gam.entities):
            if want is not None and e.cls not in want:
                continue
            self._ent_tree.insert("", tk.END, iid=str(i), values=self._row(i, e))

    @staticmethod
    def _row(i, e):
        p = e.position or ("", "", "")
        return (i, e.cls, e.tag, *[f"{c:.1f}" if c != "" else "" for c in p])

    def _panel_refresh_row(self, i):
        if self._ent_panel and self._ent_panel.winfo_exists() and self._ent_tree.exists(str(i)):
            self._ent_tree.item(str(i), values=self._row(i, self.gam.entities[i]))
            self._panel_show_props(i)

    def _panel_select(self, i):
        if self._ent_panel and self._ent_panel.winfo_exists() and self._ent_tree.exists(str(i)):
            self._ent_tree.selection_set(str(i)); self._ent_tree.see(str(i))

    def _panel_on_pick(self, _=None):
        sel = self._ent_tree.selection()
        if not sel:
            return
        i = int(sel[0]); self._panel_show_props(i)
        key = f"{ENT_PREFIX}{i}"
        if key in getattr(self.mesh_cv, "_shape_vrange", {}) and self.mesh_cv._selected_key != key:
            self.mesh_cv._selected_key = key; self.mesh_cv.redraw()

    def _panel_show_props(self, i):
        self._prop_tree.delete(*self._prop_tree.get_children())
        tn = {1: "text", 3: "float", 6: "int"}
        for k, p in enumerate(self.gam.entities[i].props):
            v = p.value
            shown = v.hex() if isinstance(v, bytes) else (f"{v:.4f}" if isinstance(v, float) else v)
            self._prop_tree.insert("", tk.END, iid=f"{i}:{k}", values=(p.name, tn.get(p.type, f"raw{p.type}"), shown))

    def _panel_edit_prop(self, _=None):
        sel = self._prop_tree.selection()
        if not sel:
            return
        i, k = map(int, sel[0].split(":"))
        p = self.gam.entities[i].props[k]
        if not p.editable:
            messagebox.showinfo("Not editable", f"{p.name} has an undecoded type ({p.type}); it is kept as-is."); return
        new = simpledialog.askstring("Edit property", f"{p.name}:", initialvalue=str(p.value), parent=self._ent_panel)
        if new is None:
            return
        try:
            p.value = new
        except Exception as e:
            messagebox.showerror("Invalid value", str(e)); return
        self._gam_dirty = True
        self._panel_refresh_row(i); self._panel_title()
        if p.name.startswith("Position"):
            self._reassemble()

    def _panel_goto(self):
        sel = self._ent_tree.selection()
        if not sel:
            return
        e = self.gam.entities[int(sel[0])]
        if not e.position:
            return
        cv = self.mesh_cv
        x, y, z = e.position
        span = getattr(cv, "_last_span", 2000.0)
        d = max(span * 0.03, 150.0)
        cv._cam_yaw, cv._cam_pitch = 25.0, -25.0
        yr, pr = math.radians(cv._cam_yaw), math.radians(cv._cam_pitch)
        cv._cam_pos = [x + math.sin(yr) * math.cos(pr) * d, y - math.sin(pr) * d, z - math.cos(yr) * math.cos(pr) * d]
        cv._selected_key = f"{ENT_PREFIX}{int(sel[0])}"
        cv.redraw()

    def _reassemble(self):
        if self.omt and getattr(self, "_last_assembly_cmd", None) == self._load_level_assembled:
            self._load_level_assembled()

    def _on_close(self):
        if self._gam_dirty and not messagebox.askyesno("Unsaved entities", "Quit without saving the .gam?"):
            return
        super()._on_close()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root = ed.TkinterDnD.Tk() if ed.HAS_DND else tk.Tk()
    app = JNEditor(root)
    if argv:
        app.open_file(argv[0])
        if "--gam" in argv:
            root.after(300, app.jn_open_gam_auto)
    root.mainloop()


if __name__ == "__main__":
    main()
