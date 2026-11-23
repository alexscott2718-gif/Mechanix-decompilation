# Jimmy Neutron: Boy Genius mode for omf2_editor

This adds a Jimmy Neutron mode on top of awefan's `omf2_editor.py`. The editor itself is not changed; the mode
subclasses it.

    python jn/jn_mode.py path/to/level1.omt --gam            # finds gam/level1.gam beside the omt folder
    python jn/jn_mode.py path/to/level1.omt --gam path/to/level1.gam

No game files are included. Point it at your own install's `omt/` and `gam/` folders.

## What it does

- **Opens a level `.omt` with its `.gam`.** Every `.gam` entity is drawn as a marker in the 3D view, coloured by
  class. Level 1 has 320 positioned entities.
- **Entity panel:** a list with a class filter and Go to. Double-click a property to edit it.
- **Moving an entity's marker** updates its `PositionX/Y/Z`. Save writes the `.gam` back.
- **Collision boxes (`BLOCK*`) are hidden.** They are the shapes with no material; without this the level is hidden
  inside a grey box.
- The triangle limit is raised to 20,000 so whole levels draw.

## `gam.py`

A lossless `.gam` reader and writer, written from the format description in
gitea.unixnight.com/UnixNight/Neutron `docs/gam-format.md`; none of that repo's code is copied.

- Big-endian. Each entity is a class code, its properties and a 4-byte trailer.
- Property types: 1 = text, 3 = float, 6 = int.
- Load and save gives a byte-identical file for all 35 `.gam` files in the retail game.

## Tested

- All 100 retail JN `.omt` files open in the editor, and a save with no edits reproduces them.
- Move an entity, save, reload: the new position is kept.
- **Tested in the original game (NeutronSW.exe, 2026-11-23).** In `level1b.gam` (Jimmy's lab), Jimmy's start was
  moved from (5071.9, 26.7, 824.1) to (4308, 26.7, 1300); 12 bytes of the file changed. A new game then spawned Jimmy at
  (4308.5, 76.4, 1300.0), read from the running game's memory.
  - A new game starts Jimmy at the `STRT` entity tagged `STARTEXP`. The `3JIM` entity sits at the same spot in the
    retail file, so both were moved; moving `STARTEXP` alone hasn't been tested.
  - The game sets Jimmy about 50 units above the `.gam` height, onto the floor.
  - The retail `.gam` files install read-only. Clear that before copying an edited file over one.

## Notes

- **Coordinates:** JN `.omt` vertices are already in world space, and `.gam` positions use the same space.
- **The Blender add-on mirrors levels.** The original Max→OMT conversion is (x,y,z)→(x,z,y), a handedness swap;
  the add-on uses (x,−z,y).
