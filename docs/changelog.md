# Changelog

Notable changes to this project are documented below.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- Node-map utilities for classic finite-element zones, importable from `tecio`:
  - `classify_cells`, `shape_counts` and `degenerate_edges` detect cells
    written as `FEBRICK` or `FEQUADRILATERAL` with repeated node indices
    (tetrahedra, pyramids, prisms, triangles).
  - `collapsed_zone_type` and `collapse_cells` collapse such cells to their
    natural zone type (`FEBRICK` to `FETETRAHEDRON`, `FEQUADRILATERAL` to
    `FETRIANGLE`) and keep the cell handedness. `expand_cells` goes the other
    way.
  - `split_cells_by_shape` separates padded cells in a `FEBRICK` or
    `FEQUADRILATERAL` zone by shape for writing them as a `FEMIXED`
    zone. Pyramids and prisms keep the handedness of the brick they came from.
  - `cell_face_neighbors` finds the cell across every face of every cell from
    the node map alone, for triangles, quadrilaterals, tetrahedra and
    bricks. It returns `-1` for boundary faces.
- `TecplotPolyZoneReader.element_faces()` and `element_nodes()` return the
  faces and the distinct nodes of every element of a polygon or polyhedron
  zone as `(offsets, flat_array)`, built from the face left and right
  elements.
- `tecelretyp` console script retypes FE zones whose cells are all upcast with
  repeated nodes, which gives a more compact node map and more descriptive
  zone metadata.
- Mixed finite-element (`FEMIXED`) zones suppport (read and write) for SZPLT
  files.

---

```{include} changelog/v0.4.1.md
```

---

```{include} changelog/v0.4.0.md
```

---

```{include} changelog/v0.3.3.md
```

---

```{include} changelog/v0.3.2.md
```

---

```{include} changelog/v0.3.1.md
```

---

```{include} changelog/v0.3.0.md
```

---

```{include} changelog/v0.2.4.md
```

---

```{include} changelog/v0.2.3.md
```

---

```{include} changelog/v0.2.2.md
```

---

```{include} changelog/v0.2.1.md
```

---

```{include} changelog/v0.2.0.md
```

---

```{include} changelog/v0.1.1.md
```

---

```{include} changelog/v0.1.0.md
```

---

[Unreleased]: https://github.com/meersman/tecio-python/compare/v0.4.1...HEAD

```{toctree}
:hidden:
:maxdepth: 1
:caption: Changelog
:glob:

changelog/*
```
