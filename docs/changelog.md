# Changelog

Notable changes to this project are documented below.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- Polygon and polyhedron zones (FEPOLYGON and FEPOLYHEDRON) can be read and
  written in PLT and DAT files. They are returned as the new
  `TecplotPolyZoneReader`, a zone class separate from the ordered and FE
  classes, with 0-based `face_nodes`, `face_node_counts`,
  `face_left_elements`, `face_right_elements` and boundary connection arrays,
  returned as stored on disk. New `write_poly_zone` writer method (BLOCK
  packing only; SZPLT raises `NotImplementedError`). Connectivity sharing and
  append mode work with poly zones. Every CLI tool that rewrites a file
  (`teconvert`, `tecextract`, `tecfix`, `tecscale`, `tecaux`, `tecmerge` and
  `tecslice`) now carries poly zones through, and exits with an error when
  asked to write them to SZPLT.  `tecmerge --merge-grid` writes an identical
  poly grid once and shares it across zones (for example transient steps),
  matching on a hash of the full face map.  `tecdump` reports poly counts and
  `tec2mat` exports a 1-based face map.  Legacy PLT v111 poly zones are
  readable.

### Fixed

- `tecpolybconn142` wrapper argument types (boundary connection zones are
  INT32 values).
- PLT v111 zone header: the extra INT32 follows the zone type.
- The ParaView reader plugin now follows the 0-indexed API (variable
  auxiliary data and connectivity were off by one, which made it fail on
  every file).
- The ParaView writer plugin could not be loaded by ParaView because a
  `@dataclass` failed at import, and it lacked the `Write()` method and input
  type that ParaView requires of a writer. Failures in the writer now report an
  error to ParaView instead of success. Unnamed blocks are labeled
  `Zone 0`, `Zone 1`, ... to match the 0-indexed API.

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

[Unreleased]: https://github.com/meersman/tecio/compare/v0.4.0...HEAD

```{toctree}
:hidden:
:maxdepth: 1
:caption: Changelog
:glob:

changelog/*
```
