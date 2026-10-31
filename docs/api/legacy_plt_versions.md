# Legacy PLT Versions

{class}`~tecio.TecplotPltReader` reads three legacy PLT binary versions in
addition to the current format (v11.2/v19.1): v7.5, v10.7, and v11.1. Support
is **read-only**.

## Version identification

| Magic (first 8 bytes) | Version | Typically written by |
|---|---|---|
| `#!TDV75 ` | v7.5 | Older CFD solvers writing Tecplot binary directly (e.g. CFD++) |
| `#!TDV107` | v10.7 | Tecplot's "2006" classic writer |
| `#!TDV111` | v11.1 | Tecplot's "2008" classic writer |
| `#!TDV112` | v11.2 | Tecplot's "2009"/"2019" classic writer |
| `#!TDV191` | v19.1 | Current format (newer zone-header layout, zone marker `298.0`) |

Byte order can be automatically determined from the INT32 value of `1`
immediately after the magic number.

---

## File header section

| Field | Bytes | Description | v7.5 | v10.7 | v11.1 | v11.2 | v19.1 |
|---|---|---|---|---|---|---|---|
| Magic number | 8 | Exact ASCII version string. | `#!TDV75 ` | `#!TDV107` | `#!TDV111` | `#!TDV112` | `#!TDV191` |
| Byte-order probe | INT32 (4) | Value `1`, in the writer's byte order. | Yes | Yes | Yes | Yes | Yes |
| FileType | INT32 (4) | `0`=FULL, `1`=GRID, `2`=SOLUTION. | **Absent** — no on-disk field at all; every v7.5 file behaves as FULL. | **Absent** — confirmed by byte-reinsertion: reading one here shifts every later field and corrupts the title. | Present | Present | Present |
| Title | INT32×N | Dataset title, one INT32 per character (ASCII code), null-terminated. | Yes | Yes | Yes | Yes | Yes |
| NumVar | INT32 (4) | Number of variables. | Yes | Yes | Yes | Yes | Yes |
| Variable names | (INT32×N) × NumVar | One null-terminated INT32-per-character string per variable, same encoding as Title. | Yes | Yes | Yes | Yes | Yes |
| Zone/Geometry/Text/CustomLabel/UserRec/DatasetAux/VarAux records | varies | Marker-delimited records, looped until EOHMARKER. | Zone (`299.0`), Geometry (`399.0`), Text (`499.0`), CustomLabel (`599.0`) only — no UserRec, DatasetAux, or VarAux markers exist in v7.5 (no discoverable marker in the format description; any other marker raises `PltReadError`). | All markers, same as v11.2/v19.1. | All markers. | All markers. | Zone marker is `298.0` instead of `299.0`; otherwise same set. |
| EOHMARKER | FLOAT32 (4) | Value `357.0`, ends the header. | Yes | Yes | Yes | Yes | Yes |

---

## Zone header (one record per zone, at the Zone marker)

| Field | Bytes | Description | v7.5 | v10.7 | v11.1 | v11.2 | v19.1 |
|---|---|---|---|---|---|---|---|
| Zone marker | FLOAT32 (4) | `299.0` (v7.5/v10.7/v11.1/v11.2) or `298.0` (v19.1). | `299.0` | `299.0` | `299.0` | `299.0` | `298.0` |
| Zone name | INT32×N | Same string encoding as Title. | Yes | Yes | Yes | Yes | Yes |
| ParentZone | INT32 (4) | Unused since Tecplot 2020 R1; still present on disk. | **Absent** | Yes | Yes | Yes | Yes |
| StrandID | INT32 (4) | Static-data strand ID. | **Absent** — v7.5 has no strand concept; every zone reads back `strand_id=0`. | Yes | Yes | Yes | Yes |
| SolutionTime | FLOAT64 (8) | Solution time. | **Absent** — every zone reads back `solution_time=0.0`. | Yes | Yes | Yes | Yes |
| Zone color | INT32 (4) | Unused, `-1` lets Tecplot choose. | Yes (immediately after Zone name) | Yes | Yes | Yes | Yes |
| Format / ZoneType | INT32 (4) | See note below — v7.5 packs packing + topology into one field; every later version has an independent ZoneType field elsewhere and packing isn't stored at all (always BLOCK on disk). | **`Format`**: `0`=ORDERED+BLOCK, `1`=ORDERED+POINT, `2`=FEBLOCK, `3`=FEPOINT. | `ZoneType` (INT32 enum) | `ZoneType` | `ZoneType` | `ZoneType` |
| IMax/JMax/KMax (ORDERED) | INT32×3 (12) | Ordered dimensions. | Yes, literal dimensions (Format 0/1 only). | Yes | Yes | Yes | Yes |
| IMax/JMax/KMax (FE, v7.5 only) | — | v7.5 repurposes these three fields for FE zones: IMax=NumPoints, JMax=NumElements, KMax=element type (`0`=tri, `1`=quad, `2`=tet, `3`=brick, and `4`=line segment, which the spec omits; GUI-confirmed). `5` and `6` are rejected by Tecplot as an invalid element type; `7` is rejected too, with a different message ("Connectivity list is zero length"). | Yes (Format 2/3 only) | n/a | n/a | n/a | n/a |
| Specify-var-location flag + locations | INT32 (4) [+ INT32×NumVar] | `1` = per-variable NODAL/CELL_CENTERED list follows; `0` = all NODAL. | **Absent** — every variable is NODAL, no such field or concept on disk. | Yes | Yes | Yes | Yes |
| Raw face-neighbors-supplied flag | INT32 (4) | Boolean. | **Absent** | Yes | Yes | Yes | Yes |
| Misc. face-neighbor count | INT32 (4) [+ mode INT32 (+ "completely specified" INT32)] | Count of miscellaneous (non-raw) face-neighbor connections; if non-zero, a mode enum follows, and for FE zone types a "completely specified" flag follows the mode. | **Absent** — no face-neighbor concept exists in v7.5 at all. | Present, **except** the "completely specified" flag: confirmed absent by diffing the same real zone written by v11.2 vs. its v10.7 re-export. | Present, including the "completely specified" flag. | Present. | Present. |
| Unidentified always-0 INT32 | INT32 (4) | One extra field, position confirmed, purpose unknown. | n/a | n/a | **Present**, unconditionally, right after the zone type. Always `0` (the data-packing field; non-zero means POINT, which is rejected). | n/a | n/a |
| NumPoints (FE) | INT32 (4) | Node/point count. | See IMax above (same field, reused). | Yes | Yes | Yes | Yes |
| NumFaces / TotalFaceNodes (FEPOLYGON/FEPOLYHEDRON) | v19.1: INT64×2 (16); pre-v19.1: INT32×4 (16, adds NumBoundaryFaces/NumBoundaryConnections) | Poly-zone-only extra counts. | **Not representable** — v7.5's Format field has no polygon/polyhedron encoding at all (only the classic element types). | INT32×4 form | INT32×4 form | INT32×4 form | INT64×2 form |
| NumElements (FE) | INT32 (4) | Element/cell count. | See JMax above (same field, reused). | Yes | Yes | Yes | Yes |
| ICellDim/JCellDim/KCellDim (FE) | INT32×3 (12) | Reserved for future use, always 0 on disk. | **Absent** | Yes | Yes | Yes | Yes |
| Zone-level aux data loop | INT32 (4) flag, then [name string, INT32 value-format, value string] × N | Repeats until the flag is 0. | **Absent** — no zone-aux-data concept or marker exists in v7.5. | Yes | Yes | Yes | Yes |

Note:
    v7.5 geometry/text records: geometry has no `DrawOrder` field (right after
    `Scope`); text's starting location is `X, Y` only (no `Z`) and has no
    `Clipping` field before the string.

---

## Data section (one record per zone, in header order)

| Field | Bytes | Description | v7.5 | v10.7 | v11.1 | v11.2 | v19.1 |
|---|---|---|---|---|---|---|---|
| Zone marker | FLOAT32 (4) | Must match the header's marker for this zone. | `299.0` | `299.0` | `299.0` | `299.0` | `298.0` |
| Repeat-var count / var data formats | v7.5: INT32 count + INT32×count "repeat" var numbers, then INT32×NumVar formats. Later: INT32×NumVar formats only. | Which variables are data type per variable (`1`=Float, `2`=Double), plus (v7.5 only) which variables are "repeated" (shared) from another zone. | `NumRepeatVars` (INT32) then that many repeated-variable numbers (**1-based** on disk, GUI-confirmed), **then** `VarDataFormat` for every variable including repeated ones. | `VarDataFormat` only, no repeat-count/list here (sharing is a separate flag below). | Same as v10.7. | Same. | Same. |
| Passive-variable flag [+ list] | INT32 (4) [+ INT32×NumVar] | `1` = a passive-variable boolean list follows; `0` = every variable active. | **Not representable** — every v7.5 variable has data or is "repeated"; there is no third (passive) state. | Yes | Yes | Yes | Yes |
| Variable-sharing flag [+ list] | INT32 (4) [+ INT32×NumVar] | `1` = a source-zone value per variable follows. | No such flag/list; handled by the `NumRepeatVars`/repeated-variable-numbers fields above instead. The source is always the immediately preceding zone, and a repeated variable's data block is omitted from the zone data. Both are **confirmed by loading hand-built files in the Tecplot GUI** (see Confidence section); the spec states neither. The public `shared_zone` this library reports is that preceding zone's 0-based `zone_index`. | On disk: `-1`=not shared, else the **0-based** source zone index and reported as-is by `shared_zone`. Confirmed against real files: writing via the actual TecIO C library and reading back with this parser round-trips correctly (`tests/test_write.py`). | Same decoding as v10.7. | Same | Same |
| Connectivity-sharing | INT32 (4) | Source-zone reference for the node map. | Different shape: an explicit **per-zone boolean** "repeat adjacency list from previous zone" (`1`/`0`), read only for FE zones, always relative to the *immediately preceding* zone — this part of the spec is explicit, not inferred. When set, the public `shared_connectivity` this library reports is that preceding zone's 0-based `zone_index`. | On disk: `-1`=not shared, else the **0-based** source zone index (any source zone, not just the previous one). Reported as-is by `shared_connectivity`. Confirmed against real files via the same C-library round-trip test. | Same as v10.7. | Same | Same |
| Min/Max pairs | FLOAT64×2 (16) per active, non-passive, non-shared variable | Per-variable value range. | **Absent** — no such field in v7.5. | Yes | Yes | Yes | Yes |
| Variable data blocks | BLOCK: contiguous run per variable. POINT: interleaved row-major, all active variables per node/element together. | The actual numeric payload. | **Both layouts real, both implemented**: `Format` (see zone header) selects BLOCK or POINT. POINT is confirmed byte-exact against a real file (see Confidence section); v7.5 BLOCK is not exercised by any real sample we have, though it shares the same simple contiguous-block code path validated extensively for v10.7/v11.1/v11.2. | BLOCK only — PLT's writer here never emits POINT; POINT is only real on the DAT (ASCII) side of this library. Confirmed against real files. | Same | Same | Same |
| Connectivity block (FE only) | INT32×(L×NumElements) | `L`=3/4/4/8 for triangle/quad/tet/brick, per-element node indices. | **1-based on disk already** — confirmed against a real, large (1.5M-node) file: every index lands in `[1, NumPoints]`, and the byte walk lands exactly on EOF. This library subtracts 1 on read so `node_map` is 0-based like every other format: this is the one place a legacy version's on-disk convention needs converting. | **0-based on disk**, returned as-is (see {doc}`index` for the indexing rules). | Same as v10.7 | Same | Same |
| Raw face-neighbors (if flagged) | INT32×(NumElements×FacesPerElement) | Dense 1-based neighbor-element table, `0`=boundary. | **Not representable** — no face-neighbor concept in v7.5. | Yes | Yes | Yes | Yes |
| Miscellaneous face-neighbor connections (if count > 0) | Mode-dependent: fixed-width for the two ONE_TO_ONE modes, ragged (record carries its own length) for the two ONE_TO_MANY modes. | Explicit face-neighbor connection list. | **Not representable.** | Yes | Yes | Yes | Yes |
| Poly face map (FEPOLYGON/FEPOLYHEDRON only) | Several INT32 blocks (face-node offsets/counts, face nodes, left/right elements, boundary connections). | Face connectivity for poly zones. | **Not representable** — v7.5 has no polygon/polyhedron zone type. | Decoded for v111 and later into `TecplotPolyZoneReader` (v111 stores the boundary connection zones as INT16, later versions as INT32). **v107 poly zones raise `PltReadError`.** v19.1 poly zones follow the guide (INT64 counts, per-face node counts for polyhedra, no boundary connections) but **no v19.1 poly file has been available to test this**. | Same | Same | Same |

---

## Element-type / topology capability matrix

What each version can express on disk:

| Capability | v7.5 | v10.7 | v11.1 | v11.2 | v19.1 |
|---|---|---|---|---|---|
| ORDERED (structured) zones | Yes (Format 0/1) | Yes | Yes | Yes | Yes |
| FETRIANGLE / FEQUADRILATERAL / FETETRAHEDRON / FEBRICK | Yes, via `KMax` element-type code | Yes | Yes | Yes | Yes |
| FELINESEG | Yes, `KMax` = `4` (2 nodes per element) | Yes | Yes | Yes | Yes |
| Pyramids / prisms | **No type of their own** (`KMax` `5` and `6` are rejected by Tecplot). Stored as FEBRICK cells with repeated nodes. | Same | Same | Same | Same |
| FEPOLYGON / FEPOLYHEDRON | No | Yes | same | same | same |
| BLOCK data packing | Yes | Yes | Yes | Yes | Yes |
| POINT data packing | Yes | Unknown | Unknown | No | No |
| Reading BLOCK (this library) | Implemented | Validated | Validated | Validated | Validated |
| Reading POINT (this library) | Implemented | n/a | n/a | n/a | n/a |

---

## Known limitations and unconfirmed items

Not implementation gaps to "get to," these are either genuine absences in the
on-disk format or explicitly-flagged-as-inferred logic:

- **v7.5 variable sharing** (repeated variables): the spec gives only the field
  layout. The 1-based numbering, previous-zone source, omitted data block,
  chains of repeats across zones, repeated columns dropped from the middle of
  POINT rows, and chained connectivity repeats were all confirmed by loading
  hand-built files in the Tecplot GUI.
- **v11.1's extra `0` INT32**: sits right after the zone type, and its value is
  confirmed by byte-reinsertion. It is the data-packing field (0 = BLOCK).
- **Bit-packed data** (DataType 6): not supported for any version, raises
  `NotImplementedError`.

## Not supported in v7.5 files

- Passive variables (every variable has data, or is a repeat; no third state).
- Cell-centered data (every value is nodal).
- StrandID / SolutionTime (every zone is static).
- Face-neighbor connections of any kind.
- Dataset, zone, or variable auxiliary data.
