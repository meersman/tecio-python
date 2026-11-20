# Changelog

Notable changes to this project are documented below.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- `tectable` console script converts a delimited text table to Tecplot zones
  with additional flags for filtering or grouping like data into zones.

### Fixed

- The PLT reader sized cell-centered variables of ordered zones as `I*J*(K-1)`
  values for every zone. TecIO pads every dimension except the last one with
  more than one node. So the PLT reader did not correctly calculate the number
  of values to read for cell-centered data in one or two dimension while in
  three dimensions the correct number of values were read. This bug did not
  affect SZL or DAT readers.

---

```{include} changelog/v0.4.2.md
```

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

[Unreleased]: https://github.com/meersman/tecio-python/compare/v0.4.2...HEAD

```{toctree}
:hidden:
:maxdepth: 1
:caption: Changelog
:glob:

changelog/*
```
