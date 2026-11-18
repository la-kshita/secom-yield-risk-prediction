# Raw data

The SECOM data files are **not committed** to this repository: `secom.data` is
54 MB. Download them once and place the three files in this directory.

## Download

1. Open the UCI dataset page: <https://archive.ics.uci.edu/dataset/179/secom>
2. Download the archive, or fetch it directly:

   ```powershell
   Invoke-WebRequest -Uri "https://archive.ics.uci.edu/static/public/179/secom.zip" -OutFile secom.zip
   Expand-Archive -Path secom.zip -DestinationPath . -Force
   Remove-Item secom.zip
   ```

The archive is ~2 MB compressed and contains exactly three files, which belong
directly in this folder:

```text
data/secom.data         590 process measurements, 1,567 rows
data/secom_labels.data  pass/fail label and timestamp, 1,567 rows
data/secom.names        variable descriptions
```

## Integrity check

These are the files this project was built with. If your downloads differ, the
numbers in the notebook and README will not reproduce exactly.

| File | Size (bytes) | SHA-256 |
|---|---|
| `secom.data` | 5,389,983 | `20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c` |
| `secom_labels.data` | 40,638 | `126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03` |
| `secom.names` | 4,223 | `6d91b0b46cdee03064ee3e3112f937c1b3f7fcd9933575794ec07974e6f1ea59` |

Verify on Windows PowerShell:

```powershell
Get-FileHash secom.data, secom_labels.data, secom.names -Algorithm SHA256
```

On macOS or Linux:

```bash
shasum -a 256 secom.data secom_labels.data secom.names
```

## What is in the files

- `secom.data`: whitespace-delimited, **no header row**, 590 anonymous sensor
  columns. Missing readings are stored as the literal `NaN`.
- `secom_labels.data`: two whitespace-delimited columns — the label (`-1` = pass,
  `1` = fail) and a quoted timestamp.
- The original data has **104 failures out of 1,567 observations** (6.6%).
- The timestamp column is deliberately **not used as a feature**, because it
  encodes when a row was produced and would leak temporal information.

## License and citation

SECOM is distributed under CC BY 4.0. If you use it, cite:

> McCann, M. & Johnston, A. (2008). SECOM. UCI Machine Learning Repository.
> <https://doi.org/10.24432/C54305>

This project analyses historical, anonymised process data for learning purposes.
It makes no claim about any current manufacturing line.
