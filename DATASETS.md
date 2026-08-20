# Datasets and data acquisition

The repository has two experiment tracks with deliberately different purposes.

## 1. UCI Human Activity Recognition Using Smartphones — primary showcase

This is the **real-data GitHub showcase track**. It represents an edge-relevant sensor workload rather than a generic image-classification toy.

- Source: UCI Machine Learning Repository, dataset 240
- DOI: `10.24432/C54S4K`
- License: **Creative Commons Attribution 4.0 International (CC BY 4.0)**
- Subjects: 30 volunteers
- Activities: 6
- Windows: 10,299
- Features: 561 time/frequency-domain features derived from smartphone accelerometer and gyroscope signals
- Upstream split: volunteer-disjoint train/test split

Official dataset page:

`https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones`

Citation:

> Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013). Human Activity Recognition Using Smartphones [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54S4K

### Acquire it

The dataset is **not committed to this repository**. Acquisition is explicit and records the exact observed upstream archive SHA-256 for the run:

```bash
edgeai dataset fetch uci-har --destination data/uci-har
```

The downloader:

- fetches only from the declared HTTPS UCI source;
- enforces a download-size ceiling;
- rejects path traversal, symlinks, duplicate members, and oversized expansion;
- extracts only the files required by the experiment;
- writes `data/uci-har/provenance.json` with source URL, DOI, license, retrieval time, byte count, and observed SHA-256.

The checksum in `provenance.json` is an **observed acquisition checksum**, not represented as an upstream-published checksum.

### Run the experiment

```bash
edgeai showcase \
  --dataset uci-har \
  --data-dir data/uci-har \
  --output-dir results/showcase/uci-har
```

or:

```bash
make showcase-real
```

## 2. scikit-learn digits — offline smoke track

The compact digits dataset remains useful for CI and completely offline development. It is **not the headline showcase workload**.

```bash
edgeai showcase --dataset digits --output-dir results/showcase/digits
```

This proves that the experiment harness, signing/deployment lifecycle, drift reporting, and report generation work with no network access.

## Data policy

Raw public datasets are ignored by Git. Generated provenance and experiment outputs may be retained as release evidence. No credentials, private telemetry, or personally identifying data are required by the showcase.
