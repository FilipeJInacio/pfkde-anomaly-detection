# Phase-Folded KDE Anomaly Detection for Satellite Telemetry
Code for "Phase-Folded KDE Anomaly Detection for Satellite Telemetry" paper

The codebase includes:

* Data processing utilities
* Phase computation tools
* The implementation of the PFKDE method
* Scripts used to reproduce the figures and ROC analysis presented in the paper

---

# Repository Structure

## 1. `process_raw_data.ipynb`

This notebook is used to process the raw telemetry dataset.

To reproduce the full pipeline from scratch:

* The dataset must be obtained from **SatNOGS**
  [https://db.satnogs.org/satellite/GPLA-1467-3987-8284-6460](https://db.satnogs.org/satellite/GPLA-1467-3987-8284-6460)
* Alternatively, contact the author to request the processed data

This notebook:

* Cleans and structures the raw CSV data
* Aligns it with computed phase information
* Produces the dataset used for PFKDE evaluation

---

## 2. `Phase_Fold.py`

This script computes the **phase of each telemetry point**.

It relies on:

* ESA's **GODOT** library
* TLE data

To reproduce the phase computation:

1. Download the relevant TLE data from
   [https://www.space-track.org](https://www.space-track.org)

2. Use the GODOT template files available in the `data/` folder

3. Run the script:

```bash
python Phase_Fold.py
```

This produces a CSV file containing the computed eclipse times.

Because this process requires ESA's GODOT, it may be cumbersome to reproduce.

For convenience, the eclipse file used in the paper is already provided:

```
data/eclipses_40014.csv
```

Therefore, reproducing the published results **does not require re-running the phase computation**.

---

## 3. PFKDE Method Implementation

### `PFKDE.py`

Defines the Phase-Folded KDE anomaly detection method.

---

### `PFKDE_run.py`

Used to generate:

* The images shown in the paper
* Example anomaly detection outputs

This script runs PFKDE for a **specific anomaly threshold**.

---

### `PFKDE_loop.py`

Used to compute the threshold sweep used for the ROC analysis.

---

# Requirements

A `requirements.txt` file is provided.

Note:

* The requirements file includes **all packages**, including those needed for phase computation.
* GODOT can be complex to install.
* These packages are **not required** to reproduce the published results, since the processed eclipse file is already provided.

Aside from space-related packages, the remaining packages are standard ML Python libraries.

---

# How to Use the Code

## Step 1 — (Optional) Recompute Phase Information

Requirements:

* GODOT
  Installation guide:
  [https://godot.io.esa.int/docs/install/pip_install.html](https://godot.io.esa.int/docs/install/pip_install.html)
* SGP4
* astropy
* ruamel.yaml
* TLE data from space-track.org

Then run:

```bash
python Phase_Fold.py
```

This generates eclipse time data.

If you only want to reproduce the paper results, you can skip this step and use:

```
data/eclipses_40014.csv
```

---

## Step 2 — Process Raw Telemetry

Use:

```
process_raw_data.ipynb
```

Inputs:

* Raw CSV from SatNOGS
* Eclipse time CSV

Output:

* `40014_dataset.mat`

---

## Step 3 — Run PFKDE

After obtaining `40014_dataset.mat`:

To generate figures:

```bash
python PFKDE_run.py
```

To compute ROC analysis:

```bash
python PFKDE_loop.py
```

---

# Citation

If you use this code, please cite the associated paper.

---
