# Agrometeorological Leaf Wetness Prediction

 [![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

[![IEEE Paper](https://img.shields.io/badge/DOI-10.1109%2FTAFE.2026.3711283-blue)](https://doi.org/10.1109/TAFE.2026.3711283)

This repository provides pre-trained Machine Learning models exported to **ONNX (Open Neural Network Exchange)** format for predicting leaf wetness status (**Dry / Wet**) using only agrometeorological parameters.

Separate models are provided for the **top side** and **bottom side** of the leaf to account for microclimatic variations across the canopy structure.

## Features

All models expect exactly **8 input parameters** passed in the following order:

| Index | Feature Name | Parameter | Units / Format |
|:---:|---|---|---|
| `0` | `soil_vwc` | Volumetric Water Content | % |
| `1` | `soil_temperature` | Soil Temperature | °C |
| `2` | `temperature` | Air Temperature | °C |
| `3` | `rh` | Relative Humidity | % |
| `4` | `rain` | Rainfall / Precipitation | mm |
| `5` | `wind_speed` | Wind Speed | m/s |
| `6` | `doy` | Day of Year | Integer (1–366) |
| `7` | `hour` | Hour of Day | Integer (0–23) |

### Output Target Format
* **`0` = Dry**
* **`1` = Wet**

## Repository Layout

```text
├── models/
│   ├── rf_lw_top.onnx         # Random Forest - Top Leaf Surface
│   └── rf_lw_bottom.onnx      # Random Forest - Bottom Leaf Surface
├── scripts/
│   └── predict.py             # Standalone ONNX inference CLI script
├── sample_data/
│   └── sample_input.csv       # Example CSV input for quick testing
├── LICENSE
├── requirements.txt           # Minimal dependencies (onnxruntime, pandas, numpy)
└── README.md
```

## Citation & Data Access

If you use these models or dataset in your research, please cite both the original publication and the dataset hosted on Mendeley Data.

**Reference Paper:**
```
@article{colucci2026machine,
  title={Machine Learning Methods for Leaf Wetness Prediction Using Agrometeorological Data},
  author={Colucci, Giovanni Paolo and Filipescu, Elena and Scatozza, Fabio and Trinchero, Daniele},
  journal={IEEE Transactions on AgriFood Electronics},
  year={2026},
  publisher={IEEE}
}
```

**Dataset**
```
Colucci, Giovanni Paolo; Filipescu, Elena; Scatozza, Fabio; Trinchero, Daniele (2025),
“Leaf Wetness and Agrometeorological Dataset”,
Mendeley Data, V1, doi: 10.17632/sd4b8vpvyb.1
```

**Note on Training Code:** All training configurations, hyperparameter setups, and validation metrics are detailed in the published paper.

