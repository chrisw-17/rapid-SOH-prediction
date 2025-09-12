
# Rapid EIS-based SOH Estimation for Unknown LIBs

---

**This repository accompanies the paper:**

> **Fast state of health evaluation of lithium-ion batteries using EIS for second-life and recycling assessment**  
> *Authors: Christopher Wett, Jörg Lampe, Joey Danz, Thomas Seeger, Bugra Turan*  
> *Journal: [Journal Placeholder]*  
> *DOI: [DOI Placeholder]*

---

## Overview

This repository provides a complete workflow for rapid State of Health (SOH) estimation of lithium-ion batteries using Electrochemical Impedance Spectroscopy (EIS) and neural network models. The code and data enable reproducible analysis, model training, and evaluation as described in the associated publication.

## Features

- Modular Python implementation for SOH prediction
- Ready-to-use datasets from multiple sources (RH Köln, Stanford, Warwick) and thereby different LIBs
- Pretrained neural network models (FCNN, LSTM)
- Easy-to-adapt for new EIS datasets
- Open-source and ready for extension

## Repository Structure

```text
├── Neural_Network_SOH.py         # Main Python script for SOH prediction
├── Experimental_Data_RH/         # Singlesine EIS data and SOH values (RH Köln)
├── Multisine_EIS_RH/             # Multisine EIS data and SOH values (RH Köln)
├── Stanford/                     # Public EIS data (see reference below)
├── Warwick/                      # Public EIS data (see reference below)
├── Saved_NN_Models/              # Pretrained neural network models
├── Requirements.txt              # Required Python packages
```

## Quick Start

1. **Clone the repository:**
	```bash
	git clone https://github.com/[your-username]/eis-soh-sustain.git
	```
2. **Install dependencies:**
	```bash
	pip install -r Requirements.txt
	```
3. **Prepare your data:**
	- Place your EIS data and SOH values in the appropriate folders.
4. **Run the main script:**
	```bash
	python Neural_Network_SOH.py
	```

## Data Sources & Citation

If you use the code or data, please cite the associated paper above. For external datasets, cite:

- **Stanford:** https://doi.org/10.1016/j.dib.2022.107995
- **Warwick:** https://doi.org/10.1016/j.est.2022.106295

## Workflow Summary

1. Load and preprocess EIS data from multiple sources
2. Extract impedance features and SOH labels
3. Train and evaluate neural network models
4. Visualize results and export trained models

## Support & Contact

For questions, bug reports, or collaboration, please contact the authors via [contact placeholder] or open an issue on GitHub.

## License

MIT License
