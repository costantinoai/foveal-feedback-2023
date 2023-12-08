---

### 🌟 **Overview**:
This repository hosts the code for the statistical analyses of functional magnetic resonance imaging (fMRI) data in the manuscript [PLACEHOLDER]. 

### 📁 **Directory Structure**:
More information about the scripts can be found in the docstrings.

├── data/  
│ └── BIDS : Copy here the BIDS folder from https://osf.io/h95a2/  
├── scripts/  
│ ├── 1_GLM.m : Perform GLM on preprocessed data.  
│ ├── 2_run_mvpa.py : Runs MVPA analysis and show plot.  
│ ├── 3_polynomial_regression.py : Runs plynomial regression on MVPA accuracy.  
│ ├── 4_perform_PPI.py : Performs PPI analysis.  
│ └── 5_inspect_results.R : Inspect results.   

### **📚 Setup Instructions**:
To set up the dataset for this repository, follow the instructions below:

1. **Clone the repository**:

   ```bash
   git clone costantinoai/foveal-feedback-2023
   cd foveal-feedback-2023
   ```

2. **Download the dataset** from [this link](https://osf.io/h95a2/).
   
3. **Ensure that the `BIDS` folder is in the `./data/` directory**.

4. **Run the analysis**.

You can now run the scripts in `./scripts` in order. The dataset includes the data before and after the GLM, so the GLM estimation can potentially be skipped.







