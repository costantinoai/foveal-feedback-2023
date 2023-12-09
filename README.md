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

3. **Extract the `BIDS` dataset**:

   - **For Linux and macOS**:
     1. Open a terminal.
     2. Navigate to the directory where you downloaded the dataset.
     3. First, combine the parts into a single ZIP file using the `cat` command:
        ```bash
        cat BIDS.zip.* > BIDS_combined.zip
        ```
     4. Next, extract the combined ZIP file:
        ```bash
        unzip BIDS_combined.zip -d ./data
        ```
     5. This will extract the `BIDS` folder into `./data`.

   - **For Windows**:
     1. Navigate to the folder where you downloaded the dataset using File Explorer.
     2. Select all the parts of the dataset (`BIDS.zip.001`, `BIDS.zip.002`, etc.).
     3. Right-click and choose `Extract All`.
     4. Choose the `.\data` folder within your cloned repository as the destination for extraction.

   Ensure that all parts of the dataset are in the same directory before extraction.

4. **Ensure that the `BIDS` folder is in the `./data/` directory**.
   
   After extraction, check that the `BIDS` folder is located within `./data/` in your cloned repository. If it's not, move it there.

5. **Run the analysis**.

   You can now run the scripts in `./scripts` in order. The dataset includes the data before and after the GLM, so the GLM estimation can potentially be skipped.
