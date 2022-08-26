# Associations between molecular and environmental changes along the proximal-to-distal axis of the colon

## Set up
### Clone the repository
First you need to download this repository. In the terminal navigate to the folder were you want to store the project and paste the following command:

```bash
git clone https://github.com/Dallavilla-Tiziano/profile_analysis.git
```
### Requirements
In order to run all the steps of the analysis you need python (>3.7.0) and the following python libraries:
- os
- random
- math
- glob
- pickle
- configparser
- shutil
- warnings
- ast
- pandas
- joblib
- sklearn
- numpy
- matplotlib
- scipy
- supervenn
- sympy
- tqdm
- jupyter lab

You can install the packages manually or use conda (>4.6) (RECOMMENDED). If you decide to use conda please run the following script from the git project folder which create a conda environment and install all the packages that are not available in the conda repository:
```bash
./conda/conda_setup.sh
```
## Run the example
This example run the profile analysis on 1000 genes from COAD and READ TCGA samples.  
If you decided to use conda activate the environment with:
```bash
conda activate pa_env
```
The example run in jupyter lab therefore the next step is to launch jupyter. Navigate to the git folder of this project and execute the following command:
```bash
jupyter lab .
```
from jupyter menu open <i>/example/example_profiling.ipynb</i> and then run all the cells.
