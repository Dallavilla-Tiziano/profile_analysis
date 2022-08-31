# Associations between molecular and environmental changes along the proximal-to-distal axis of the colon
## Aim of the workflow
This workflow allow to test the performance of linear, polynomial and sigmoidal model in describing the spatial dependency of different molecular, environmental and clinical properties in an organ. The detailed functioning along with extensive examples are discussed in [CITATION].

## Set up
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
### Clone the repository
First you need to download this repository. In the terminal navigate to the folder were you want to store the project and paste the following command:

```bash
git clone https://github.com/Dallavilla-Tiziano/profile_analysis.git
```

## Summary of the workflow
The workflow consist of 4 steps:
1. Define project settings
2. Create Project folder and sub-folders
3. Manually add input and clinical data
4. Run the analysis

The easiest way to understand and run the workflow is to run the example described in the section ```Run the example```.
#### 1. Define project settings
In order to run the analysis you need to define a series of settings. These settings need to be contained in a file called ```SETTINGS.ini``` that needs to be placed in the folder were you want to run the analysis. An example of a setting file is available in */profile_analysis/example/SETTINGS.ini*. The settings file is divided in N sections:
- **ORGAN**: this section contains the parameters that define the organ that we want to simulate.
	- **sections**: python dictionary that define which sections constitute the organ under analysis. Each dictionary key is the name of a section as it appear in clinical data and its associated value is a list that contains the names of the sections that are going to be grouped under that section, eg: ```{'Section 1':['Section 1']}```. The algorithm can work with whatever number of samples per section, but ideally you want to have more than 5 samples in each section to avoid problems during calculation of median values per section. Therefore if one section of your organ have 5 or less samples you can decide to group it into an adjacent section. You can see this happening in the setting file located */profile_analysis/example/SETTINGS.ini*. Since in our work we only had 5 samples in the splenic flexure section of the colon we grouped those samples into the descending colon section by defining it as ```'Descending colon': ['Descending colon', 'Splenic flexure of colon']``` and removing the splenic flexure from the key values of the dictionary.
	- **sections_distance_from_reference**: python list that define the order of the sections. The list should contain the distance of each section from a reference point, list element number 0 correspond to key 0 in **section** and so on. The number of element in the list needs to match the number of keys defined in **section**. If this data is not available you can assign to each section an integer number that define the order of the sections.
	- **plot_names**: Python list that contains the names of the sections that will be used in plots. Sometimes clinical data contains sections names that are abbreviated or non standard names. Here you can rename each section, the strings in this list will be used as names for each section in plots. The number of element in the list needs to match the number of keys defined in **section**.
- **ANALYSIS_SETTINGS**
	- **sample_0_threshold**: float from 0 to 1. This threshold indicate the proportion of 0 value above which a feature is discarded from the analysis. Eg: If we fix this value to 0.3 every gene in a expression quantification experiment that as 0 counts in more than 30% of the samples will be discarded from the analysis. Set this value to 1 to avoid this filter step, or set it to 0 to keep only feature with non-zero measurements in all of the samples.
	- **polynomial_degree_to_test**: Integer. This number indicates the maximum polynomial coefficient to test. All the coefficients from 1 to  **polynomial_degree_to_test** will be used in the analysis
	- **random_permutation_n**: Integer. Number of random permutations to perform during bootstrapping.
	- **threshold_area**: float from 0 to 1. After bootstrapping the distribution of random permutation scores is estimated for each model tested. These distributions are then compared with the distribution of observable scores. All the observed features with a score above the **threshold_area** percentile are selected to proceed to the next steps of the analysis.
	- **medians_nan**: ```drop``` or ```keep```. Depending on the number of samples in a section and on the value of **sample_0_threshold** na value can appear in the median value of a section. you can decide to keep them or to drop them.
	- **index_col**: The name of the index colon in input data.
	- **data_type**: ```numeric``` or ```binary```. If input data are continuous use ```numeric```, if they are  categorical use ```binary```.
- **MISC**
	- **cores**: Number of cores to use in the analysis.
	- **plot_font_size**: Size of the font in plots.
	- **set_seed**: ```True``` or ```False```. If you are interested in your analysis to be repeatable
	- **random_seed**: random seed.




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
from jupyter menu open */example/example_profiling.ipynb* and then run all the cells.
