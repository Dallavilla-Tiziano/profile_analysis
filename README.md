# Spatial differences in the molecular organisation of tumours along the colon are linked to interactions within the tumour ecosystem
## Aim of the workflow
This workflow allow to test the performance of linear, polynomial and sigmoidal model in describing the spatial dependency of different molecular, environmental and clinical properties in an organ model. The detailed functioning, along with extensive examples, are discussed in [CITATION].

## Reproducing the results of [CITATION]
If you just need to reproduce the results in Dallavilla et al then the easiest way is through the dedicated docker image.
In order to run the image you need to install docker following the instructions available at [Get Docker](https://docs.docker.com/get-docker/)).
You then need to follow these steps:
1. Pull the docker image available [here](https://hub.docker.com/repository/docker/tizianodallavilla/profiler/general) on Docker Hub from the terminal using the command:
```bash
docker pull tizianodallavilla/profiler
```
2. Launch the image from the terminal by using the following command:
```bash
docker run -it -p 8889:8889 tizianodallavilla/profiler start.sh jupyter lab /home/jovyan/git/profiler/notebooks/paper_analysis/profile_analysis_colon.ipynb --port 8889 --NotebookApp.token=''
```
3. Open your favourite internet browser at [http://localhost:8889](http://localhost:8889)
4. From the left menu open the notebook 'profile_analysis_colon.ipynb'
5. Run the whole notebook by selecting 'Run > Run all cells' from the top menu
These steps allow to generate the data presented in [CITATION]. Please note that data coming from model fitting have already been generated in order to save time (When you run a step for which data have already been computed the message 'This step has already been executed...loading results...' will be printed to screen). If you want to generate results starting from raw data you need to [WILL BE AVAILABLE SOON]. Please keep in mind that this will require a considerable amount of time. On a machine with 20 cores and 20GB of memory, execution is estimated to be around 48 hours for all the datasets.
