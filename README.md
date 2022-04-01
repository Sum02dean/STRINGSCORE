# String Score Optimization
This repository contains code for the optimization of String protein-protein interaction prediction score. 

### Installation 
This requires a valid installation of [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [miniconda](https://docs.conda.io/en/latest/miniconda.html). Create the Python environment as described below:

```sh
cd configs
conda env create -f environment.yml 
conda activate string_score
```

### Data
The data folder should be copied within your src directory. The data folder contains everything required to run the model scripts.
```sh
cd src
cp -r /mnt/mnemo1/sum02dean/dean_mnt/projects/STRINGSCORE/src/data .
```

### Data Preparation
To change the default pre-processing arguments please modify the values in the run_pre_process.sh file.
```sh
cd src
bash run_pre_process.sh -o pre_processed_data/
```
### Running the model
Each model comes with a seperate bash file. Within each bash file there are sets of options to choose from. To see the definition of each option, run the corresponding python file name followed by --help. 
- run_xgboost.sh
- run_bambi.sh
- run_net.sh

Example of calling help for model args:
```sh
python xgboost_model.py
```

### Model Output
Outputs will be organised as such:
```
    models
    |_ _output_directory
    |   |_ _model_name
    |       |_ _(plots and main results)
    |       |_ _ensemble 
    |           |_ _ (ensemble model data)
```

### Hyper-parameter selection

Below are listed some useful parameters to use for initial hyper-parameter optimization, see the /models/ section for model specific documentation. Note that for Bambi - the majority of parameters are determined using [NUTS](http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf).

XGBoost:
| Parameter | Value |
| ------ | ------ | 
| n_samples |3|
| max_depth |15|
| eta |0.1|
| objective | 'binary:logistic'|
| alpha |0.1|
| lambda |0.01|
| subsample |0.9|
|colsample_bynode|0.2|

Bambi:
| Parameter | Value |
| ------ | ------ | 
| n_runs |1|
| n_chains |2|
| n_draws |1000|
|n_tune |3000|
|family |bernouli|

Neural Network (Pytorch):
| Parameter | Value |
| ------ | ------ | 
| n_runs |3|
| batch_size |50|
| epochs |100|
|hidden_size|200|
|learning_rate|0.001|

### Models

| Models | Resources |
| ------ | ------ | 
| XGBoost | [XGBoost Documentation](https://xgboost.readthedocs.io/en/stable/) |
| Bambi | [Bambi Documentation](https://bambinos.github.io/bambi/main/index.html) |
| Neural Net (Pytorch) | [Pytorch Documentation](https://pytorch.org/) |

### Potential conflicts
```py
"""Pandas Multiindex deprecation: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead."""
```
### Major dependencies
The conda environemnt provided should contain all of these requirements. If not you can find them at the following sources.
| Dependency | Intallation |
| ------ | ------ | 
| R for Python (second option) |[Anaconda](https://anaconda.org/r/r)|
| Perl for Pytthon |[Pypi](https://pypi.org/project/perl/)|
| Perl-Json module for Python |[Pypi](https://anaconda.org/bioconda/perl-json)|

### Pymc3 (Bambi)
This dependency causes the most problems. If the provided Python environement does not work, try to install this package in a new Python env (make sure this is first). Follow the instructions on [Github](https://github.com/pymc-devs/pymc/wiki/Installation-Guide-(Linux)) under sub-section "Theano-PyMC Installation".

```bash
git clone https://github.com/pymc-devs/pymc
cd pymc
git checkout v3
pip install -r requirements.txt
```

