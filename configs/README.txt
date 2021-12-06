To load environment: conda env create --name [envname] --file=environment.yml  
Don't foget to specify the --name as [envname] of your choice. The environment.yml file is found in this directory.

1. Go to src dir
2. Activate the python env
3. Copy the data: cp /mnt/mnemo1/sum02dean/dean_mnt/projects/STRINGSCORE/src/data [PATH], where [Path] is your project directory.
4. Configure settings in the run_xgboost.sh file
4. Run pipeline with bash run_xgboost.sh
 
2. Configure run_xgboost_model.sh file with desired parameters
3. Run 'bash run_xgboost.sh'
4. Done
