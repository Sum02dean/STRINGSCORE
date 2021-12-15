To load environment: $: conda env create --name [envname] --file=environment.yml  
Don't foget to specify the --name as [envname] of your choice. The environment.yml file is found in this directory.

1. Go to src dir
2. Activate the python env
3. Copy the data: $: cp /mnt/mnemo1/sum02dean/dean_mnt/projects/STRINGSCORE/src/data [PATH], where [PATH] is your project directory.
4. Configure settings in the run_xgboost.sh file
5. Run pipeline with $: bash run_xgboost.sh
 
