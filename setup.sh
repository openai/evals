#!/bin/bash
### Maintainer: @panayao
### Tested on macOS Monterey 12.6.3

### This script will setup the environment variables and aliases for the project
### Make sure to run with 'source setup.sh' since we are doing export of environment variables and aliases

#######################################################
### This is to setup the environment variables and aliases for the project. 
## Just copy '.env.example' to '.env' and fill in the values, and run the export line below. 
## The rest of the script is optional, but helpful
export $(grep -v '^#' .env | xargs)
## put this in your .bashrc or .zshrc file to make it always accessible
alias setdotenv=`export $(grep -v '^#' .env | xargs)`
#######################################################

## unset all environment variables in the same file, still ignoring commented lines 
# unset $(grep -v '^#' .env | sed -E 's/(.*)=.*/\1/' | xargs)
## put this in your .bashrc or .zshrc file to make it always accessible
alias unsetdotenv=`unset $(grep -v '^#' .env | sed -E 's/(.*)=.*/\1/' | xargs)`

# perform a git lfs pull to get the latest model files
git lfs fetch --all
git lfs pull

### Create Conda env for the project
### This Conda env has been tailored for macOS Monterey 12.6.3
createFreshCondaEnv() {
    conda deactivate
    rm -rf ~/${1}/envs/oaievals
    CONDA_SUBDIR=osx-arm64 conda create -n oaievals python=3.10 -c conda-forge -y
    conda activate oaievals
    conda env config vars set CONDA_SUBDIR=osx-arm64
    echo "set CONDA_SUBDIR=osx-arm64 in 'oaievals' conda env ..."
    conda deactivate
    conda activate oaievals
    echo "Finished installing conda 'oaievals' env !!!"
    pip install --no-cache-dir -r requirements.txt --upgrade
}

### NOTE: Choose one of the following ways to install dependencies

## install dependencies (normal way if you don't want to use Conda)
pip install -e .
## install Conda virtual environment for the project (optional)
# createFreshCondaEnv "mambaforge"

# uncomment line below for testing
# oaieval gpt-3.5-turbo test-match

## Create aliases
## Put this in your .bashrc or .zshrc file to make it always accessible
# for no Conda
alias prepoaievals=`export $(grep -v '^#' .env | xargs) && git pull && git lfs fetch --all && git lfs pull && pip install -e .`
# for Conda
# alias prepoaievals=`export $(grep -v '^#' .env | xargs) && git pull && git lfs fetch --all && git lfs pull && conda activate oaievals && pip install --no-cache-dir -r requirements.txt --upgrade`
