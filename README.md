# X-Light
This is the offical repo of X-Light: Cross-City Traffic Signal Control Using Transformer on Transformer as Meta Multi-Agent Reinforcement Learner.


## How to use
**First**, you need to [install sumo](https://sumo.dlr.de/docs/Downloads.php) or install it from requirement.txt, and then you need to set SUMO_HOME in the environment variable. For example, if sumo is installed from requirement.txt, the env should be setted like:
```bash
export SUMO_HOME=/your python env path/lib/python3.6/site-packages/sumo
```
**Second**, export PYTHONPATH to the root directory of this folder. That is 
```bash
export PYTHONPATH=${PYTHONPATH}:/your own folder/root directory of this folder
```
**Third**, unzip scenarios' files:
```bash
cd onpolicy/envs/sumo_files_marl
unzip scenarios.zip
cd ../../../
```
**Training**:
```bash
python onpolicy/scripts/train/train_sumo.py
```
