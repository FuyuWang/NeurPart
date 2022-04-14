# NeurPart

NeurPart is an automatic and adaptive resource partitioner 
for spatial accelerators with meta reinforcement learning.
This repository contains the source code for NeurPart.

### Setup ###
* Download the NeurPart source code 
```
git clone https://github.com/FuyuWang/NeurPart.git
```
* Create virtual environment through anaconda
```
conda create --name NeurPartEnv python=3.8
conda activate NeurPartEnv
```
* Install packages
   
```
pip install -r requirements.txt
```

* Install MAESTRO
```
python build.py
```

### Run NeurPart ###

* Run AutoPart on cloud and edge platforms
```
./run_auto_cloud.sh  ./run_auto_edge.sh
```

* Run AdaptPart on cloud and edge platforms
```
./run_adapt_cloud.sh  ./run_adapt_edge.sh
```
