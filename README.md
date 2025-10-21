# How to clone?

### Github warning !!

```bash
# Please donot push anthing or merge code in develop and main branch, create your own separate branch and push code in it
```

### Cloning project

```bash
git clone https://github.com/vishalthapa07/rule_extrapolation.git
```

- Setup virtual environment

```bash
# install venv
sudo apt install python3-venv

# creating virtual env with name venv
python3 -m venv venv

# activate virtual env
source venv/bin/activate
```

- Run basic commands

```bash
pip install -e .
pip install -r requirements.txt
pip install --requirement tests/requirements.txt --quiet
pre-commit install
```

- To run weight and bias

```bash
pip install

# shows username
wandb login

# which yaml file you need to run, for eg- adversarial.yaml, replace it with your file name
wandb sweep sweeps/adversarial.yaml
# output of above line:
# wandb: Creating sweep with ID: xmghavqr
# wandb: View sweep at: https://wandb.ai/rule-extrapolation-learning/rule_extrapolation/sweeps/xmghavqr
# wandb: Run sweep agent with: wandb agent rule-extrapolation-learning/rule_extrapolation/xmghavqr

# To run in terminal use above wandb: Run sweep agent with:line
wandb agent rule-extrapolation-learning/rule_extrapolation/xmghavqr

```

OR

<div align="center">    
 
# Rule Extrapolation in Large Language Models+

[//]: # "[![Paper](http://img.shields.io/badge/arxiv-cs.LG:2311.18048-B31B1B.svg)](https://arxiv.org/abs/2311.18048)"
[//]: # "[![Conference](http://img.shields.io/badge/CI4TS@UAI-2023.svg)](https://sites.google.com/view/ci4ts2023/accepted-papers?authuser=0)"

![CI testing](https://github.com/meszarosanna/rule_extrapolation/workflows/CI%20testing/badge.svg?branch=main&event=push)

</div>
 
## Description

## How to run

### Installing dependencies

```bash
# clone the repo with submodules
git clone --recurse-submodules https://github.com/meszarosanna/rule_extrapolation


# install the package
cd rule_extrapolation
pip install -e .
pip install -r requirements.txt



# install requirements for tests
pip install --requirement tests/requirements.txt --quiet

# install pre-commit hooks (only necessary for development)
pre-commit install
```

### Weights and Biases sweep

```bash
# login to weights and biases
wandb login

# create sweep [spits out <sweepId>]
wandb sweep sweeps/<configFile>.yaml

# run sweep
./scripts/sweeps <sweepId>
```
to run without GPU 
python rule_extrapolation/cli.py fit --config configs/config.yaml --trainer.accelerator=cpu

## Citation

```

@inproceedings{

}

```
