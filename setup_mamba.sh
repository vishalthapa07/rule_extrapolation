#!/bin/bash
# Script to setup Mamba module for the project

echo "Setting up Mamba module..."

cd /home/lenevo/python/rule_extrapolation

# Try to initialize git submodule
echo "Initializing git submodule..."
git submodule update --init --recursive 2>&1

# Check if mamba directory has files
if [ -z "$(ls -A mamba/)" ]; then
    echo "Mamba submodule is empty. Trying alternative installation..."
    
    # Try installing mamba-ssm via pip
    echo "Attempting to install mamba-ssm via pip..."
    source venv/bin/activate
    pip install mamba-ssm 2>&1 || echo "Failed to install mamba-ssm via pip"
    
    # Alternatively, clone directly
    if [ -z "$(ls -A mamba/)" ]; then
        echo "Cloning mamba repository directly..."
        cd mamba
        git clone https://github.com/rpatrik96/mamba.py . 2>&1 || echo "Failed to clone mamba repository"
        cd ..
    fi
fi

# Check if mamba is now available
source venv/bin/activate
python3 -c "try:
    from mamba.mamba_lm import MambaLM
    print('Mamba module is now available!')
except ImportError as e:
    print(f'Mamba module not available: {e}')
    print('You may need to install mamba-ssm: pip install mamba-ssm')
    print('Or install the mamba submodule manually'
)" 2>&1

echo "Mamba setup complete!"

