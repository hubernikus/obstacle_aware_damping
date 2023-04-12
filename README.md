# Obstacle Aware Damping

## Setup 

### Create Environment (Optional)
Choose your favorite python-environment. I recommend to use [virtual environment venv](https://docs.python.org/3/library/venv.html).
Setup virtual environment (use whatever compatible environment manager that you have with Python >3.10).

``` bash
python3.11 -m venv .venv
```

Activate your environment
``` sh
source .venv/bin/activate
```

## Installation

Install requirements and current library (editable):
``` bash
pip install -r requirements.txt
pip install -e .
```

### Install the required passive_control.

Mathematical tools (vartools)
``` bash
pip install git+https://github.com/hubernikus/various_tools.git@main
```

Dynamic Obstacle Avoidance
``` bash
pip install git+https://github.com/hubernikus/dynamic_obstacle_avoidance.git@main
```



python script/robot_with_control_command.py

# Issues
`python setup.py egg_info did not run successfully.` -> make sure you have the most recent version of setup-tools:
``` shell
pip install --upgrade setuptools
```
