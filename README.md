# Application of Reinforment Learning in the air quality monitoring

> **Note:** All the commands are based on a Unix based system such as _OSX_.
> For a different system look for similar commands for it.


## Setup

We are using Python version 3.11.9

```bash
$ python --version
Python 3.11.9
```

### Python virtual environment

**Create** a virtual environment:

```bash
$ python3 -m venv .venv
```

`.venv` is the name of the folder that would contain the virtual environment.

**Activate** the virtual environment:

```bash
$ source .venv/bin/activate
```

**Windows**
```bash
source .venv/Scripts/activate
```
### Requirements

```bash
(.venv) $ pip install -r requirements.txt
```

1. Fill in the values appropriately

## Run the queries

Open a **jupyterlab** instance

```bash
$ jupyter-lab
```

### Reading AQI value
```bash
$ ssh user@pi.address
$ cat /dev/ttyACM0
$ source ~/piicodev-env/bin/activate
$ rshell -p /dev/ttyACM0
$ ls /pyboard
$ cd /pyboard && edit main.py # modify the script

$ rshell -p /dev/ttyACM0

```
