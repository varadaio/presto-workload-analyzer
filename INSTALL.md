# Install the Presto Workload Analyzer

To install the Presto Workload Analyzer in Python virtual environments, use the commands in section 1. 

For Docker environments, see section 2, after the video below.

## 1. Virtual environment


### Install (using Python 3.6+)
```bash
python3 -m venv .env
source .env/bin/activate && pip install -U pip wheel
pip install .
```

See the following screencast for installation example:


[![asciicast](https://asciinema.org/a/v30C629gK6zq4YkCxMBrPZmA3.svg)](https://asciinema.org/a/v30C629gK6zq4YkCxMBrPZmA3)


## 2. Docker

### Install (on Ubuntu 18.04 @ EC2)
```bash
$ sudo apt update && sudo apt install docker.io # install Docker CE edition
$ sudo usermod -aG docker $USER && logout   	# to allow the user to run Docker without sudo
$ docker run hello-world          				# make sure Docker works

Hello from Docker!
This message shows that your installation appears to be working correctly.
<snip>
```

### Build
```bash
$ cd $ANALYZER_REPOSITORY/
$ docker build -t analyzer:latest .
# lot's of output, takes ~1 minute

$ docker images
REPOSITORY          TAG                 IMAGE ID            CREATED              SIZE
analyzer            latest              8d36ac887157        About a minute ago   1.35GB
python              3.7                 cda8c7e31f89        44 hours ago         919MB
hello-world         latest              bf756fb1ae65        3 months ago         13.3kB
```
