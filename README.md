# word-search-robot
Software for automatically solving word searches. 
Written to run on Nvidia Jetson Nano. 


## Installation
### NVIDIA Jetson Nano
1. `./build_opencv.sh 4.1.1`
2. `sudo apt-get install swig`
3. `pip3 install -r requirements-jetson.txt`

### Other platforms
1. `pip3 install -r requirements.txt`

## Usage
`./pipeline.py`

### Input from a file
`./pipeline.py -i path/to/file`

### Run with UART output
`./pipeline.py -e`
