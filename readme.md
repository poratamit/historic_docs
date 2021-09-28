# Historic Documents

## Disclaimer
The tool classifies the images, and registers Version 1 of the images, as Dr. Teicher instructed. \ 
The analysis is done only on type 1 of version 1. It needs to be extended to different types. 

## Installation

On Ubuntu: \
`apt install tesseract-ocr-deu`
`git clone https://github.com/poratamit/historic_docs` \
`cd historic_docs` \
`pip3 install -r requirements.txt`


## Usage
### pdftoppm
First, we need to convert the pdf into a directory of images. \
In this directory, the name of the image should the page number it was extracted from in the pdf. \
In order to do so, we use a tool called pdftoppm , which is pre-installed on Linux. \
`mkdir <path_to_output>` \
`pdftoppm -jpeg <path_to_pdf> <path_to_output_dir>/<img name format>` \
For example, \
`mkdir extracted ` \
`pdftoppm -jpeg G34.pdf extracted/G34` \  
It unpacks G34.pdf to a directory named called extracted, and the named of the files are G34-xxx.jpg, where xxx is the page number. 


### The tool
For help, run `python3 main.py -h`

#### Analysis and registration
`python3 main.py -s <source_dir> -d <dest_dir>` \
The source dir is the output directory of pdftoppm. \
After the tool finishes, the directory will contain classified and registered images, with the full analysis \ 
and the cropped images.
