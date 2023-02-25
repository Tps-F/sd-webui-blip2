# SD-webui-blip2
<img alt="LICENSE" src="https://img.shields.io/badge/license-MIT-blue.svg?maxAge=43200">

## Introduction
sd-webui-blip2 is a stable diffusion extension that generates image captions with blip2
Using that caption as a prompt may help you get closer to your ideal picture.


## Installation
### Extensions
Open "Extensions" -> "Install from URL" paste the link below

    https://github.com/Tps-F/sd-webui-blip2.git
       
### LAVIS
If you receive the message "Can't install salesforce-lavis" please follow the steps below.

**C++ build environment is required**

    git clone https://github.com/salesforce/LAVIS.git
    cd LAVIS
    pip install -e .


## Usage

First select a model, If that model does not exist, the download will begin. Please be patient...
Next, select the image for which you want to choose a caption and press "Generate Caption"!

Information about the parameters is as follows
| Name | Description|
----|----
| Beam Search | Generates a single prompt |
| Nucleus Sampling | Generates three prompts |
| Length Penalty | Answer length |
| Repeat Penalty | arger value prevents repetition |
| Temperature | higher temperature => more likely to sample low probability tokens |

## Other Information

If there are any other features you need, please report them in an issue!

## License

[MIT](https://choosealicense.com/licenses/mit/)
