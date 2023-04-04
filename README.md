# SD-webui-blip2
<img alt="LICENSE" src="https://img.shields.io/badge/license-MIT-blue.svg?maxAge=43200">


## Introduction
sd-webui-blip2 is a stable diffusion extension that generates image captions with blip2
Using that caption as a prompt may help you get closer to your ideal picture.

<!-- ![image](https://user-images.githubusercontent.com/63702646/221370369-1e418ede-17b2-47ad-adf4-36f2e0f44f97.png) -->
![image](https://user-images.githubusercontent.com/63702646/229653904-ca294840-fe5a-47f4-ae7f-3cd23bc78568.png)
![image](https://user-images.githubusercontent.com/63702646/229654122-06f1c5ed-6dcc-408b-bbb8-ae4dc513cce2.png)

## Installation
### Extensions
Open "Extensions" -> "Install from URL" paste the link below

    https://github.com/Tps-F/sd-webui-blip2.git

[On Mac, sometimes need action](#fix-issue)
       
### LAVIS
If you receive the message "Can't install salesforce-lavis" please follow the steps below.

Windows: Open powershell on your-stable-diffusion-webui location and type

    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force 
    ./venv/scripts/activate
    pip install salesforce-lavis
    
Mac: Open terminal on your stabl-diffusion-webui location and type

    source venv/bin/activate
    pip install salesforce-lavis

Build from source

[C++ build environment is required](#install-build-tools)

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

### Install Build Tools
Windows

    winget install Microsoft.VisualStudio.2022.BuildTools
    
Linux

    sudo apt-get install cmake
or

    pacman -Syu cmake
    
Mac(has some issue)
    
    xcode-select --install
    brew install cmake

### Fix Issue

> ImportError: cannot import name 'ALL_LAYERNORM_LAYERS' from 'transformers.pytorch_utils' (/Users/ftps/stable-diffusion-webui/venv/lib/python3.10/site-packages/transformers/pytorch_utils.py)

Open stable-diffusion-webui/venv/lib/python3.10/site-packages/transformers/pytorch_utils.py and add line

    ALL_LAYERNORM_LAYERS = [nn.LayerNorm]

Before this line

    logger = logging.get_logger(__name__)

> PermissionError: [Errno 13] Permission denied: '/Users/ftps/.cache/torch/hub/checkpoints

open Terminal and type

    chmod +x .cache


## License

[MIT](https://choosealicense.com/licenses/mit/)
