
# [Clone repo with data submodule]
!#/bin/bash
 git clone --recursive https://github.com/Al-ek/shakespeare_sonnet_stylometry.git

# [Create python venv and install dependencies]
!#/bin/bash
 python3 -m venv venv
 source venv/bin/activate
 pip install requirments.txt
