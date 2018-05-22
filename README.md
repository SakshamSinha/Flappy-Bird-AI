Installation
============

All the following command suppose you are using a virtualenv (https://virtualenv.pypa.io/en/stable/)

Requirements:

Clone the repo:
```
 sudo apt install python3-tk
git clone git@gitlab.scss.tcd.ie:chronc/ai-project-5.git --recursive
```

Install the dependencies:
```
cd ai-project-5
pip install -r requirements.txt --upgrade
or
sudo -H pip3 install -r requirements.txt --upgrade
```

If you have a GPU, you should install tensorflow-gpu:
```
pip install tensorflow-gpu
```

Install PLE:
```
cd PLE
pip install -e .
cd ..
```
Install deer:
```
cd deer
pip install -e .
cd ..
```

Tests:
======

Test deep-q:
```
cd deep-q
python run_PLE.py
```

