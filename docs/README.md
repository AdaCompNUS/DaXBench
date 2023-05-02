# DaxBench documentation

## Setup

Install `enchant` C library following [the official
instruction](https://pyenchant.github.io/pyenchant/install.html#installing-the-enchant-c-library).

Install the python dependencies:

``` sh
pip install -r requirements.txt
```

## Build

Run

``` sh
make html
```

The built doc can be found at `build/html/index.html`.

### Apple Silicon enchant library path

You might need to specify the environment variable `PYENCHANT_LIBRARY_PATH`

``` sh
# Use your installed version at /opt/homebrew/lib/
PYENCHANT_LIBRARY_PATH=/opt/homebrew/lib/libenchant-2.2.dylib make html
```
