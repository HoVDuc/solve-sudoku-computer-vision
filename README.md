# solve-sudoku-computer-vision
solve sudoku puzzles with computer vision

## Installation

### Requirements:
- Python >= 3.7
- Pytorch 

### Clone repo
```bash
    # clone repo
    git clone https://github.com/HoVDuc/solve-sudoku-computer-vision.git
    cd solve-sudoku-computer-vision
```

### Tạo môi trường
```bash
    # Tạo môi trường với conda
    conda create -n sudoku_env
    conda activate sudoku_env
```

### Cài đặt các gói
```bash
    # Cài đặt các gói thư viện cần thiết
    python -m pip install -r requirements.txt
```
```bash
    # Cài Pytorch và torchvision nếu sử dụng CPU
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
```bash
    # Cài Pytorch và torchvision nếu sử dụng cuda
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
```

## Run

```bash
    # Download model weights
    mkdir weights
    cd weights
    pip install -U --no-cache-dir gdown --pre
    gdown 1-Nx4iIv2RBlQ6uEhTo-jLHPg9XPer6w3
    cd ..
```

```bash
    # Run file program.py
    python program.py
```