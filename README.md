# solve-sudoku-computer-vision
solve sudoku puzzles with computer vision
## Introduction 

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
Cài đặt torch và torchvision CPU
```bash
    #CPU
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
hoặc CUDA
```bash
    #CUDA
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
```

## Run

Download trained model lưu trong thư mục `weights`
```bash
    # Download model weights
    pip install -U --no-cache-dir gdown --pre
    gdown 1-Nx4iIv2RBlQ6uEhTo-jLHPg9XPer6w3
```

Chạy chương trình
```bash
    # Run file program.py
    python program.py
```