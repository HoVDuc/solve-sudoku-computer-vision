# solve-sudoku-computer-vision
solve sudoku puzzles with computer vision
## Introduction 

Ứng dụng thị giác máy tính và học máy đã trở thành một lĩnh vực đầy triển vọng trong nhiều ứng dụng thực tế. Giới thiệu một ứng dụng thị giác máy tính và học máy để giải quyết bài toán Sudoku. Sudoku là một trò chơi ô số học với quy tắc đơn giản là điền các số từ 1 đến 9 vào một lưới 9x9 sao cho mỗi hàng, mỗi cột và mỗi ô vuông 3x3 con đều chứa đủ các số từ 1 đến 9 mà không bị trùng lặp.

Ứng dụng này sử dụng kỹ thuật thị giác máy tính để nhận diện và trích xuất dữ liệu từ hình ảnh Sudoku được chụp từ thiết bị di động hoặc máy tính cá nhân. Sau đó, dữ liệu này được đưa vào một thuật toán học máy để giải quyết bài toán Sudoku. Thuật toán học máy sử dụng phương pháp tìm kiếm và giải quyết đệ quy để tìm ra lời giải đúng cho bài toán Sudoku.

Kết quả thực nghiệm trên các bộ dữ liệu thực tế đã cho thấy ứng dụng này có khả năng giải quyết bài toán Sudoku với độ chính xác cao và thời gian giải quyết nhanh.

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
    conda create -n sudoku_env python=3.8
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
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu116
```

## Run

### Download trained model 
Tải trained mode rồi lưu vào trong thư mục `solve-sudoku-computer-vision/weights/`
```bash 
    # Download trained model
    gdown 1-Nx4iIv2RBlQ6uEhTo-jLHPg9XPer6w3
```

Nếu khi tải bị Permission denied thì chạy câu lệnh sau rồi chạy lại câu lệnh trên
```bash
    pip install -U --no-cache-dir gdown --pre
```

### Chạy chương trình
```bash
    # Run file program.py
    python program.py
```

Giao diện chương trình

![alt](data/docs/giao-dien-chuong-trinh.png)

Chức năng bao gồm:
- Select image: Chọn ảnh
- Run: Trích xuất câu đố
- Test: Giải câu đố
- Back: Quay trở lại giao diện trước
- Save: Lưu lại ảnh đã được giải
- Reset: Reset lại chương trình

### Demo
Từ giao diện chính chọn ```Select image``` sau đó chọn ```Run```

![alt](data/docs/giao-dien-chuong-trinh-run.png)

Nếu khi ```Run``` câu đố có bị sai có thể sửa lại bằng cách chọn vào ô có số bị sai và sửa lại số. Và nhấn ```Test```

![alt](data/docs/giao-dien-chuong-trinh-test.png)

Lưu lại ảnh và Kết thúc chương trình.

## Train model