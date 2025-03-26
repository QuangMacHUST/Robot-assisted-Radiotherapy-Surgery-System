# RoboRadAssist - Hệ Thống Robot Hỗ Trợ Xạ Trị Phẫu Thuật

![RoboRadAssist Logo](docs/images/logo.png)

## Tổng Quan

RoboRadAssist là một hệ thống robot phẫu thuật tiên tiến được thiết kế để hỗ trợ các bác sĩ trong việc thực hiện các ca phẫu thuật xạ trị với độ chính xác cao, nhắm đến các khối u ở nhiều cơ quan khác nhau. Dự án kết hợp công nghệ robot, trí tuệ nhân tạo, xử lý hình ảnh y tế và kỹ thuật xạ trị hiện đại nhằm nâng cao hiệu quả điều trị ung thư.

## Mục Tiêu Dự Án

- Định vị và điều trị khối u với độ chính xác micromet, giảm thiểu tổn thương cho các mô lành xung quanh
- Rút ngắn thời gian phẫu thuật và thời gian hồi phục của bệnh nhân
- Cá nhân hóa quy trình điều trị dựa trên đặc điểm riêng của từng khối u và cơ quan
- Tạo nền tảng tích hợp công nghệ hiện đại, dễ dàng nâng cấp trong tương lai
- Cung cấp giao diện trực quan, dễ sử dụng cho đội ngũ y tế

## Công Nghệ Sử Dụng

### Phần Cứng
- **Robot Chính Xác Cao**: Hệ thống cánh tay robot 6-DOF với độ chính xác micromet
- **Hệ Thống Hình Ảnh Y Tế**: Tích hợp CT, MRI, PET scanner
- **Thiết Bị Xạ Trị**: Tương thích với các máy xạ trị LINAC, CyberKnife, proton therapy
- **Cảm Biến Theo Dõi Thời Gian Thực**: Giám sát chuyển động của bệnh nhân và cơ quan

### Phần Mềm
- **Framework**: Python, C++, ROS (Robot Operating System)
- **Trí Tuệ Nhân Tạo**: PyTorch, TensorFlow cho phân tích hình ảnh và lập kế hoạch điều trị
- **Xử Lý Hình Ảnh Y Tế**: ITK, VTK, DICOM processing
- **Lập Kế Hoạch Xạ Trị**: Thuật toán tối ưu hóa liều lượng, mô phỏng Monte Carlo
- **Giao Diện Người Dùng**: Qt, React, WebGL cho 3D visualization
- **Cơ Sở Dữ Liệu**: PostgreSQL, MongoDB cho lưu trữ dữ liệu bệnh nhân và kết quả điều trị

## Cấu Trúc Dự Án

```
RoboRadAssist/
├── docs/                      # Tài liệu dự án
│   ├── images/                # Hình ảnh, sơ đồ
│   ├── api/                   # Tài liệu API
│   └── user_guides/           # Hướng dẫn sử dụng
├── src/                       # Mã nguồn chính
│   ├── core/                  # Thành phần lõi của hệ thống
│   │   ├── robot_control/     # Điều khiển robot
│   │   ├── image_processing/  # Xử lý hình ảnh y tế
│   │   ├── treatment_planning/# Lập kế hoạch điều trị
│   │   └── dose_calculation/  # Tính toán liều xạ trị
│   ├── ai/                    # Mô hình AI và thuật toán
│   │   ├── segmentation/      # Phân đoạn hình ảnh
│   │   ├── registration/      # Đăng ký hình ảnh
│   │   └── optimization/      # Tối ưu hóa kế hoạch điều trị
│   ├── ui/                    # Giao diện người dùng
│   │   ├── web/               # Giao diện web
│   │   ├── desktop/           # Ứng dụng desktop
│   │   └── visualization/     # Hiển thị 3D
│   ├── database/              # Quản lý cơ sở dữ liệu
│   └── communication/         # Giao tiếp hệ thống
├── tests/                     # Kiểm thử
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── system/                # System tests
├── tools/                     # Công cụ phát triển
├── examples/                  # Ví dụ sử dụng
├── data/                      # Dữ liệu mẫu
│   ├── phantoms/              # Dữ liệu phantom
│   ├── sample_images/         # Hình ảnh y tế mẫu
│   └── beam_data/             # Dữ liệu chùm tia
├── scripts/                   # Scripts hỗ trợ
├── requirements.txt           # Dependencies cho Python
├── CMakeLists.txt             # Build configuration cho C++
├── Dockerfile                 # Docker configuration
├── LICENSE                    # Giấy phép sử dụng
└── README.md                  # Tài liệu này
```

## Tính Năng Chính

### 1. Quản Lý Bệnh Nhân
- Tạo và quản lý hồ sơ bệnh nhân
- Nhập và lưu trữ dữ liệu hình ảnh y tế (DICOM)
- Theo dõi lịch sử điều trị và kết quả

### 2. Xử Lý Hình Ảnh Y Tế
- Tự động phân đoạn (segmentation) cơ quan và khối u
- Đăng ký hình ảnh (image registration) giữa các phương thức chụp khác nhau
- Tái tạo hình ảnh 3D chất lượng cao
- Phân tích định lượng khối u và cơ quan

### 3. Lập Kế Hoạch Điều Trị
- Mô phỏng và tối ưu hóa kế hoạch xạ trị
- Tính toán liều xạ chính xác dựa trên mô hình Monte Carlo
- Đề xuất góc chiếu và vị trí tối ưu dựa trên AI
- So sánh các phương án điều trị khác nhau

### 4. Điều Khiển Robot
- Lập trình chuyển động robot dựa trên kế hoạch điều trị
- Điều chỉnh thời gian thực theo chuyển động của bệnh nhân
- Hệ thống an toàn đa lớp với khả năng dừng khẩn cấp
- Mô phỏng chuyển động trước khi thực hiện

### 5. Theo Dõi Thời Gian Thực
- Giám sát vị trí và chuyển động của khối u
- Điều chỉnh kế hoạch điều trị theo thời gian thực
- Ghi nhận và phân tích dữ liệu trong suốt quá trình điều trị

### 6. Báo Cáo và Phân Tích
- Tạo báo cáo tự động về kế hoạch và kết quả điều trị
- Phân tích thống kê hiệu quả điều trị
- Xuất dữ liệu theo nhiều định dạng khác nhau

## Hướng Dẫn Cài Đặt

### Yêu Cầu Hệ Thống
- **Hệ điều hành**: Windows 10/11, Ubuntu 20.04/22.04, macOS 12+
- **CPU**: Intel i7/i9 hoặc AMD Ryzen 7/9 (8+ cores)
- **RAM**: 32GB+
- **GPU**: NVIDIA RTX 3080+ với 12GB+ VRAM (cho AI và render 3D)
- **Ổ cứng**: 1TB SSD
- **Phần mềm**: Python 3.9+, CUDA 11.7+, Docker

### Cài Đặt Môi Trường
1. Clone dự án:
```bash
git clone https://github.com/username/RoboRadAssist.git
cd RoboRadAssist
```

2. Sử dụng Docker (khuyến nghị):
```bash
docker-compose up -d
```

3. Hoặc cài đặt trực tiếp:
```bash
# Cài đặt dependencies Python
pip install -r requirements.txt

# Biên dịch các thành phần C++
mkdir build && cd build
cmake ..
make -j8
```

## Hướng Dẫn Sử Dụng

1. **Khởi động hệ thống**:
```bash
python src/main.py
```

2. **Truy cập giao diện web**:
```
http://localhost:8080
```

3. **Sử dụng ứng dụng desktop**:
```bash
python src/ui/desktop/main.py
