# Dự án Data Mining: Khai phá sở thích và xây dựng hệ thống gợi ý Phim/Âm nhạc

## 1. Tóm tắt dự án
Dự án tập trung khai phá sở thích người dùng từ dữ liệu đánh giá, lịch sử tương tác và metadata nội dung (phim/âm nhạc) để tạo hệ thống gợi ý phù hợp, có khả năng mở rộng và dễ giải thích.

Vai trò định hướng: xem đây là một sản phẩm phân tích dữ liệu có thể triển khai theo từng giai đoạn, từ nghiên cứu dữ liệu đến mô hình hóa và tạo khuyến nghị thực tế.

## 2. Bài toán kinh doanh
- Người dùng có quá nhiều lựa chọn nội dung, khó tìm đúng phim/bài hát phù hợp.
- Hệ thống cần cá nhân hóa để tăng mức độ hài lòng và thời gian sử dụng.
- Cần một cách giải thích trực quan cho gợi ý, ví dụ theo luật kết hợp:
	- Nếu người dùng xem Avengers thì gợi ý Iron Man.

## 3. Mục tiêu dự án
### 3.1 Mục tiêu chính
- Phân tích dữ liệu người dùng để khai phá sở thích.
- Xây dựng mô hình gợi ý nội dung phù hợp.
- Trực quan hóa kết quả để hỗ trợ quyết định sản phẩm.

### 3.2 Mục tiêu kỹ thuật
- Làm sạch và chuẩn hóa dữ liệu đánh giá/lịch sử xem-nghe.
- Phân cụm người dùng để nhận diện nhóm sở thích tương đồng.
- Khai phá luật kết hợp từ hành vi đồng xuất hiện nội dung.
- Kết hợp insight từ clustering + association rules để tạo gợi ý.

## 4. Phạm vi dữ liệu
- Dữ liệu hành vi:
	- Lịch sử xem/nghe, lượt tương tác, timestamp.
- Dữ liệu đánh giá:
	- Rating, like/dislike, watch-time hoặc completion rate.
- Dữ liệu nội dung:
	- Thể loại, diễn viên/ca sĩ, đạo diễn, từ khóa, năm phát hành.

Gợi ý cấu trúc dữ liệu trong dự án:
- data/raw: dữ liệu gốc.
- data/cleaned: dữ liệu đã xử lý.

## 5. Phương pháp khai phá dữ liệu
### 5.1 Tiền xử lý dữ liệu
- Xử lý thiếu dữ liệu, trùng lặp, định dạng thời gian.
- Chuẩn hóa giá trị rating và biến phân loại.
- Tạo đặc trưng hành vi theo người dùng.

### 5.2 Clustering người dùng
Mục tiêu: tìm các nhóm người dùng có sở thích giống nhau.

Đầu vào gợi ý:
- Tần suất tiêu thụ theo thể loại.
- Điểm đánh giá trung bình theo nhóm nội dung.
- Hành vi theo thời gian (khung giờ, chu kỳ sử dụng).

Đầu ra mong đợi:
- Hồ sơ từng cụm người dùng.
- Insight hành vi đặc trưng của từng nhóm.
- Ứng dụng vào chiến lược gợi ý theo phân khúc.

### 5.3 Association Rule Mining
Mục tiêu: tìm quy luật đồng xuất hiện nội dung để gợi ý trực tiếp.

Ví dụ nghiệp vụ:
- Nếu người dùng xem Avengers thì gợi ý Iron Man.

Chỉ số đánh giá luật:
- Support: mức độ phổ biến của luật trong dữ liệu.
- Confidence: độ tin cậy của kết luận.
- Lift: mức tăng xác suất so với ngẫu nhiên.
## 6. Cấu trúc dự án
Smart-Movies-Recommoder/
│
├── data/
├── notebooks/
├── src/
├── outputs/
## 7. Kiến trúc triển khai trong repository
- notebooks/preprocessing.ipynb: khám phá và xử lý dữ liệu.
- notebooks/clustering.ipynb: thử nghiệm phân cụm và đánh giá cụm.
- notebooks/association_rules.ipynb: khai phá luật kết hợp.
- notebooks/eda.ipynb: phân tích dữ liệu khám phá.
- src/: module hóa code cho pipeline sản xuất.
- outputs/figures: biểu đồ.
- outputs/models: mô hình lưu trữ.
- outputs/results: bảng kết quả và luật khai phá.

## 8. How to Run

### 8.1 Yêu cầu môi trường
- Python 3.10+ (khuyến nghị 3.10 hoặc 3.11)
- pip
- Jupyter Notebook hoặc VS Code + Jupyter extension

### 8.2 Cài đặt nhanh
Trong thư mục gốc dự án, chạy lần lượt:

```bash
python -m venv .venv
```

Windows (CMD):

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Sau đó cài thư viện:

```bash
pip install -r requirements.txt
```

Lưu ý: nếu file requirements.txt chưa có package, hãy thêm các thư viện bạn dùng trong notebook trước khi chạy.

### 8.3 Chạy theo hướng Notebook
Mở Jupyter hoặc VS Code và chạy theo thứ tự:
1. notebooks/eda.ipynb
2. notebooks/preprocessing.ipynb
3. notebooks/clustering.ipynb
4. notebooks/association_rules.ipynb

Kết quả kỳ vọng:
- Biểu đồ trong outputs/figures
- Mô hình hoặc artifact trong outputs/models
- Bảng luật kết hợp/kết quả đánh giá trong outputs/results

### 8.4 Chạy theo hướng script (khi hoàn thiện src)
Khi các file trong thư mục src đã có code đầy đủ, chạy pipeline mẫu:

```bash
python src/preprocessing.py
python src/clustering.py
python src/association.py
python src/recommend.py
```

### 8.5 Kiểm tra nhanh kết quả rule-based
Ví dụ kỳ vọng sau khi chạy association rules:
- Input: người dùng đã xem Avengers
- Output gợi ý: Iron Man (nếu luật đạt ngưỡng support/confidence/lift)

## 9. Kết luận
Dự án này là nền tảng tốt cho một hệ thống gợi ý có tính ứng dụng cao, vừa tạo được insight hành vi người dùng (clustering), vừa tạo được luật gợi ý dễ giải thích (association rules).