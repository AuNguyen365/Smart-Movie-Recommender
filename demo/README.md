# Smart Movie Recommender - Web Demo Guide

Chào mừng bạn đến với bản Demo của hệ thống gợi ý phim thông minh! Tài liệu này sẽ hướng dẫn bạn cách khởi động cả Backend và Frontend để trải nghiệm đầy đủ các tính năng của hệ thống.

## 📋 Yêu cầu hệ thống
- **Python 3.10+** (Đã cài đặt trong môi trường ảo `.venv`)
- **Node.js & npm** (Để chạy giao diện React)

---

## 🚀 Cách khởi động Demo

Để chạy ứng dụng, bạn cần mở **2 cửa sổ Terminal** riêng biệt:

### 1. Khởi động Backend (FastAPI)
Backend chịu trách nhiệm xử lý các thuật toán gợi ý, tìm kiếm mờ và cung cấp dữ liệu phân cụm.

1. Mở Terminal và di chuyển vào thư mục backend:
   ```powershell
   cd demo/backend
   ```
2. Chạy lệnh khởi động server:
   ```powershell
   ..\..\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
   ```
   *Lưu ý: Giữ cửa sổ này mở trong suốt quá trình sử dụng.*

### 2. Khởi động Frontend (React + Vite)
Frontend là giao diện người dùng trực quan để bạn tương tác với hệ thống.

1. Mở cửa sổ Terminal thứ hai và di chuyển vào thư mục frontend:
   ```powershell
   cd demo/frontend
   ```
2. Cài đặt thư viện (nếu là lần đầu chạy):
   ```powershell
   npm install
   ```
3. Khởi động giao diện:
   ```powershell
   npm run dev
   ```
4. Truy cập vào địa chỉ: [http://localhost:5175](http://localhost:5175)

---

## ✨ Các tính năng chính trong Demo

1. **Top Recommendations:** Chọn một người dùng trong danh sách để xem gợi ý phim được cá nhân hóa dựa trên gu của họ. Có hỗ trợ chế độ "Người dùng mới" (Cold Start).
2. **Movie Search:** Tìm kiếm phim và xem các phim liên quan. Hệ thống sử dụng **Fuzzy Search** nên bạn có thể gõ sai chính tả hoặc thiếu dấu cách (ví dụ: `jhon wick`, `spiderman`) vẫn tìm ra kết quả.
3. **Audience Clusters:** Xem cách AI phân loại người dùng thành các nhóm sở thích khác nhau. Khi bạn chọn một User ở Sidebar, cụm tương ứng của họ sẽ tự động phát sáng.

---

## 🛠 Xử lý sự cố
- **Lỗi "Port already in use":** Đảm bảo bạn không có cửa sổ terminal cũ nào đang chạy backend hoặc frontend. Hãy tắt chúng đi và chạy lại lệnh.
- **Không tải được dữ liệu:** Kiểm tra xem cửa sổ Backend (cổng 8000) có đang báo lỗi đỏ không. Đảm bảo file `recommendations.csv` và các file trong `outputs/` tồn tại.

---
*Chúc bạn có những trải nghiệm tuyệt vời với Smart Movie Recommender!*
