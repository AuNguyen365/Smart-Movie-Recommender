# Báo Cáo Giải Thích Về Kỹ Thuật Encoding 

## 1. Encoding là gì và tại sao chúng ta phải làm nó?

Nói một cách đơn giản nhất: **Máy tính và các thuật toán Trí Tuệ Nhân Tạo (AI) hoàn toàn "mù chữ".** 

Các công thức toán học và AI không thể hiểu được chữ `"Hành động" (Action)`, tên diễn viên `"Tom Cruise"`, hay mã người dùng `"Sejian"`. AI chỉ hiểu và tính toán được với **các con số (0, 1, 0.5, 99...)**. 

**Encoding (Mã hóa)** chính là quá trình chúng ta làm "phiên dịch viên": Dịch các dữ liệu chữ viết, nhóm loại (Category), và dạng danh sách sang dạng Tần số hoặc Số có quy luật, để máy tính có thể phân tích được sự tương đồng giữa chúng nhằm đưa ra bộ phim gợi ý tốt nhất.

---

## 2. Dữ liệu của chúng ta đã được xử lý như thế nào? (Trước & Sau)

Trong file `preprocessing.py` vừa chạy trên dữ liệu `integrated_dataset_cleaned.csv`, đây là cách dữ liệu biến hình:

### A. Label Encoding (Chuyển chuỗi hoặc ID sang Số đếm bắt đầu từ 0)
- **Áp dụng cho:** Người dùng (`user_id`), Bộ phim (`movie_id`), Thể loại chính (`primary_genre`).
- **Nó làm gì:** Phát cho mỗi người/vật một "số thứ tự định danh" (từ `0` đến `N`). Rất quan trọng khi xây dựng ma trận Lọc Cộng tác (Collaborative Filtering).
- **Ví dụ Trước/Sau:**
  - `user_id` "Sejian" ➔ Chuyển thành `user_idx`: **3102**
  - `user_id` "MovieGuys" ➔ Chuyển thành `user_idx`: **124**
  - Giờ đây thuật toán hiểu là "User số 3102 đang xem phim số 45".

### B. Multi-Label Binarizer (Mã hóa Đa Nhãn thành Cột 0/1)
- **Áp dụng cho:** Danh sách thể loại của một bộ phim (`genres_list`).
- **Nó làm gì:** Vì một phim có thể có nhiều thể loại, chúng ta tách tổng cộng **27 loại phim khác nhau** thành **27 cột mới**. Nếu phim có thể loại nào thì đánh số `1`, không có thì đánh số `0`.
- **Ví dụ Truyền thuyết Batman:**
  - **Trước:** `['action', 'crime', 'drama']`
  - **Sau:** Cột `genre_action` = **1**, Cột `genre_crime` = **1**, Cột `genre_drama` = **1**, Cột `genre_comedy` = **0**...

### C. Min-Max Scaling (Đưa số liệu lớn về cùng một hệ quy chiếu)
- **Áp dụng cho:** Điểm đánh giá (`rating`), Năm phát hành (`release_year`), v.v.
- **Nó làm gì:** Scale dữ liệu về cùng một cái cân đo chuẩn ở mức `[0.0 đến 1.0]`. Tránh việc con số "năm 2024" to gấp hàng trăm lần "điểm 8.5" làm máy học ưu tiên "năm" hơn là "điểm" (AI cực kỳ nhạy cảm với những con số to).
- **Ví dụ:**
  - `rating`: Điểm 10.0 ➔ Scale thành **1.00**
  - `rating`: Điểm 5.5 ➔ Scale thành **0.50**

### D. TF-IDF Vectorizer (Mã hóa văn bản thành Vector Tần số)
- **Áp dụng cho:** Diễn viên (`cast`).
- **Nó làm gì:** Có hàng vạn diễn viên, làm sao máy tính biết bộ phim nào chung nhiều diễn viên với bộ phim nào? Kỹ thuật này sẽ biến 500 diễn viên hot nhất thành một vector tính điểm. Diễn viên nào ít đóng phim nhưng xuất hiện ở phim nào là làm nên dấu ấn phim đó thì điểm sẽ cao; còn diễn viên quần chúng đóng 1000 phim ở đâu cũng thấy thì điểm tính toán sẽ thấp.
- **Kết quả:** Biến chuỗi text `"Robert Downey Jr., Chris Evans"` thành một mảng số ngầm định để chấm khoảng cách giữa các phim với nhau.

---

## 3. Các File được sinh ra sau quá trình này để làm gì?

Giờ đây, bạn đã có một bộ dữ liệu hoàn hảo, cực kỳ thích hợp để train Model AI Recommender System! Tính năng này đẻ ra 3 khối tài sản:

1. **Bảng số liệu `integrated_dataset_encoded.csv` (Nằm ở `data/cleaned/`)**
   - Bộ dữ liệu này phình ra thêm tận ~48 cột (chiếm diện tích lớn nhất là 27 cột 0/1 của thể loại). Các cột cũ dạng Text vẫn được giữ nguyên để con người nhìn; nhưng chúng ta có thêm những cột mới như `rating_scaled`, `user_idx`, `movie_idx`, `genre_action`... dành riêng cho máy tính. Phân tích trên file này sẽ nhanh gấp nhiều lần!
2. **Ma Trận Diễn Viên `cast_tfidf_matrix.npz` (Nằm ở `outputs/encoders/`)**
   - Sự ma thuật của toán học. Đó là một file bị nén lại vì chứa đến (10659 x 500) phần tử số thể hiện sự tương đồng của các dàn diễn viên. Sẽ dùng lập tức khi bạn muốn build tính năng: *"Vì bạn thích Phim có Tom Cruise, hệ thống gợi ý các phim này...".*
3. **Từ Điển Dịch Thuật Các file `.pkl` (Nằm ở `outputs/encoders/`)**
   - Máy học các con số, nhưng khi xuất kết quả lên website hay ứng dụng cho người dùng thì hệ thống cần hiện ra Hình Ảnh và Chữ Nghĩa. Có **7 file .pkl** này giống như 7 cuốn từ điển ngược, giúp thuật toán kết xuất từ con số gán nhãn `3102` dịch ngược về lại nick người dùng có mã `"Sejian"` cho Front-end hiển thị đúng người đó.

**Tóm lại:** Dữ liệu file `_cleaned.csv` trước đó là file sạch để **Con người (bạn)** đọc. Còn cái nùi thư mục tôi vừa hoàn thành là mớ đồ ăn dặm xay nhuyễn để **Máy Học (AI Model)** dễ tiêu hóa!
