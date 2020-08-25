

Toàn bộ code của mô hình Tranformer được mình xây dựng lại từ dầu. Các bạn có thể sử dụng mô hình có sẵn của PyTorch.   

Mô hình Transformer gồm 2 phần: Encoder và Decoder. Decoder có thể được huấn luyện bằng 2 phương pháp: teacher và none-teacher


# Vietnamese_auto_tone
Để giải quyết bài toán này, mình chỉ sử dụng Encoder. Mô hình không cần Decoder do input và ouput có cùng chiều dài và cùng cấu trúc ngữ pháp.

Đầu vào: 'cau vang di doi roi ong giao a'
Đầu ra mong muốn: 'cậu vàng đi đời rồi ông giáo ạ'

# Chuẩn bị dữ liệu
Dữ liệu dùng cho quá trình huấn luyện được lấy từ Wikipedia. Tải về file có format tên **viwiki-yyyymmdd-pages-articles.xml.bz2** và giải nén bằng tool *WikiExtractor*.
```shell
python WikiExtractor.py \<XML wiki dump file\> -p \<number of cpu cores\> -o \<output directory\> --json
```
Các file json sau khi giải nén sẽ được lưu ở *output directory*. Quá trình giải nén tốn khá nhiều thời gian, vì thế *number of cpu cores* nên bằng tổng số CPU cores.   
Tiếp theo, tách nội dung trong các file json thành các câu có nghĩa, mỗi câu 1 dòng. 