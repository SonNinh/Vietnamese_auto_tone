

Toàn bộ code của mô hình Tranformer được mình xây dựng lại từ dầu. Các bạn có thể sử dụng mô hình có sẵn của PyTorch.   

Mô hình Transformer gồm 2 phần: Encoder và Decoder. Decoder có thể được huấn luyện bằng 2 phương pháp: teacher và none-teacher


# Vietnamese_auto_tone
Để giải quyết bài toán này, mình chỉ sử dụng Encoder. Mô hình không cần Decoder do input và ouput có cùng chiều dài và cùng cấu trúc ngữ pháp.

Đầu vào: 'cau vang di doi roi ong giao a'
Đầu ra mong muốn: 'cậu vàng đi đời rồi ông giáo ạ'

Gỉả sử:
- thư mục chứa data là */mnt/data/sonninh/vietnamese_tone*
- thư mục chứa project là */home/sonninh/Vietnamese_auto_tone*

# Chuẩn bị dữ liệu
Dữ liệu dùng cho quá trình huấn luyện được lấy từ Wikipedia. Tải về file có format tên **viwiki-yyyymmdd-pages-articles.xml.bz2** và giải nén bằng tool *WikiExtractor*.
```shell
$ pwd
/home/sonninh/Vietnamese_auto_tone/my_utils
$ python WikiExtractor.py /mnt/data/sonninh/vietnamese_tone/\<XML wiki dump file\> -p \<number of cpu cores\> -o /mnt/data/sonninh/vietnamese_tone/output --json
```
Các file json sau khi giải nén sẽ được lưu ở */mnt/data/sonninh/vietnamese_tone/output*. Quá trình giải nén tốn khá nhiều thời gian, vì thế *number of cpu cores* nên bằng tổng số CPU cores.

Tiếp theo, tách nội dung các file json thành những câu có nghĩa, mỗi câu 1 dòng, lưu trong các file \*.txt có tên tương ứng với các file \*.json. Trong file *save_to_txt.py*, *input_path* là đường dẫn đến thư mục chứa các file json, *output_path* là dường dẫn các file đầu ra. Giả sử *output_path* là */mnt/data/sonninh/vietnamese_tone/pre_processed/*
```shell
$ pwd
/home/sonninh/Vietnamese_auto_tone/my_utils
$ python save_to_txt.py
```
Copy file sort.sh vào trong thư mục */mnt/data/sonninh/vietnamese_tone/pre_processed/*
```shell
$ pwd
/mnt/data/sonninh/vietnamese_tone/pre_processed
$ bash sort.sh
```
Thu được các file sorted_\*.txt tương ứng. Di chuyển các file này vào 3 thư mục *train*, *val* và *test*.
Tiền xử lý, lưu data thành định dạng *pickle*. Trong file *data_processing.py*, **MAX_LEN** là chiều dài tối đa của chuỗi input, các câu dài hơn **MAX_LEN** sẽ bị loại bỏ. **OUTPUT_FILE** là tên file pickle chứa data để train, test, val và một số thông tin khác. **INPUT_DIR** là đường dẫn tới 3 thư mục *train*, *val* và *test*.
```shell
$ pwd
/home/sonninh/Vietnamese_auto_tone/my_utils
$ python data_processing.py
```
Hoàn thành các bước trên, thư mục chứa data có dạng:
```
output/
  | AA/
    | wiki_\*
  | AB/
  | ...
pre_processed/
  | AA.txt
  | \*.txt
  | sort.sh
  | vietnamese.pkl
  | val/
    | sorted_\*.txt
  | train/
    | sorted_\*.txt
  | test/
    | sorted_\*.txt
```
# Huấn luyện mô hình
```shell
$ pwd
/home/sonninh/Vietnamese_auto_tone
$ export PYTHONPATH=(pwd)  
```
Mô hình Encoders-Decoders:
```shell
bash train.sh
```
Mô hình Encoders:
```shell
bash train2.sh
```
Huấn luyện Encoders sau 10 epoch, độ chính xác đạt ~96%.
Hardware: 2 GPUs 2080RTX 11GB
Thời gian huấn luyện: 180 phút
# Inference
Mô hình Encoders
```shell
python infer.py
<<< cach mang cong nghiep lan thu hai
<init>cách mạng công nghiệp lần thứ hai<eos>

<<< nhung nuoc phat trien day manh nen kinh te dich vu
<init>những nước phát triển đẩy mạnh nền kinh tế dịch vụ<eos>

```
# Kết quả và đánh giá
Đối với bài toàn này, Encoders hoạt động tốt hơn Encoders-Decoders.   
Hàm loss tính theo từng ký tự trong chuỗi input nên độ chính xác đạt được (96%) chưa quá cao. Do mô hình chỉ cần giữ nguyên input làm output thì độ chính xác đã là khoảng 70%. Hướng cải thiện là chỉ tính hàm loss trên các nguyên âm.
