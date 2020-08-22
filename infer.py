import argparse
from os import path
import os
import dill as pickle
import torch
import torch.nn.functional as F
import torchtext
from torch import optim
from torchtext.data import BucketIterator
from tqdm import tqdm
from train2 import TransformerEnc
from collections import OrderedDict


def copyStateDict(state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict

def main():

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    model_pth = '/mnt/data/sonninh/trained_models/vietnamese_tone_enc_bigdata/epoch_9_1.chkpt'
    checkpoint = torch.load(model_pth)
    opt = checkpoint['opt']

    model = TransformerEnc(
        n_vocab_in=opt.n_vocab_in, n_vocab_out=opt.n_vocab_out,
        emb_dim=opt.emb_dim, pad_idx=opt.pad_idx,
        max_seq_len=opt.max_seq_len, dropout=opt.dropout,
        n_block=opt.n_block, attn_dim=opt.attn_dim,
        feedforward_dim=opt.feedforward_dim ,
        n_head=opt.n_head, training=False
    )
    
    model.load_state_dict(
        copyStateDict(checkpoint['model'])
    )
    model = model.to(device)

    itos, stoi = checkpoint['trg_vocab'].itos, checkpoint['src_vocab'].stoi

    model.eval()
    with torch.no_grad():
        while True:
            raw_seq = input('<<< ')
            idx_seq = [stoi[c] for c in raw_seq]
            idx_seq.insert(0, 2)
            idx_seq.append(3)
            src_seq = torch.tensor(idx_seq).type(torch.long).unsqueeze(0).to(device)
            pred = model(src_seq)
            pred = pred.max(-1)[1]
            pred = pred.view(-1).tolist()
            pred = ''.join([itos[int(i)] for i in pred])
            print(pred, '\n')
            


if __name__ == "__main__":
    main()
'''
<init>cách mạng công nghiệp lần thứ hai<eos>
<init>trong tại điều khiển trận đấu<eos>
<init>trung quốc đã mở rộng ảnh hưởng của họ trong khu vực<eos>
<init>thông qua các bưộc leo tháng ép bưộc các nước làng giềng<eos>
<init>các bước được ghi trong hướng dẫn<eos>
<init>thành công của đội tuyển bóng đá<eos>
<init>sự suy thoái của nền kinh tế dẫn đến nhiều hệ lũy<eos>
<init>những nước phát triển đẩy mạnh nền kinh tế dịch vụ<eos>
<init>nhà nước đáng nỗ lực xóa nạn mù chủ<eos>
<init>nhịệu công ty nước ngoài đầu tư vào ngành du lịch<eos>
<init>ngành viên thông là mũi nhọn kinh tế nước ta<eos>
<init>nhà máy sản xuất sữa lớn nhất thế giới<eos> 
<init>hỗ trợ miễn dịch dưỡng ruột<eos> 
<init>công ty cổ phần sửa việt nam<eos> 
<init>sửa chữa uống mến sông<eos> 
<init>cậư bé ban rau quyết không từ bỏ ước mở đến trưởng<eos> 
<init>lời đi mới cho người mất gốc tiếng anh<eos> 
<init>giải quyết bà vấn đề của người viết khi học tiếng anh<eos> 
<init>thiết bị cảnh báo chống trộm<eos> 
<init>hàng nghìn thí sinh đã trung tuyến top đầu<eos> 
<init>kỹ năng đặc thư của tinh thần doanh nghiệp<eos> 
<init>du học sinh chia sẻ trai nghiệm đại học lớn nhất xử wale<eos> 
<init>làm gia giấy trung tuyến đại học<eos>
<init>học sinh giả mạo văn bản của sở gddt<eos> 
<init>hàng loạt trường công bố điểm chuẩn học ba<eos> 
<init>tôi tùng không thể đi lại bình thường<eos> 
<init>lịch sử khẩu trang của người nhật<eos> 
<init>hàn quốc tại áp đặt quy định giản cách xã hội<eos> 
<init>thẩm phán tòa án tối cao<eos> 
<init>tranh cãi về ý tưởng tiệm vaccine<eos> 
<init>bề bởi khiến cứu cô văn vương vòng lao lý<eos>
<init>nhưng điểm công của biển so oto mỗi so với loài cũ<eos>  
<init>xe tải bị xe toặc đầu đó tránh xe máy<eos> 
<init>tôi có ý định mùa xe vào tháng này<eos> 
<init>tranh chấp kết quả bầu cử<eos> 
<init>gia vàng đi xuống cưối tuần<eos> 
<init>giải cứu 1000 còn cả khi kênh nước vỡ<eos> 
<init>một con trâu với hai đầu hai tại hai miệng<eos> 
<init>thiết bị cảnh báo ô nhiễm không khi nhỏ như báo điểm<eos> 
<init>hỗ trợ bộ thuộc là sau 5 ngày<eos> 
<init>đào tạo kỹ sư trì tuệ nhân tạo<eos> 
<init>trẻ bắt đầu cao nhánh từ 5 tuổi<eos> 
<init>chuồn chuồn cái giá chết để tránh con đực quay nhiều<eos> 
<init>cảnh sát truy đuổi tên trộm<eos> 
<init>lập bệnh viện để lửa đào người đến khâm<eos> 
<init>dùng đơn tâm lý để khuất phục kế sát nhân<eos> 
<init>facebook lộ trump có thể căn thiếp kết quả bầu cứ<eos> 
<init>mỹ kiểm soát ngành công nghiệp sản xuất chip<eos> 
<init>cuộc đua internet vệ tinh của elon musk<eos> 
<init>aplle để lộ ngày ra mắt iphone<eos> 
<init>235 triệu tại khoản mang xã hội bị phát tấn<eos> 
<init>tại khoản ngân hàng của tới có 9 số 0<eos> 
<init>ai đây hàng loạt người vào nguy cơ thất nghiệp<eos> 
<init>cơ sở dữ liệu chưa thông tin cá nhân<eos> 
<init>3 công trình giao thông trong điểm<eos> 
<init>cho nghiệp vụ tập vượt chương ngại vật<eos> 
<init>trường đại học sang chế robot diệt khuẩn<eos> 
<init>cuộc sống lạc quan của cơ giao nhiệm hiv<eos> 
<init>cô giao chủ nhiệm<eos> 
<init>nghị lực của người phụ nữ bị ung thư<eos>
<init>ngôi nhà 2 tầng thuộc sở hữu của tới<eos> 
<init>5 cách tự nhiên để dưới mười trong nhà<eos> 
<init>trung tâm y tế trung ương<eos> 
<init>chúng chỉ hành nghệ<eos> 
<init>các trò chơi truyền thống được tổ chức trong lễ hội<eos> 
<init>công ty công nghệ sinh học<eos> 
<init>đây thị là giai đoạn phát triển chiếu cao tốt nhất<eos> 
<init>tâm lý mẹ bầu ảnh hưởng tới chí thông minh củâ trễ<eos> 
<init>thấy xuống ba vài nhân tạo cho cậu bé bị ung thư<eos> 
<init>nhưng bức ảnh du lịch đẹp nhất năm<eos> 
<init>chuyên gia nói gì về đế xuất mở đường bay quốc tê<eos> 
<init>mùa trả góp oto<eos> 
<init>tham vòng của sunhouse trên thị trường diện dẫn dụng<eos> 
<init>sự thống trị của bigfour trong kinh tế mỹ<eos> 
<init>hướng dẫn thay đổi mặt khẩu trên điện thoại<eos> 
<init>quân đội phòng tòa toàn bộ thành phố<eos> 
'''