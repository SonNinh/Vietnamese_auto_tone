from glob import glob

import dill as pickle
import torch
import torchtext
from torchtext.data import BucketIterator

from nltk import ngrams
__author__ = 'Son Ninh'
MAX_LEN = 61
OUTPUT_FILE = '/mnt/data/sonninh/vietnamese_tone/pre_processed/mini_vietnamese.pkl'
INPUT_DIR = '/mnt/data/sonninh/vietnamese_tone/pre_processed'


s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
def remove_accents(input_str):
	s = ''
	for c in input_str:
		if c in s1:
			s += s0[s1.index(c)]
		else:
			s += c
	return s


def cvt_accents(input_str):
    map_encoded = []
    str_acc = {'a':'a', 'á':'a1', 'à':'a2', 'ả':'a3', 'ã':'a4', 'ạ':'a5',
            'o':'o', 'ó':'o1', 'ò':'o2', 'ỏ':'o3', 'õ':'o4', 'ọ':'o5',
            'i':'i', 'í':'i1', 'ì':'i2', 'ỉ':'i3', 'ĩ':'i4', 'ị':'i5',
            'y':'y', 'ý':'y1', 'ỳ':'y2', 'ỷ':'y3', 'ỹ':'y4', 'ỵ':'y5',
            'u':'u', 'ú':'u1', 'ù':'u2', 'ủ':'u3', 'ũ':'u4', 'ụ':'u5',
            'e':'e', 'é':'e1', 'è':'e2', 'ẻ':'e3', 'ẽ':'e4', 'ẹ':'e5',
            'â':'a6', 'ấ':'a61', 'ầ':'a62', 'ẩ':'a63', 'ẫ':'a64', 'ậ':'a65',
            'ă':'a8', 'ắ':'a81', 'ằ':'a82', 'ẳ':'a83', 'ẵ':'a84', 'ặ':'a85',
            'ô':'o6', 'ố':'o61', 'ồ':'o62', 'ổ':'o63', 'ỗ':'o64', 'ộ':'o65',
            'ơ':'o7', 'ớ':'o71', 'ờ':'o72', 'ở':'o73', 'ỡ':'o74', 'ợ':'o75',
            'ư':'u7', 'ứ':'u71', 'ừ':'u72', 'ử':'u73', 'ữ':'u74', 'ự':'u75',
            'ê':'e6', 'ế':'e61', 'ề':'e62', 'ể':'e63', 'ễ':'e64', 'ệ':'e65',
            'đ':'d9',
            '0':'))))))', '1':'!!!!!!', '2':'@@@@@@', '3':'######', '4':'$$$$$$',
            '5':'%%%%%%', '6':'^^^^^^', '7':'&&&&&&', '8':'******', '9':'(((((('}

    s = ''
    for i, c in enumerate(input_str[::-1]):
        if c in str_acc:
            s += str_acc[c]
            map_encoded += [i]*len(str_acc[c])
        else:
            s += c
            map_encoded.append(i)
    # print(s, map_encoded)
    return s, map_encoded


def tokenizer(text):
    return [c for c in text.lower()]


def main():
    SRC = torchtext.data.Field(
        sequential=True, use_vocab=True,
        tokenize=tokenizer,
        init_token='<init>', eos_token='<eos>'
    )
    TRG = torchtext.data.Field(
        sequential=True, use_vocab=True,
        tokenize=tokenizer,
        init_token='<init>', eos_token='<eos>'
    )

    fileds = [('src', SRC), ('trg', TRG)]

    datasets = {'val':None, 'train':None, 'test':None}
    for key in datasets.keys():
        examples =[] 
        root_dir = f'{INPUT_DIR}/{key}/*'
        for file_pth in glob(root_dir):
            print(file_pth)
            f = open(file_pth, 'r')
            row = f.readline().strip('\n')
            
            while row:
                if len(row) < MAX_LEN: 
                    src_name = remove_accents(row)
                    trg_name = row
                    ex = torchtext.data.Example.fromlist([src_name, trg_name], fileds)
                    examples.append(ex)
                else:
                    break
                row = f.readline().strip('\n')
            f.close()
            break

        datasets[key] = torchtext.data.Dataset(examples, fileds)

    SRC.build_vocab(datasets['train'].src)
    TRG.build_vocab(datasets['train'].trg)
    
    print(SRC.vocab.stoi)
    print()
    print(TRG.vocab.stoi)
    print()
    print(len(SRC.vocab))
    print(len(TRG.vocab))


    data = {
        'fields': {'src': SRC, 'trg': TRG},
        'train': datasets['train'].examples,
        'valid': datasets['val'].examples,
        'test': datasets['test'].examples
    }
    pickle.dump(data, open(OUTPUT_FILE, 'wb'))

  

if __name__ == "__main__":
    main()
