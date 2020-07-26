import os
from os import path
import json
import re


def parse_json(content):
    parten = re.compile(r'([^ \w])')
    for para in content:
        data = json.loads(para)['text']
        # print(data)
        data = re.split(' *(?:\n|\. |, )+ *', data)
        for i in data:
            print(parten.sub(r' \1 ', i))
            print()
        return 

input_path = '/mnt/data/sonninh/vietnamese_tone/output/'
output_path = '/mnt/data/sonninh/vietnamese_tone/all.txt'
fw = open(output_path, 'w+')
characters = '^[ _abcdefghijklmnopqrstuvwxyz0123456789áàảãạâấầẩẫậăắằẳẵặóòỏõọôốồổỗộơớờởỡợéèẻẽẹêếềểễệúùủũụưứừửữựíìỉĩịýỳỷỹỵđ!\"\',\-\.:;?_\(\)]+$'



for root, dirs, files in os.walk(input_path):

    for f in files:
        with open(path.join(root, f), 'r') as fr:
            content = fr.read()
            content = re.sub("(\s)+", r"\1", content)
            content = content.split('\n')
            parse_json(content)
            
            fr.close()

        quit()