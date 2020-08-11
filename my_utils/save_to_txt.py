# -*- coding: utf-8 -*-

import os
from os import path
import json
import re
from tqdm import tqdm


def write_in_the_star(files, parten1, parten2, parten3):
    for f in tqdm(files):
        with open(path.join(root, f), 'r') as fr:
            content = fr.read()
            content = content.split('\n')

            for para in content:
                if para:
                    data = json.loads(para)['text']
                    data = parten3.split(data)
                    for i in data:
                        i_out = parten1.sub(r' \1 ', i)
                        i_out = parten2.sub(r' ', i_out)
                        yield i_out+'\n'
        
            fr.close()
                

if __name__ == "__main__":

    input_path = '/home/sonninh/Downloads/transformer_with_self_attention/data'
    output_path = '/home/sonninh/Downloads/transformer_with_self_attention/data_txt/{}.txt'
    
    characters = '^[ _abcdefghijklmnopqrstuvwxyz0123456789áàảãạâấầẩẫậăắằẳẵặóòỏõọôốồổỗộơớờởỡợéèẻẽẹêếềểễệúùủũụưứừửữựíìỉĩịýỳỷỹỵđ!\"\',\-\.:;?_\(\)]+$'

    parten1 = re.compile(r'([^ \w])')
    parten2 = re.compile(r'( {2,})')
    parten3 = re.compile(r' *(?:\n|\. |, )+ *')

    for root, dirs, files in os.walk(input_path):
        if files:
            file_name = path.basename(root)
            fw = open(output_path.format(file_name), 'w+')
            # write_in_the_star(files, parten1, parten2, parten3)
            # for txt in write_in_the_star(files, parten1, parten2, parten3):
            #     fw.write(txt)
            fw.writelines(
                write_in_the_star(files, parten1, parten2, parten3)
            )
            fw.close()
            quit()