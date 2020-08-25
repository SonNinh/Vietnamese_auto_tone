# -*- coding: utf-8 -*-

import json
import os
import re
from os import path

from tqdm import tqdm


def write_in_the_star(files, parten_split, parten_match, parten_pick):
    bar = tqdm(files)
    for f in bar:
        with open(path.join(root, f), 'r') as fr:
            content = fr.read()
            content = content.split('\n')

            for para in content:
                if para:
                    data = json.loads(para)['text']
                    data = parten_split.split(data)
               
                    for line in data:
                        line_lower = line.lower()
                        ite = parten_pick.finditer(line_lower)
                        remain = []
                        pre_start = 0
                        for each in ite:
                            start = each.start(0)
                            end = each.end(0)
                            segment  = line[start:end]
                            segment_lower = line_lower[start:end]
                            if len(segment_lower.split()) > 2 and parten_match.match(segment_lower):
                                yield segment+'\n'
                                remain.append(line[pre_start:start])
                            else:
                                remain.append(line[pre_start:end])
                            pre_start = end
                        else:
                            remain.append(line[pre_start:])
                        if remain:
                            remain = ''.join(remain)
                            if len(remain.split()) > 2 and parten_match.match(remain.lower()):
                                yield remain+'\n'
        
            fr.close()

    
if __name__ == "__main__":

    input_path = '/mnt/data/sonninh/vietnamese_tone/output/'
    output_path = '/mnt/data/sonninh/vietnamese_tone/pre_processed/'
    output_path += '{}.txt'
    characters = '^[ _abcdefghijklmnopqrstuvwxyz0123456789áàảãạâấầẩẫậăắằẳẵặóòỏõọôốồổỗộơớờởỡợéèẻẽẹêếềểễệúùủũụưứừửữựíìỉĩịýỳỷỹỵđ!\"\',\-\.:;?_\(\)]+$'

    parten_split = re.compile(r' *[,;.:\x3F\x21\n]+ *')
    parten_match = re.compile(characters)
    parten_pick = re.compile(r'\x28[^\x28\x29]+\x29')

    for root, dirs, files in os.walk(input_path):
        if files:
            print(root)
            file_name = path.basename(root)
            fw = open(output_path.format(file_name), 'w+')
            
            fw.writelines(
                write_in_the_star(files, parten_split, parten_match, parten_pick)
            )
            fw.close()
