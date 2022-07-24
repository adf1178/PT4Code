import json
import os

def pro_file(name):
    cs_file_name = name+".java-cs.txt.cs"
    java_file_name = name+".java-cs.txt.java"
    re = []
    cnt = 0
    with open(cs_file_name, 'r') as f1, open(java_file_name, 'r') as f2:
        for cs_line,java_line in zip(f1, f2):
            data_item = {}
            data_item['cs'] = cs_line
            data_item['java'] = java_line
            data_item['idx'] = cnt
            cnt+=1
            re.append(data_item)
    with open(name+".jsonl", 'w') as f:
        for item in re:
            f.write(json.dumps(item)+"\n")
            
pro_file('train')
pro_file('test')
pro_file('valid')