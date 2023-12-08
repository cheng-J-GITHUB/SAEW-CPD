import re
import pandas as pd
results = open('./nohup.txt')
data = results.readlines()
out = pd.DataFrame()
name = ' '
mid = []
for i in data:
  if re.findall(r"fishkiller/(.+?).mat",i):
    out[name] = mid
    name = re.findall(r"fishkiller/(.+?).mat",i)
    mid = []
  if re.findall(r"lossD(.+?)\n",i):
    mid.append(re.findall(r"lossD(.+?)\n",i))
out[name] = mid
out.to_excel('tb.xlsx',float_format='%.2f',)            # 路径和文件名 保留两位小数

              