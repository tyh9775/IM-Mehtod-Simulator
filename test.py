import csv
import glob
import os

output_folder='test folder'
os.makedirs(output_folder,exist_ok=True)
new_file=os.path.join(output_folder,os.path.basename("test_file.csv"))
with open(new_file,'w',newline='') as nf:
  nf.close()

with open("data.csv",'r') as file:
  f=csv.reader(file,delimiter=',')
  with open(new_file,'a',newline='') as nf:
    nwriter=csv.writer(nf)
    for i in range(0,10):
      row=next(f)
      nwriter.writerow(row)
    nf.close()    
  file.close()    
