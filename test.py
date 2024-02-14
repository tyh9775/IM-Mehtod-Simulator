import myconst as mc
from array import array
print(mc.m_del0)

with open('data_bin.bin','rb') as f:
    datab=array('f')
    datab.frombytes(f.read())
    f.close()

print(datab)