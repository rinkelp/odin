from sys import argv
import os
import re

a = "884"
b = "861"

#a = "879"
#b = "864"

gridFile = argv[1]
if os.path.isfile(gridFile):
	gridFiles = [gridFile]
else:
	gridFiles = []
	if gridFile[-1] != "/":
		gridFile = gridFile + "/"
	for i in os.listdir(gridFile):
		if re.search("[0-9].bin",i):
			gridFiles.append(gridFile + i)

f = open(gridFiles[0],"r")

header = f.read(1024)

header = header.split("\n")

f.close()

X = header[1].split("=")[1].strip()
Y = header[2].split("=")[1].strip()
maxQ = header[5].split("=")[1].strip()
wavelen = header[6].split("=")[1].strip()
detDist = header[7].split("=")[1].strip()
maxQ = "800"
#detDist = "0.100"
pixSize = "0.000110"
polariz = "0.99"

qres = "0.02"

cmd = ["/reg/neh/home3/dermen/test/MakePolarBin",\
	a,b,maxQ,pixSize,detDist,wavelen,polariz,\
	X,Y,qres]
cmd += gridFiles
cmd = " ".join(cmd)
print cmd
os.system(cmd)
