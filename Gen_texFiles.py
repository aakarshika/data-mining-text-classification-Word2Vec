
import json

with open('namelist2.txt', 'r') as f:
	filenamelist = f.readlines()
filenamelist = [x.strip() for x in filenamelist] 

for i in range(0,499610):
	docj = json.load(open("./json/"+filenamelist[i]))
	doctext=docj["text"]
	print(filenamelist[i])
	url=docj["thread"]["url"]
	date=docj["published"]
	title=docj["title"]
	doctext=title+"\n\n"+date+"\n\n"+url+"\n\n"+doctext
	with open("./text/"+filenamelist[i][0:12]+".txt", "w") as f:
		f.write(doctext)

