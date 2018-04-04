import zipfile
zip_ref = zipfile.ZipFile("zip.zip", 'r')

# for i in range(1,10):
# 	print(i)
# 	zip_ref.extract("news_000000{}.json".format(i), "./json/")
# for i in range(10,100):
# 	print(i)
# 	zip_ref.extract("news_00000{}.json".format(i), "./json/")
# for i in range(100,1000):
# 	print(i)
# 	zip_ref.extract("news_0000{}.json".format(i), "./json/")
# for i in range(1000,10000):
# 	print(i)
# 	zip_ref.extract("news_000{}.json".format(i), "./json/")
for i in range(100000,1000000):
	print(i)
	zip_ref.extract("news_0{}.json".format(i), "./json/")
zip_ref.close()