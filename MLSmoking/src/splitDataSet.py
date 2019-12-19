
# file = open("training.txt",'w')
# file2 = open("remaing.txt", 'w')
# count = 0
# with open('testing.txt','r') as f:
# 	while(count < 4):
# 		b = f.readline()
# 		if(b != "\n"):
# 			count = count + 1
# 			file.write(b)
# 	file.close()
# 	while(count >=4 and count <= 8):
# 		b = f.readline()
# 		if(b != "\n"):
# 			count = count + 1
# 			file2.write(b)		
	# start_index = 0
	# f.seek(start_index)
	# data = f.read(4 - start_index)
	# print(data.replace("\n"," "))

training_cold_turkey = open("training_cold_turkey.txt",'w')
testing_cold_turkey = open("testing_cold_turkey.txt", 'w')

training_vape_ex = open("training_vape_ex.txt", 'w')
testing_vape_ex = open("testing_vape_ex.txt", 'w')

training_bad_vape = open("training_bad_vape.txt", 'w')
testing_bad_vape = open("testing_bad_vape.txt",'w')

count = 1
with open('coldTurkeyData.csv', 'r') as f:
	while(count < 2326):
		lineContent = f.readline()
		if(lineContent != "\n"):
			count = count + 1
			training_cold_turkey.write(lineContent)
	training_cold_turkey.close()
	while(count >= 2326 and count <= 4563):
		lineContent = f.readline()
		if(lineContent != "\n"):
			count = count + 1
			testing_cold_turkey.write(lineContent)		
	testing_cold_turkey.close()
count = 1
with open('vaping_ex_mac.csv', 'r', encoding="ISO-8859-1") as f:
	while(count < 4000):
		lineContent = f.readline()
		if(lineContent != "\n"):
			count = count + 1
			training_vape_ex.write(lineContent)
	training_vape_ex.close()
	while(count >= 4000 and count <= 8000):
		lineContent = f.readline()
		if(lineContent != "\n"):
			count = count + 1
			testing_vape_ex.write(lineContent)		
	testing_vape_ex.close()
count = 1
with open('bad_vape.txt', 'r') as f:
	while(count < 3888):
		lineContent = f.readline()
		if(lineContent != "\n"):
			count = count + 1
			training_bad_vape.write(lineContent)
	training_bad_vape.close()
	while(count >= 3888 and count <= 7776):
		lineContent = f.readline()
		if(lineContent != "\n"):
			count = count + 1
			testing_bad_vape.write(lineContent)		
	testing_bad_vape.close()
