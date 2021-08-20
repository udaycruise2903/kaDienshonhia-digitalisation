import re

# enter file path and names
filename1 = "~/Documents/khasi-dictionary-digitalisation/clean-conv2txt/mainfile_OCRinput.txt"
filename2 = "~/Documents/khasi-dictionary-digitalisation/clean-conv2txt/cleaned_textfile.txt"

file1 = open(filename1, 'r')

file2 = open(filename2, 'w')

# map(lambda line: line.rstrip('\n'), file1):
for line in file1.readlines():
	r1 = re.findall(r'^[A-ZÑÏÌÀ\- ]+$', line, flags = re.M)
	r2 = re.findall(r'^[0-9]+$', line, flags = re.M)
	r3 = re. findall(r'[/*|]',line, flags = re.M)
	#r3 = re.findall(r'(?:\n)$', line, flags = re.M)
	#r3 = re.findall(r'.*\n', line, flags = re.M)
	#r4 = re.sub(r'^\s*$','\t','line', flags = re.M)


	#not r1 and not r2 and
	if not r1 and not r2 and not r3:

		print(line)
        
		file2.write(line)

file1.close()
file2.close() 



'''
int.|int|(ka—) n.|(u—), n.|adv.|
'''
