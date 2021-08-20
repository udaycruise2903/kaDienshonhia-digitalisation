"""
Script for converting plain textfile to xlsx.
"""

import re
import xlsxwriter

input_filename = "~/Documents/khasi-dictionary-digitalisation/letter_files/Y.txt"

file1 = open(input_filename, 'r')


# enter the filename 
workbook = xlsxwriter.Workbook('Y-lines.xlsx')

# add new worksheet
worksheet = workbook.add_worksheet()

# specify data headers here.
worksheet.write('B31', 'word_number')
worksheet.write('C31', 'headword')
worksheet.write('D31', 'part-of-speech')
worksheet.write('E31', 'khasi_meaning')
worksheet.write('F31', 'english_meaning')
worksheet.write('G31', 'language')
worksheet.write('H31', 'additional_info')

# specify cell format
cell_format = workbook.add_format()
cell_format.set_text_wrap()


# specify the starting cell number 
cell_num = 32 

# specify initial counter value
word_num = 32

# word_count value
word_count = 1


#r2 = r'^.*(\badj\b|\badv\b|\bad\b|\baux\b|\bconj\b|\bimpe\b|\bimpers\b|\binter\b|\binterj\b|\bint\b|\blit\b|\bn\b|\bv\b|\bpron\b|\bphr\b|\bprep\b|\bpres\b|\bpl\b|\bsing\b|\bka\b|\bki\b|\bi—\b|\bu—\b).*?$'
r3 = r'^(?!\badj\b|\badv\b|\bad\b|\baux\b|\bconj\b|\bimpe\b|\bimpers\b|\binter\b|\binterj\b|\bint\b|\blit\b|\bn\b|\bv\b|\bpron\b|\bphr\b|\bprep\b|\bpres\b|\bpl\b|\bsing\b|\bka—\b|\bki—\b|\bi—\b|\bu—\b).*'



for lines in file1:
	if re.findall(r3, lines, flags = re.M):
		regex_var3 = re.findall(r3, lines, flags = re.M)[0]
		if regex_var3:
			#print(r1)
			# specifies the index of row cells.
			#cell_index1 = 'B' + str(word_num)
			cell_index3 = 'E' + str(cell_num)


			#worksheet.write(cell_index1, word_count) #B-column
			# output is written in the rows of C
			worksheet.write(cell_index3, regex_var3, cell_format)

			#word_num += 1
			#word_count += 1
			

		else:
			worksheet.write(cell_index3, '')
	


		cell_num += 1

workbook.close()
file1.close()
