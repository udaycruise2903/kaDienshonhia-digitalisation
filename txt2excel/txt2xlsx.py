"""
Script for converting plain textfile to xlsx.
"""

import re
import xlsxwriter


input_filename = "~/Documents/khasi-dictionary-digitalisation/letter_files/Y.txt"

file1 = open(input_filename, 'r')

# enter the filename 
workbook = xlsxwriter.Workbook('Y.xlsx')

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

#cell_index_list = ['C', 'D', 'G', 'H']

# specify cell format
cell_format = workbook.add_format()
cell_format.set_text_wrap()

# specify the starting cell number 
words_cell_num = 32
cell_num = 32

# specify initial word_count value
word_count = 1



# specify all regex here
r1 = r'^[s ][0-9Ïa-záìïñù\-]+'
r2 = r'^.*(\badj\b|\badv\b|\bad\b|\baux\b|\bconj\b|\bimpe\b|\bimpers\b|\binter\b|\binterj\b|\bint\b|\blit\b|\bn\b|\bv\b|\bpron\b|\bphr\b|\bprep\b|\bpres\b|\bpl\b|\bsing\b|\bka—\b|\bki—\b|\bi—\b|\bu—\b).*?$' # add ? for non greedy search
r3 = r'^(?!(\badj\b|\badv\b|\bad\b|\baux\b|\bconj\b|\bimpe\b|\bimpers\b|\binter\b|\binterj\b|\bint\b|\blit\b|\bn\b|\bv\b|\bpron\b|\bphr\b|\bprep\b|\bpres\b|\bpl\b|\bsing\b|\bka—\b|\bki—\b|\bi—\b|\bu—\b)$).*$'
#r4 = r'(.*)$'
r5 = r'(H..d.|Eng.....|Bengal.|Assamese|Arabic)'
r6 = r'(Synteng|Phareng)'

#regexlist = [r1, r2, r5, r6]

for lines in file1:
	if re.findall(r1, lines, flags = re.M):  
		reg_var1 = re.findall(r1, lines, flags = re.M)[0]
		if reg_var1:
			
			#for y in cell_index_list:
			# specifies the index of row cells.
			cell_index0 = 'B' + str(words_cell_num)
			cell_index1 = 'C' + str(cell_num)
			
			worksheet.write(cell_index0, word_count) #B-column
			# output of  r1 is written in the rows of C
			worksheet.write(cell_index1, reg_var1, cell_format) #C-column
				
			words_cell_num += 1
			word_count += 1

		else:
			worksheet.write(cell_index1, '')

	#if re.findall(r3, lines, flags = re.M):
	#	regex_var3 = re.findall(r3, lines, flags = re.M)[0]
	#	if regex_var3:

	#		cell_index3 = 'E' + str(cell_num)

			# output is written in the rows of E
	#		worksheet.write(cell_index3, regex_var3, cell_format)
			
	#	else:
	#		worksheet.write(cell_index3, '')

	if re.findall(r2, lines, flags = re.M):
		reg_var2 = re.findall(r2, lines, flags = re.M)[0]
		if reg_var2:
			
			# specifies the index of row cells.			 
			cell_index2 = 'D' + str(cell_num)
			# output is written in the rows of D
			worksheet.write(cell_index2, reg_var2, cell_format)

		else:
			worksheet.write(cell_index2, '')

	#if re.findall(r3, lines, flags = re.M):
	#	reg_var3 = re.findall(r3, lines, flags = re.M)[0]
	#	if reg_var3:
	#		# specifies the index of row cells.			 
	#		cell_index3 = 'E' + str(cell_num)
	#		# output is written in the rows of H
	#		worksheet.write(cell_index3, reg_var3, cell_format)

		#else:
			#worksheet.write(cell_index3, '')

	

	#if re.findall(r4, lines, flags = re.M):
	reg_var4 = lines[lines.find('(')+1:lines.find(')')]
	if reg_var4:
		#specifies the index of row cells.			 
		cell_index4 = 'F' + str(cell_num)
		#output is written in the rows of H
		worksheet.write(cell_index4, reg_var4, cell_format)

	else:
		worksheet.write(cell_index4, '')	

	if re.findall(r5, lines, flags = re.M):
		reg_var5 = re.findall(r5, lines, flags = re.M)[0]
		if reg_var5:
			
			# specifies the index of row cells.			 
			cell_index5 = 'G' + str(cell_num)
			# output is written in the rows of G
			worksheet.write(cell_index5, reg_var5, cell_format)

		else:
			worksheet.write(cell_index5, '')	

	if re.findall(r6, lines, flags = re.M):
		reg_var6 = re.findall(r6, lines, flags = re.M)[0]
		if reg_var6:
			
			# specifies the index of row cells.			 
			cell_index6 = 'H' + str(cell_num)
			# output is written in the rows of H
			worksheet.write(cell_index6, reg_var6, cell_format)

		else:
			worksheet.write(cell_index6, '')	

	 
		

		#if words in lines:
		#	cell_index4 = 'F' + str(cell_num)
		#	worksheet.write(cell_index4, words, cell_format)
		#else:
		#	worksheet.write(cell_index4, '')	

	cell_num += 1
	
			


workbook.close()
file1.close()
#file2.close()
