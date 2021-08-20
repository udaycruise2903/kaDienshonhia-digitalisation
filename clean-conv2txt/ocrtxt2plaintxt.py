import re 

filename2 = "~/Documents/khasi-dictionary-digitalisation/clean-conv2txt/cleaned_textfile.txt"
filename3 = "~/Documents/khasi-dictionary-digitalisation/clean-conv2txt/processed_file.txt"

file3 = open(filename3, 'w')

main_list = []
third_list = []

file2 = open(filename2, 'r')	
for line in file2.readlines():
	# splits on basis of empty new line in a list of lists.
	split_lines = line.split("\'\n\'") 
	main_list.append(split_lines)

#replace existing empty newlines with tab spaces
second_list = [list(map(lambda x: x if x != "\n" else "\n ", i)) for i in main_list]

#print(second_list)
#print(main_list)


# Convert List of lists to list of Strings
for list in second_list:
	joined_str = (" ".join(list))
	#removes trailing newline after half-column strings.
	processed_str = joined_str.rstrip("\n") 
	#print(processed_str)
	file3.write(processed_str)

#for line in processed_str:
	# removes tab space after headword and its meaning.
	#print(line)
	#split_lines = line.split("\t")
	#third_list.append(split_lines)
	#print(third_list)

file2.close()
file3.close() 
