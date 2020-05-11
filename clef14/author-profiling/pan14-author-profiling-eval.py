#!/usr/bin/python

import sys, getopt, os
from lxml import objectify
import json

LANGUAGES = {"english":"en","spanish":"es"}
GENRES = ["blogs","reviews","socialmedia","twitter"]


################################################################################
# Usage
#
def usage():
	print "Usage: python " + sys.argv[0] + " -i <runDir> -t <truthFile> -o <outputFile>"
	sys.exit(1)
#
################################################################################

################################################################################
# Get Language
#	
def get_language(truth_file):
	for i in LANGUAGES.keys():
		if i in truth_file:
			return LANGUAGES[i]
#
################################################################################

################################################################################
# Get Genre
#	
def get_genre(truth_file):
	for i in GENRES:
		if i in truth_file:
			return i
#
################################################################################

	
################################################################################
# Get Truth
#	
def get_truth_dictionary(truth_file):
	truth_dictionary = {}
	t = [x.strip() for x in file(truth_file)]

	for line in t:
		uid, gender, age = line.split(":::")
		truth_dictionary[uid]=(gender,age)
	return truth_dictionary
#
################################################################################
	
################################################################################
# Get Answers
#
def get_answer_dictionary(run_dir):
	answers_dictionary = {}
	answer_filenames = os.listdir(run_dir)
	
	# for each file in run directory
	for filename in answer_filenames:
		filepath = run_dir + "/" + filename
		f = open(filepath,"r")
		# read user answers
		root = objectify.fromstring(f.read())
		answers_dictionary[root.attrib.get("id")] = (root.attrib.get("gender"), root.attrib.get("age_group"))
	
	return answers_dictionary
#
################################################################################
		
	
################################################################################
# Get Accuracy
#
def get_accuracy(truth_dictionary, answers_dictionary):
	correct_age_count = 0 
	correct_gender_count = 0
	correct_both_count = 0
	
	number_of_cases=float(len(truth_dictionary.keys()))
	
	# for each uid
	for uid in truth_dictionary.keys():
		truth = [x.lower() for x in truth_dictionary[uid]]
		if uid in answers_dictionary.keys():
			answer = [x.lower() for x in answers_dictionary[uid]]
		else:
			answer = ["xx","xx"]
		
		# check gender
		if truth[0] == answer[0]:
			correct_gender_count+=1
		
		#check age	
		if truth[1] == answer[1]:
			correct_age_count+=1
		
		# check both
		if truth == answer:
			correct_both_count+=1
			
	results = {}
	results["age"]=correct_age_count/number_of_cases
	results["gender"]= correct_gender_count/number_of_cases
	results["both"]= correct_both_count/number_of_cases
	return results
#
################################################################################

################################################################################
# Main
#
def main():

	run_dir=''
	truth_file=''
	output_file=''

	# Read command line args
	opts, args = getopt.getopt(sys.argv[1:],"i:t:o:")

	for o, a in opts:
		if o == '-i':
			run_dir=a
		elif o == '-o':
			output_file=a
		elif o == '-t':
			truth_file=a
		else:
			usage()
	
	if run_dir == '' or truth_file == '' or output_file == '':
		usage()
	 
	truth=get_truth_dictionary(truth_file) 
	answers=get_answer_dictionary(run_dir) 
	results= get_accuracy(truth,answers)
	language=get_language(truth_file)
	genre=get_genre(truth_file)
	
	output_string = '{\n'+ \
									'"genre":"%s",\n' % genre + \
									'"language":"%s",\n' % language + \
									'"instances":"%d",\n' % len(truth.keys()) + \
									'"instancesDone":"%d",\n' % len(answers.keys()) + \
									'"age":"%0.4f",\n' % results["age"] + \
									'"gender":"%0.4f",\n' % results["gender"] + \
									'"both":"%0.4f"\n' % results["both"] +	'}'
	
	print output_string
	
	o=open(output_file, "w")
	o.write(output_string)
	o.close()
	
	# write prototext file
	json_data = json.loads(output_string)
	prototext_filename= output_file[:output_file.rindex(".")]+".prototext"
	prototext_file=open(prototext_filename,"w")
	
	for i in json_data:
		text = '''measure{
  key  : "%s"
  value: "%s"
}''' % (i, json_data[i])
		prototext_file.write(text+"\n")
	prototext_file.close()	
	
#
################################################################################
	

if __name__ == '__main__':
	main()
	sys.exit(0)	
