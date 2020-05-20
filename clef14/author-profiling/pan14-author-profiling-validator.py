#!/usr/bin/python

import sys, getopt, os
import lxml
from lxml import objectify

IS_VALID=True

################################################################################
# Usage
#
def usage():
	print "Usage: python " + sys.argv[0] + " -i <runDir> -t <truthFile>"
	sys.exit(1)
#
################################################################################
	
################################################################################
# Get Truth IDs
#	
def get_truth_ids(truth_file):
	truth_list = []
	t = [x.strip() for x in file(truth_file)]

	for line in t:
		uid, gender, age = line.split(":::")
		truth_list.append(uid)
	return truth_list
#
################################################################################
	
################################################################################
# Get Answer IDs
#
def get_answer_ids(run_dir):
	answer_ids = []
	answer_filenames = os.listdir(run_dir)
	
	global IS_VALID
	
	# for each file in run directory
	for filename in answer_filenames:
		filepath = run_dir + "/" + filename
		f = open(filepath,"r")
		# read user answers
		try:
			root = objectify.fromstring(f.read())
		
			# check for attribute id
			try:
				root.attrib.get("id")
				# save id
				answer_ids.append(root.attrib.get("id"))
			except KeyError:
				sys.stderr.write('Error: ' + filename + ': Missing attribute: "id"' +"\n")
				IS_VALID=False
				
			# check for attribute gender
			try:
				root.attrib.get("gender")
			except KeyError:
				sys.stderr.write('Error: ' + filename + ': Missing attribute: "gender"' +"\n")	
				IS_VALID=False
		
			# check for attribute age_group
			try:
				root.attrib.get("age_group")
			except KeyError:
				sys.stderr.write('Error: ' + filename + ': Missing attribute: "age_group"' +"\n")
				IS_VALID=False
		
		except lxml.etree.XMLSyntaxError:
			sys.stderr.write("XMLSyntaxError: " + filename +"\n")
			IS_VALID=False
		
		f.close()
	return answer_ids
#
################################################################################

################################################################################
# Check IDs
#
def check_ids(truth_ids, answer_ids):
	global IS_VALID
	
	for uid in sorted(truth_ids):
		if uid not in answer_ids:
			sys.stderr.write("Error: Not processed: " + uid +"\n")
			IS_VALID=False
#
################################################################################
		
################################################################################
# Main
#
def main():

	run_dir=''
	truth_file=''

	# Read command line args
	opts, args = getopt.getopt(sys.argv[1:],"i:t:")

	for o, a in opts:
		if o == '-i':
			run_dir=a
		elif o == '-t':
			truth_file=a
		else:
			usage()
	
	if run_dir == '' or truth_file == '':
		usage()
	
	truth_ids=get_truth_ids(truth_file) 
	answer_ids=get_answer_ids(run_dir) 
	
	check_ids(truth_ids, answer_ids)
			
	sys.stdout.write(str(IS_VALID)+"\n")		
#
################################################################################
	

if __name__ == '__main__':
	main()
	sys.exit(0)	
