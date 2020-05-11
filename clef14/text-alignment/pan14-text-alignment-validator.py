#!/usr/bin/python

import sys, getopt, os
import lxml
from lxml import objectify
from lxml import etree

IS_VALID=True

################################################################################
# Usage
#
def usage():
	print "Usage: python " + sys.argv[0] + " -i <runDir> -t <pairsFile>"
	sys.exit(1)
#
################################################################################
	
################################################################################
# Get Truth
#	
def get_truth_ids(truth_file):
	truth_list = []
	t = [x.strip() for x in file(truth_file)]

	for line in t:
		p1, p2 = line.split(" ")
		truth_list.append(p1[:-4]+"-"+p2[:-4])
	return truth_list
#
################################################################################
	
################################################################################
# Get Answers
#
def get_answer_ids(run_dir):
	answer_ids = []
	answer_filenames = os.listdir(run_dir)
	
	global IS_VALID
	
	# for each file in run directory
	for filename in answer_filenames:
		filepath = run_dir + "/" + filename
		answer_ids.append(filename.replace(".txt","")[:-4])
		f = open(filepath,"r")
		# read user answers
		try:
			root = etree.fromstring(f.read())
			suspicious = root.attrib.get("reference")
			
			# iterate over feature tags
			for feature in root.iterchildren():
				if feature.attrib.get("name") == None:
					sys.stderr.write('Error: ' + filename + ': Missing attribute: "name"' +"\n")	
					IS_VALID=False
				if feature.attrib.get("this_offset") == None:
					sys.stderr.write('Error: ' + filename + ': Missing attribute: "this_offset"' +"\n")	
					IS_VALID=False
				if feature.attrib.get("this_length") == None:
					sys.stderr.write('Error: ' + filename + ': Missing attribute: "this_length"' +"\n")	
					IS_VALID=False
				if feature.attrib.get("source_reference") == None:
					sys.stderr.write('Error: ' + filename + ': Missing attribute: "source_reference"' +"\n")	
					IS_VALID=False
				if feature.attrib.get("source_offset") == None:
					sys.stderr.write('Error: ' + filename + ': Missing attribute: "source_offset"' +"\n")	
					IS_VALID=False
				if feature.attrib.get("source_length") == None:
					sys.stderr.write('Error: ' + filename + ': Missing attribute: "source_length"' +"\n")	
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
	
	for pair in sorted(truth_ids):
		if pair not in answer_ids:
			sys.stderr.write("Error: Not processed: " + pair +"\n")
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

################################################################################
	

if __name__ == '__main__':
	main()
	sys.exit(0)	
