#!/usr/bin/python

import sys, getopt, os

IS_VALID=True

################################################################################
# Usage
#
def usage():
	print "Usage: python " + sys.argv[0] + " -i <answersFile> -t <truthFile>"
	sys.exit(1)
#
################################################################################
	
################################################################################
# Get Truth IDs
#	
def get_truth_ids(truth_file):
	t =  [x.rstrip() for x in file(truth_file).readlines()]
	# dirty hack [-5:], fixes encoding problem
	return [x.split(" ")[0][-5:] for x in t]
#
################################################################################
	
################################################################################
# Get Answer IDs
#
def get_answer_ids(answers_file):
	answers = [x.rstrip() for x in file(answers_file).readlines()]
	answer_ids = []
	for a in answers:
		try:
			uid, score = a.split(" ");	
			answer_ids.append(uid)
			try:
				s = float(score)
			except:
				sys.stderr.write('Error: ' + answers_file + ': non float value \n')	
				IS_VALID=False
		except:
			sys.stderr.write('Error: ' + answers_file + ': not space separated \n')	
			IS_VALID=False
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
