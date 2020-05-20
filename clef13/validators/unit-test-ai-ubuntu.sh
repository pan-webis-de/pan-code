#!/bin/bash

# Fill in your VM credentials here.
user="<user>"
host="<host-pc>"
sshport="<port-ssh>"

# Don't edit inputdir and output.
inputdir="/media/pan13-training-data/pan13-ai-mini-2013-03-13"
output="test-run-ai"
outputfilename="answers.txt"

# Fill in the path to your software on your VM.
# We change to this directory before starting your software (cd $path).
path="<path-to-software>"

# Fill in the command to execute your software.
# The cmd must contain $inputdir and $output.
cmd="<cmd-to-start-your-software> $inputdir $output"

# This is the command for testing your submission - don't edit.
ssh $user@$host -p $sshport -o StrictHostKeyChecking=no -t "cd $path; rm -rf $output; mkdir $output; $cmd; cd $output; cat $outputfilename" 

