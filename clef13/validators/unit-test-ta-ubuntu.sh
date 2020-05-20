#!/bin/bash

# Fill in your VM credentials here.
user="<user>"
host="<host-pc>"
sshport="<port-ssh>"

# Don't edit inputdir and output.
inputdir="/media/pan13-training-data/pan13-ta-mini-2013-03-13"
output="test-run-ta"

# Fill in the path to your software on your VM.
# We change to this directory before starting your software (cd $path).
path="<path-to-software>"

# Fill in the command to execute your software.
# The cmd must contain $inputdir/pairs, $inputdir/src, $inputdir/susp and $output.
cmd="<cmd-to-start-your-software> $inputdir/pairs $inputdir/src $inputdir/susp $output"

# This is the command for testing your submission - don't edit.
ssh $user@$host -p $sshport -o StrictHostKeyChecking=no -t "cd $path; rm -rf $output; mkdir $output; $cmd; cd $output; ls" 

