#!/bin/bash

# Fill in your VM credentials here.
user="<user>"
host="<host-pc>"
sshport="<port-ssh>"

# Don't edit inputdir and output.
inputdir="/\/\VBOXSVR/\pan13-training-data/\pan13-ap-mini-2013-03-13"
output="test-run-ap"

# Fill in the path to your software on your VM. Escape "\" with "/\".
# We change to this directory before starting your software (cd $path).
path="<c:/\path/\to/\your/\software>"

# Fill in the command to execute your software.
# The cmd must contain $inputdir and $output.
cmd="<cmd-to-start-your-software> $inputdir $output"

# This is the command for testing your submission - don't edit.
ssh $user@$host -p $sshport -o StrictHostKeyChecking=no -t "cd $path; rm -rf $output; mkdir $output; $cmd; cd $output; ls" 

