#!/bin/bash

# Fill in your VM credentials here.
user="<user>"
host="<host-pc>"
sshport="<port-ssh>"
token="c114e249d2ecc65e5ac732da507d8496"

# Don't edit inputdir and output.
inputdir="/\/\VBOXSVR/\pan13-training-data/\pan13-sr-mini-2013-03-13"
output="test-run-sr"

# Fill in the path to your software on your VM. Escape "\" with "/\".
# We change to this directory before starting your software (cd $path).
path="<c:/\path/\to/\your/\software>"

# Fill in the command to execute your software.
# The cmd must contain $inputdir, $output and $token
cmd="<cmd-to-start-your-software> $inputdir $output $token"

# This is the command for testing your submission - don't edit.
ssh $user@$host -p $sshport -o StrictHostKeyChecking=no -t "cd $path; rm -rf $output; mkdir $output; $cmd; cd $output; ls" 

