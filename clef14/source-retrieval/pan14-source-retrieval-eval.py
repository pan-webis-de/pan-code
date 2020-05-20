#!/usr/bin/env python


EVALUATION_URL = 'http://webis15.medien.uni-weimar.de/proxy/evaluation'

import os
import sys
import urllib2
import re
import json


def usage():
    print "\nUsage: " +  sys.argv[0] +  "<inputDir> <outputFile> <token>"
    sys.exit(0)

def path2ids(path):
    files =  os.listdir(path)
    ids = set()

    for f in files:
       try:
           m = re.search("\w(\d+)-{0,1}\w*\.\w+",f)
           ids.add(int(m.group(1)))
       except AttributeError:
            continue
    return list(ids)


# Source Retrieval Evaluation
# ===========================

def evaluate(token, path):
    json_query = u"""
    {{
      "token": "{token}",
      "suspicious-docids": {suspdocids}
    }}
    """.format(suspdocids = str(path2ids(path)), token = token)
    request = urllib2.Request(EVALUATION_URL, json_query)
    request.add_header("Content-Type", "application/json")
    request.add_header("Accept", "application/json")
    request.get_method = lambda: 'POST'
    try:
        response = urllib2.urlopen(request)
        results = json.loads(response.read())
        response.close()
        return results
    except urllib2.URLError as e:
        error_message = e.read()
        print >> sys.stderr, error_message
        sys.exit(1)

# Main
# ====

if __name__ == "__main__":
    """ Process the commandline arguments. We expect two arguments:
        - The path to the run directory where the log files are located.
        - The access token that was used for the run to be evaluated.
        - One of "pan13-training-ids", "pan13-test1-ids", "pan13-test2-ids".
    """
    if len(sys.argv) == 4:
        inputpath    = sys.argv[1]  # Not used until now.
        output_filename   = sys.argv[2]
        token      = sys.argv[3]
        results    = evaluate(token, inputpath)

        output_string = '{\n'+ \
                  '"queries":"%0.1f",\n' % results["average-queries"] + \
                  '"downloads":"%0.1f",\n' % results["average-downloads"] + \
                  '"precision":"%0.5f",\n' % results["average-precision"]+ \
                  '"recall":"%0.5f",\n' % results["average-recall"] + \
                  '"fMeasure":"%0.5f",\n' % results["average-f-measure"]+ \
                  '"queriesUntilFirstDetection":"%0.1f",\n' % results["average-queries-to-source"] + \
                  '"downloadsUntilFirstDetection":"%0.1f",\n' % results["average-downloads-to-source"] + \
                  '"noDetection":"%d",\n' % results["documents-without-sources"] + \
                  '"errors":"%d",\n' % results["errors"] + \
                  '"details":"http://webis15.medien.uni-weimar.de/pan-logs/pan-evaluation-%s.json"\n' % token + '}'
                  

        print output_string
  
        o=open(output_filename, "w")
        o.write(output_string)
        o.close()

        # write prototext file
        json_data = json.loads(output_string)
        prototext_filename= output_filename[:output_filename.rindex(".")]+".prototext"
        prototext_file=open(prototext_filename,"w")

        for i in json_data:
            text = '''measure{
  key  : "%s"
  value: "%s"
}''' % (i, json_data[i])
            prototext_file.write(text+"\n")
        prototext_file.close()	


    else:
        print "Unexpected number of command line arguments."
        usage()



