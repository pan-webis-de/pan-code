#!/usr/bin/env python


EVALUATION_URL = 'http://webis15.medien.uni-weimar.de/proxy/evaluation'

import os
import sys
import urllib2
import re
import json

def path2ids(path):
    files =  os.listdir(path)
    ids = set()

    for f in files:
       try:
           m = re.search("\w(\d+)\.\w+",f)
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
        print >> sys.stderr, e
        sys.exit(1)

# Main
# ====

if __name__ == "__main__":
    """ Process the commandline arguments. We expect two arguments:
        - The path to the run directory where the log files are located.
        - The access token that was used for the run to be evaluated.
        - One of "pan13-training-ids", "pan13-test1-ids", "pan13-test2-ids".
    """
    if len(sys.argv) == 3:
        inputpath    = sys.argv[1]  # Not used until now.
        token      = sys.argv[2]
        results    = evaluate(token, inputpath)
        sys.stderr.write("""
{{"Queries":"{queries}"}}
{{"Downloads":"{downloads}"}}
{{"Precision":"{precision}"}}
{{"Recall":"{recall}"}}
{{"Queries until 1st Detection":"{queries2}"}}
{{"Downloads until 1st Detection":"{downloads2}"}}
{{"No Detection":"{nodetection}"}}
{{"F-Measure":"{fmeasure}"}}
{{"Details":"http://webis15.medien.uni-weimar.de/pan-logs/pan2013-evaluation-{token}.json"}}
""".format(queries = results["average-queries"],
       downloads = results["average-downloads"],
       precision = results["average-precision"],
       recall = results["average-recall"],
       fmeasure = results["average-f-measure"],
       queries2 = results["average-queries-to-source"],
       downloads2 = results["average-downloads-to-source"],
       nodetection = results["documents-without-sources"],
       token = token))
    else:
        print('\n'.join(["Unexpected number of command line arguments.",
        "Usage: ./pan13-sr-eval.py {rundir} {token}"]))



