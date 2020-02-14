#!/usr/bin/env python
""" Plagiarism source retrieval example.

    This program provides an example of plagiarism source retrieval
    for the PAN 2013 Plagiarism Detection task.
"""
__author__ = 'Martin Potthast'
__email__ = 'martin.potthast@uni-weimar.de'
__version__ = '1.0'

import codecs
import glob
import os
import random
import re
import simplejson
import sys
import time
import unicodedata
import urllib
import urllib2

CHATNOIR = 'http://webis15.medien.uni-weimar.de/proxy/chatnoir/batchquery.json'
CLUEWEB = 'http://webis15.medien.uni-weimar.de/proxy/clueweb/id/'


# Source Retrieval Example
# ========================

""" The following class implements a naive strategy to retrieve sources for a 
given suspicious document. It is merely intended as an example, not as a
serious solution to the problem.
"""

class Example:
    def process(self, suspdoc, outdir, token):
        """ Run the source retrieval pipeline. """
        # Extract the ID and initiate the log writer.
        self.init(suspdoc, outdir)
        # Read and tokenize the suspicious document.
        text = self.read_file(suspdoc)
        words = self.tokenize(text)
        # Extract queries from the suspicious document.
        open_queries = self.extract_queries(words)
        while len(open_queries) > 0:
            # Retrieve search results for the first query.
            query = open_queries.pop()
            results = self.pose_query(query, token)
            # Log the query event.
            self.log(query)
            # Download the first-ranked result, if any.
            if results["chatnoir-batch-results"][0]["results"] == 0:
                continue;  # The query returned no results.
            download, download_url = self.download_first_result(results, token)
            # Log the download event.
            self.log(download_url)
            # Check the text alignment oracle's verdict about the download.
            self.check_oracle(download)
        # Close the log writer.
        self.teardown()


    def init(self, suspdoc, outdir):
        """ Sets up the output file in which the log events will be written. """
        logdoc = ''.join([suspdoc[:-4], '.log'])
        logdoc = ''.join([outdir, os.sep, logdoc[-26:]])
        self.logwriter = open(logdoc, "w")
        self.suspdoc_id = int(suspdoc[-7:-4])  # Extracts the three digit ID.


    def teardown(self):
        self.logwriter.close()


    def read_file(self, suspdoc):
        """ Reads the file suspdoc and returns its text content. """
        f = codecs.open(suspdoc, 'r', 'utf-8')
        text = f.read()
        f.close()
        return text


    def tokenize(self, text):
        """ Preprocess the suspicious and source document. """
        rx = re.compile("[\w']+", re.UNICODE)
        return rx.findall(text)


    def extract_queries(self, token):
        """ Creates two queries by selecting three random token per query. """
        return [' '.join([random.choice(token),
                          random.choice(token), 
                          random.choice(token)]),
                ' '.join([random.choice(token), 
                          random.choice(token), 
                          random.choice(token)])]


    def pose_query(self, query, token):
        """ Poses the query to the ChatNoir search engine. """
        # Double curly braces are escaped curly braces, so that format
        # strings will still work.
        json_query = u"""
        {{
           "max-results": 5,
           "suspicious-docid": {suspdoc_id},
           "queries": [
             {{
               "query-string": "{query}"
             }}
           ]
        }}
        """.format(suspdoc_id = self.suspdoc_id, query = query)
        json_query = \
            unicodedata.normalize("NFKD", json_query).encode("ascii", "ignore")
        request = urllib2.Request(CHATNOIR, json_query)
        request.add_header("Content-Type", "application/json")
        request.add_header("Accept", "application/json")
        request.add_header("Authorization", token)
        request.get_method = lambda: 'POST'
        try:
            response = urllib2.urlopen(request)
            results = simplejson.loads(response.read())
            response.close()
            return results
        except urllib2.HTTPError as e:
            error_message = e.read()
            print >> sys.stderr, error_message
            sys.exit(1)


    def download_first_result(self, results, token):
        """ Downloads the first-ranked result from a given result list. """
        first_result = results["chatnoir-batch-results"][0]["result-data"][0]
        document_id = first_result["longid"]
        document_url = first_result["url"]
        request = urllib2.Request(CLUEWEB + str(document_id))
        request.add_header("Accept", "application/json")
        request.add_header("Authorization", token)
        request.add_header("suspicious-docid", str(self.suspdoc_id))
        request.get_method = lambda: 'GET'
        try:
           response = urllib2.urlopen(request)
           download = simplejson.loads(response.read())
           response.close()
           return download, document_url
        except urllib2.HTTPError as e:
           error_message = e.read()
           print >> sys.stderr, error_message
           sys.exit(1)
    
    
    def check_oracle(self, download):
        """ Checks is a given download is a true positive source document,
            based on the oracle's decision. """
        if download["oracle"] == "source":
            print "Success: a source has been retrieved."
        else:
            print "Failure: no source has been retrieved."


    def log(self, message):
        """ Writes the message to the log writer, prepending a timestamp. """
        timestamp = int(time.time())  # Unix timestamp
        self.logwriter.write(' '.join([str(timestamp), message]))
        self.logwriter.write('\n')


# Main
# ====

if __name__ == "__main__":
    """ Process the commandline arguments. We expect three arguments: 
        - The path to the directory where suspicious documents are located.
        - The path to the directory to which output shall be written.
        - The access token to the PAN search API.
    """
    if len(sys.argv) == 4:
        suspdir = sys.argv[1]
        outdir  = sys.argv[2]
        token   = sys.argv[3]
        suspdocs = glob.glob(suspdir + os.sep + 'suspicious-document???.txt')
        for suspdoc in suspdocs:
            print "Processing " + suspdoc
            example = Example()
            example.process(suspdoc, outdir, token)
    else:
        print('\n'.join(["Unexpected number of command line arguments.",
        "Usage: ./pan13_source_retrieval_example.py {susp-dir} {out-dir} {token}"]))

