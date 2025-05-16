#!/usr/bin/env python
""" Plagiarism detection for near-duplicate plagiarism.

    This program provides the baseline for the PAN 2013 Plagiarism Detection
    task and can be used as outline for a plagiarism detection program.
"""
__author__ = 'Arnd Oberlaender'
__email__ = 'arnd.oberlaender at uni-weimar dot de'
__version__ = '1.1'

import os
import string
import sys
import xml.dom.minidom
from pathlib import Path


# Const
# =====

DELETECHARS = ''.join([string.punctuation, string.whitespace])
LENGTH = 50


# Helper functions
# ================

""" The following functions are some simple helper functions you can utilize
and modify to fit your own program.
"""

def tokenize(text, length):
    """ Tokeniz a given text and return a dict containing all start and end
    positions for each token.
    Characters defined in the global string DELETECHARS will be ignored.

    Keyword arguments:
    text   -- the text to tokenize
    length -- the length of each token
    """
    tokens = {}
    token = []

    for i in range(0, len(text)):
        if text[i] not in DELETECHARS:
            token.append((i, text[i]))
        if len(token) == length:
            ngram = ''.join([x[1].lower() for x in token])
            if ngram not in tokens:
                tokens[ngram] = []
            tokens[ngram].append((token[0][0], token[-1][0]))
            token = token[1:]

    return tokens

def serialize_features(susp, src, features, outdir):
    """ Serialze a feature list into a xml file.
    The xml is structured as described in the readme file of the 
    PAN plagiarism corpus 2012. The filename will follow the naming scheme
    {susp}-{src}.xml and is located in the current directory.
    Existing files will be overwritten.

    Keyword arguments:
    susp     -- the filename of the suspicious document
    src      -- the filename of the source document
    features -- a list containing feature-tuples of the form
                ((start_pos_susp, end_pos_susp),
                 (start_pos_src, end_pos_src))
    """
    impl = xml.dom.minidom.getDOMImplementation()
    doc = impl.createDocument(None, 'document', None)
    root = doc.documentElement
    root.setAttribute('reference', susp)
    doc.createElement('feature')

    for f in features:
        feature = doc.createElement('feature')
        feature.setAttribute('name', 'detected-plagiarism')
        feature.setAttribute('this_offset', str(f[1][0]))
        feature.setAttribute('this_length', str(f[1][1] - f[1][0]))
        feature.setAttribute('source_reference', src)
        feature.setAttribute('source_offset', str(f[0][0]))
        feature.setAttribute('source_length', str(f[0][1] - f[0][0]))
        root.appendChild(feature)

    doc.writexml(open(outdir + susp.split('.')[0] + '-'
                      + src.split('.')[0] + '.xml', 'w'),
                 encoding='utf-8')


# Plagiarism pipeline
# ===================

""" The following class implement a very basic baseline comparison, which
aims at near duplicate plagiarism. It is only intended to show a simple
pipeline your plagiarism detector can follow.
Replace the single steps with your implementation to get started.
"""

class Baseline:
    def __init__(self, susp_id, susp_text, src_id, src_text, outdir):
        self.susp_text = susp_text
        self.src_text = src_text
        self.susp_id = susp_id
        self.src_id = src_id
        self.output = self.susp_id + '-' + self.src_id + '.xml'
        self.detections = None
        self.outdir=outdir

    def process(self):
        """ Process the plagiarism pipeline. """
        # if not os.path.exists(self.output):
        #    ...
        self.preprocess()
        self.detections = self.compare()
        self.postprocess()

    def preprocess(self):
        """ Preprocess the suspicious and source document. """
        # TODO: Implement your preprocessing steps here.
        self.susp_text = self.susp_text
        self.tokens = tokenize(self.susp_text, LENGTH)

        self.src_text = self.src_text

    def compare(self):
        """ Test a suspicious document for near-duplicate plagiarism with regards to
        a source document and return a feature list.
        """

        #TODO: Implement your comparison here and replace the following
        #      algorithm with your own.

        detections = []
        skipto = -1
        token = []
        for i in range(0, len(self.src_text)):
            if i > skipto:
                if self.src_text[i] not in DELETECHARS:
                    token.append((i, self.src_text[i]))
                if len(token) == LENGTH:
                    ngram = ''.join([x[1].lower() for x in token])
                    if ngram in self.tokens:
                        d = ((token[0][0],token[-1][0]),
                             (self.tokens[ngram][0][0],
                              self.tokens[ngram][0][1]))
                        for t in self.tokens[ngram]:
                            start_src = token[0][0]
                            start_susp = t[0]
                            while (start_susp < len(self.susp_text) and
                                   start_src < len(self.src_text) and
                                   self.src_text[start_src] == self.susp_text[start_susp]):
                                start_susp = start_susp + 1
                                start_src = start_src + 1
                                while (start_susp < len(self.susp_text) and
                                       self.susp_text[start_susp] in DELETECHARS):
                                    start_susp = start_susp + 1
                                while (start_src < len(self.src_text) and
                                       self.src_text[start_src] in DELETECHARS):
                                    start_src = start_src + 1
                            if (start_src - 1) - token[0][0] > d[0][1] - d[0][0]:
                                d = ((token[0][0], start_src), (t[0], start_susp))
                        detections.append(d)
                        skipto = d[0][1]
                        if skipto < len(self.src_text):
                            token = [(skipto, self.src_text[skipto])]
                        else:
                            break
                    else:
                        token = token[1:]

        print(len(detections))

        return detections

    def postprocess(self):
        """ Postprocess the results. """
        # TODO: Implement your postprocessing steps here.
        serialize_features(self.susp_id, self.src_id, self.detections, self.outdir)

# Main
# ====

if __name__ == "__main__":
    """ Process the commandline arguments. We expect two arguments: The path or TIRA dataset id
    pointing to the corpus (containing a pairs file file and the actual source
    and suspicious documents) and the path where the outputs should be stored.
    """

    if len(sys.argv) == 3:
        corpus_dir = sys.argv[1]
        outdir = sys.argv[2]
        if outdir[-1] != "/":
            outdir+="/"
        Path(outdir).mkdir(exist_ok=True, parents=True)
        from tira.io_utils import dataset_as_iterator

        for i in dataset_as_iterator(corpus_dir, 'text-alignment-corpus'):
            baseline = Baseline(
                susp_id=i['suspicious_document_id'],
                susp_text=i['suspicious_document_text'],
                src_id=i['source_document_id'],
                src_text=i['source_document_text'],
                outdir=outdir
            )
            baseline.process()

    else:
        print('\n'.join(["Unexpected number of commandline arguments.",
                         "Usage: ./pan12-plagiarism-text-alignment-example.py {corpus-dir} {out-dir}"]))
