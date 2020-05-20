#!/usr/bin/env python

import unittest
import sys
import os
import codecs
from os.path import exists, isdir, isfile, join, getsize, basename
from StringIO import StringIO
from lxml import etree
from glob import glob

class TestTextAlignmentDataset(unittest.TestCase):

    XML_SCHEMA = f = StringIO('''\
<?xml version="1.0" encoding="UTF-8"?>
<xsd:schema xmlns:xsd="http://www.w3.org/2001/XMLSchema">

  <xsd:element name="document" type="document" />

  <xsd:complexType name="document">
    <xsd:sequence>
      <xsd:element name="feature" type="feature" minOccurs="0" maxOccurs="unbounded" />
    </xsd:sequence>
    <xsd:attribute name="reference" type="xsd:anyURI" use="required"/>
  </xsd:complexType>

  <xsd:complexType name="feature">
    <xsd:attribute name="name" type="xsd:string" use="required"/>
    <xsd:anyAttribute namespace="##any" processContents="skip" />
  </xsd:complexType>

</xsd:schema>
''')

    DATASET = "."

    def test_src_directory(self):
        self.run_srcsusp_directory_check("src", "^source-document\d+.txt$")


    def test_susp_directory(self):
        self.run_srcsusp_directory_check("susp", "^suspicious-document\d+.txt$")


    def test_main_pairs_file(self):
        self.run_pairs_file_check(".")
    

    def test_no_extra_files(self):
        files = [f for f in os.listdir(self.DATASET) if isfile(join(self.DATASET, f)) and not f == "pairs"]
        self.assertEqual(len(files), 0, "Unrelated files: " + str(files))


    def test_obfuscation_directories(self):
        dirs = [d for d in os.listdir(self.DATASET) if isdir(join(self.DATASET, d)) and not d == "src" and not d == "susp"]
        self.assertGreater(len(dirs), 0, "No obfuscation directories.")
        for d in dirs:
            self.run_pairs_file_check(d)
            self.run_obfuscation_directory_check(d)


    def run_srcsusp_directory_check(self, dirname, filenameregexp):
        dir = join(self.DATASET, dirname)
        self.assertTrue(exists(dir), "Directory missing: " + dir)
        self.assertTrue(isdir(dir), "Not a directory: " + dir)
        files = [f for f in os.listdir(dir) if isfile(join(dir, f))]
        txts = [f for f in files if f.endswith(".txt")]
        self.assertGreater(len(txts), 0, "No TXT source documents.")
        others = [f for f in files if not f.endswith(".txt")]
        self.assertEqual(len(others), 0, "Unrelated files: " + str(others))
        dirs = [d for d in os.listdir(dir) if isdir(join(dir, d))]
        self.assertEqual(len(dirs), 0, "Unrelated directories; " + str(dirs))
        for f in files:
            self.assertRegexpMatches(f, filenameregexp, "File nam invalid")
            self.assertGreater(getsize(join(dir, f)), 0, "File empty: " + f)
            self.assertTrue(self.try_utf8(join(dir, f)), "Not UTF-8 encoded: " + f)


    def try_utf8(self, path):
        "Returns True on success, or False on failure"
        with open(path, "r") as filein:
            data = filein.read()
            try:
               data.decode('utf-8')
            except UnicodeDecodeError:
               return False
        return True

   
    def run_pairs_file_check(self, dirname):
        path = join(self.DATASET, dirname, "pairs")
        self.assertTrue(exists(path), "File missing: " + path)
        self.run_pairs_file_content_check(dirname, path)

    
    def run_pairs_file_content_check(self, dirname, path):
        self.assertTrue(isfile(path), "Not a file: " + path)
        self.assertGreater(getsize(path), 0, "File empty: " + path)
        if dirname == ".":
            allxmlfiles = [basename(f) for f in glob(self.DATASET + "/**/*.xml")]
        with open(path, "r") as filein:
            lines = filein.readlines()
            for line in lines:
                line = line.strip()
                self.assertRegexpMatches(line, "^suspicious-document\d+.txt source-document\d+.txt$")
                suspdoc = line.split(" ")[0]
                srcdoc = line.split(" ")[1]
                self.run_file_exists_check("susp", suspdoc)
                self.run_file_exists_check("src", srcdoc)
                xmlfile = line.replace(".txt", "").replace(" ", "-") + ".xml"
                if not dirname == ".":
                    self.assertTrue(exists(join(self.DATASET, dirname, xmlfile)), "XML file missing: " + join(dirname, xmlfile))
                else:
                    self.assertTrue(xmlfile in allxmlfiles, "XML file missing in one of the obfuscation directories: " + xmlfile)


    def run_obfuscation_directory_check(self, dirname):
        self.assertRegexpMatches(dirname, "^\d\d(-[a-z0-9]+)+$", "Invalid directory name: " + dirname)
        path = join(self.DATASET, dirname)
        files = [f for f in os.listdir(path) if isfile(join(path, f))]
        xmls = [f for f in files if f.endswith(".xml")]
        self.assertGreater(len(xmls), 0, "Obfuscation directory contains no XML documents: " + dirname)
        others = [f for f in files if not f.endswith(".xml") and not f == "pairs"]
        self.assertEqual(len(others), 0, "Unrelated files: " + str(others))
        dirs = [d for d in os.listdir(path) if isdir(join(path, d))]
        self.assertEqual(len(dirs), 0, "Unrelated directories; " + str(dirs))
        xmlschema_doc = etree.parse(self.XML_SCHEMA)
        self.xmlschema = etree.XMLSchema(xmlschema_doc)
        for xml in xmls:
            self.run_xml_filename_check(dirname, xml)
            self.run_xml_file_content_check(dirname, xml)


    def run_xml_filename_check(self, dirname, xmlname):
        self.assertRegexpMatches(xmlname, "^suspicious-document\d+-source-document\d+.xml$")
        suspdoc = "-".join(xmlname[:-4].split("-")[0:2]) + ".txt"
        srcdoc =  "-".join(xmlname[:-4].split("-")[2:4]) + ".txt"
        self.run_file_exists_check("susp", suspdoc)
        self.run_file_exists_check("src", srcdoc)
        with open(join(self.DATASET, dirname, xmlname)) as filein:
            xmlstring = filein.read()
            self.assertTrue(suspdoc in xmlstring, "File name and file content mismatch: " + join(dirname, xmlname))
            if "source_reference" in xmlstring:
                self.assertTrue(srcdoc in xmlstring, "File name and file content mismatch: " + join(dirname, xmlname))
        pairsline = suspdoc + " " + srcdoc
        pairsfile = join(self.DATASET, dirname, "pairs")
        if exists(pairsfile):
            with open(pairsfile) as filein:
                pairslines = [line.strip() for line in  filein.readlines()]
                self.assertTrue(pairsline in pairslines, "XML file " + join(dirname, xmlname) + " has no corresponding entry in " + join(dirname, "pairs"))
        pairsfile = join(self.DATASET, "pairs")
        if exists(pairsfile):
            with open(pairsfile) as filein:
                pairslines = [line.strip() for line in  filein.readlines()]
                self.assertTrue(pairsline in pairslines, "XML file " + join(dirname, xmlname) + " has no corresponding entry in global pairs file")


    def run_xml_file_content_check(self, dirname, filename):
        path = join(self.DATASET, dirname, filename)
        self.assertTrue(exists(path), "File missing: " + join(dirname, filename))
        with open(path, "r") as filein:
            try:
                xmldoc = etree.parse(filein)
                self.assertTrue(self.xmlschema.validate(xmldoc), "XML does not comply to schema: " + join(dirname, filename))
                self.run_xmldoc_check(dirname, filename, xmldoc)
            except etree.XMLSyntaxError:
                self.fail("Invalid XML syntax: " + join(dirname, filename))


    def run_xmldoc_check(self, dirname, filename, xmldoc):
        documenttag = xmldoc.getroot()
        suspdoc = documenttag.get("reference")
        self.assertRegexpMatches(suspdoc, "^suspicious-document\d+.txt$")
        self.assertTrue(suspdoc.replace(".txt", "") in  filename, "Mismatch between XML filename " + join(dirname, filename) + " and referenced suspicious document " + suspdoc.replace(".txt", ""))
        self.run_file_exists_check("susp", suspdoc)
        for featuretag in documenttag:
           name = featuretag.get("name")
           if name == "about" or name == "md5":
               continue
           srcdoc = featuretag.get("source_reference")
           self.assertRegexpMatches(srcdoc, "^source-document\d+.txt$")
           self.assertTrue(srcdoc.replace(".txt", "") in  filename, "Mismatch between XML filename " + join(dirname, filename) + " and referenced source document " + srcdoc.replace(".txt", ""))
           self.run_file_exists_check("src", srcdoc)
           suspoffset = int(featuretag.get("this_offset"))
           susplength = int(featuretag.get("this_length"))
           srcoffset = int(featuretag.get("source_offset"))
           srclength = int(featuretag.get("source_length"))
           self.run_file_offset_check("susp", suspdoc, suspoffset, susplength, join(dirname, filename))
           self.run_file_offset_check("src", srcdoc, srcoffset, srclength, join(dirname, filename))
           

    def run_file_offset_check(self, dirname, filename, offset, length, xmlfile):
        path = join(self.DATASET, dirname, filename)
        with codecs.open(path, "r", encoding="utf8") as filein:
            text = filein.read()
            self.assertGreaterEqual(offset, 0, "Negative offset ")
            self.assertLessEqual(offset, len(text), 
                "Text too short: offset=" + str(offset) +
                " < utf8textlength=" + str(len(text)) +
                " in " + join(dirname, filename) +
                " as per " + xmlfile)
            self.assertLessEqual(offset + length, len(text),
                "Text too short: offset=" + str(offset) +
                "+length=" + str(length) +
                " < utf8textlength=" + str(len(text)) +
                " in " + join(dirname, filename) +
                " as per " + xmlfile)
            


    def run_file_exists_check(self, dirname, filename):
        path = join(self.DATASET, dirname, filename)
        self.assertTrue(exists(path), "File missing: " + join(dirname, filename))


if __name__ == '__main__':
    if len(sys.argv) == 0:
       print("Usage: python pan15-text-alignment-dataset-validator.py <path/to/corpus>")
    TestTextAlignmentDataset.DATASET = sys.argv.pop()
    unittest.main()

