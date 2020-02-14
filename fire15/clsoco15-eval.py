import xml.dom.minidom, sys



def getKey(item):
    return item[0]


def getText(nodelist):
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)


def handleReuse(docs):
    tuples=[]
    cases = docs.getElementsByTagName("reuse_case")
    for case in cases:
        a=handleCase(case)
        tuples.append(a[0])
        tuples.append(a[1])
    return tuples
	
    
		
def handleCase(case):
    tuple=[[str(case.attributes['source_codeC'].value),str(case.attributes['source_codeJ'].value)],[str(case.attributes['source_codeJ'].value),str(case.attributes['source_codeC'].value)]]
    return tuple


import sys
if len(sys.argv) !=3:
    print "Number of arguments must be 2: QREL_FILE DETECTION_FILE"
    sys.exit()
#print 'Number of arguments:', len(sys.argv), 'arguments.'

	

qrel_file=str(sys.argv[1])
det_file=str(sys.argv[2])

document = open(det_file, 'r').read()

lines = [line.strip() for line in open(qrel_file)]
qrel= [l.split() for l in lines]
gs = [ [rj[1],rj[0]] for rj in qrel ]
gold_standard = gs+qrel
sorted(gold_standard, key=getKey)
	
#Minidom parses the xml file
dom = xml.dom.minidom.parseString(document)
########################################
#Initiating gold standard extraction
result=handleReuse(dom)

relevant_documents= len(gold_standard)
retrieved_documents= len(result)
contados=[]
intersection= 0
for i in result:
   if i in gold_standard and i not in contados and [i[1],i[0]] not in contados:
        intersection=intersection+1
        contados=contados+[i]

intersection=intersection*2
#print gold_standard
precision = intersection/float(retrieved_documents)
recall = intersection/float(relevant_documents)
f1= 2 * ((precision*recall)/(precision+recall))

print "\tF1 = %.3f" % f1, 
print "\tPrec. = %.3f" % precision,
print "\tRec. = %.3f" % recall
