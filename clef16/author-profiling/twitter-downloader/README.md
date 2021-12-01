================================================================================
PAN-AP 2016 Twitter Contents Downloader Tool README - v1.0 - April 26th, 2016
================================================================================


The TwitterDownloader is a Java program that allows downloading the 
texts of tweets using HTML requests. 


1.- How to use:

The tool needs a Java JRE 1.5 or higher installed in 
the operating system (available at http://www.java.com/download). To run the tool execute: 

	java -jar TwitterDownloader.jar -data path_to_dataset
		
	
Valid options are: 

	--data 		Reads all the files in the input folder and download 
						the text of each tweet. Result is written in the original file, 
						populating the empty section <document>. 
						The input format should be the PAN-AP 2013 xml format. 
	
	
The tool includes the resume option at file level. The toool downloads around 1,500
tweets per minute.

An example:

	java -jar TwitterDownloader.jar --data "/tmp/pan16-author-profiling-training-corpus-2016-02-29/pan16-author-profiling-training-dataset-english-2016-02-27/"

2.- PAN-AP 2016 xml format

Each xml file contains a series of document lines with the url to the Tweet and an empty space for the content that will be populated with the 
downloader tool:

<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<author url="https://twitter.com/USERNAME">
	<documents count="N">
		<document id="STATUSID" url="https://twitter.com/USERNAME/status/STATUSID">THIS EMPTY SPACE WILL BE POPULATED BY THE DOWNLOADER</document> 
		...
	</documents>
</author>

		
An example: 

<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<author url="https://twitter.com/kicorangel">
	<documents count="2">
		<document id="435361001790140416" url="https://twitter.com/kicorangel/status/435361001790140416"></document>
		<document id="434291330202615808" url="https://twitter.com/kicorangel/status/434291330202615808"></document>
	</documents>
</author>

3.- Output format 

The output of the system is the same input files with the document section populated: 

<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<author url="https://twitter.com/USERNAME">
	<documents count="N">
		<document id="STATUSID" url="https://twitter.com/USERNAME/status/STATUSID"><![CDATA[CONTENTS OF THE TWEET]]</document> 
		...
	</documents>
</author>
			
An example: 

<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<author url="https://twitter.com/kicorangel">
	<documents count="2">
		<document id="435361001790140416" url="https://twitter.com/kicorangel/status/435361001790140416">
<![CDATA[Nueva entrada en mi Blog &quot;Hablemos de I+D&quot;: Is the NSA spying on me?¿Me está espiando la NSA? <a href="http://t.co/UqlNkJhKFn" rel="nofollow" dir="ltr" data-expanded-url="http://www.kicorangel.com/?p=954" class="twitter-timeline-link" target="_blank" title="http://www.kicorangel.com/?p=954" ><span class="tco-ellipsis"></span><span class="invisible">http://www.</span><span class="js-display-url">kicorangel.com/?p=954</span><span class="invisible"></span><span class="tco-ellipsis"><span class="invisible">&nbsp;</span></span></a>]]></document>
		<document id="434291330202615808" url="https://twitter.com/kicorangel/status/434291330202615808"><![CDATA[Nueva entrada en mi Blog &quot;Hablemos de I+D&quot;: Author Profiling at PAN 2014Author Profiling en PAN 2014 <a href="http://t.co/ZhKGQClIBK" rel="nofollow" dir="ltr" data-expanded-url="http://www.kicorangel.com/?p=943" class="twitter-timeline-link" target="_blank" title="http://www.kicorangel.com/?p=943" ><span class="tco-ellipsis"></span><span class="invisible">http://www.</span><span class="js-display-url">kicorangel.com/?p=943</span><span class="invisible"></span><span class="tco-ellipsis"><span class="invisible">&nbsp;</span></span></a>]]></document>
	</documents>
</author> 

