# Random Baseline for the shared task on Profiling Cryptocurrency Influencers with Few-shot Learning at PAN23
For each subtask you have a available thee file. 
1. random-baseline-subtask-.py script makes random prediction and and save the predictions in the TIRA format.  
2. Dockerfile is a text file that contains the instructions that you would execute on the command line to create an image.
3. README.md

# Create docker imagen and Tira submission steps

For a software submission on tira, you can dockerize the trained model with the given dockerfile.

1. After training the model, you need build docker image. 
2. Upload the image to a public docker repository, i.e. dockerhub. 
3. Create a new docker software on TIRA.
