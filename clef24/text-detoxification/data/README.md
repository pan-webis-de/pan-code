# Instructions for CodaLab participants

The files `sample_submission_dev.tsv` and `sample_submission_test.tsv` are example submissions for **Development** and **Test** phases of the shared task, respectively. In these files the column `neutral_sentence` is the copy of the `toxic_sentence` column. 

You can use these example submission to refactor yours. When submitting, please make sure that:
1. You are submitting `.tsv` file with `\t` as a separator. In pandas you can ensure the saving format by using `.to_csv("your_submission.tsv", sep="\t", index=False)` 
2. There are **no empty lines** in your submission file. If you are making predictions for only one or several languages, replace the respective rows in `neutral_sentence` column with your predictions and leave others untouched.
3. You are submitting a zipped file. Example for zipping: `zip my_submission.zip my_submission.tsv`
   