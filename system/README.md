MVA - Speech & NLP : Lab 2
========================================

Source Code for the second lab of the NLP MVA master's course. 

## Requirements :

Some requirements are needed to run the code

```
pip install -r requirements.txt
```

## Use the System :

### Results on the test split :

To get results on the test split (last 10%) of sequoia-corpus. run :

```
bash run.sh
```
This will run the script. It will takes around 15 minutes to proceed.
The parsed test sentences are saved in the file **evaluation_data.parser_output**.
A sentence with a POS OOG (Out of Grammar) means the algorithm was not able to parse the sentence.

### Results on new sentences :

In order to test the algorithm on new sentences, you must run :
```
bash run.sh --mode eval
```


