
# Read Me
This repository contains code and documentation regarding my master thesis: "Evaluating the Incremental Performance of Pre-Trained Models for Reference Resolution", 2023 supervised by Brielen Madueira. It contains offers scripts to replicate the results found in my thesis and guidelines on setting up the data structures.

# Setup
This thesis works on two pretrained models. Please follow the installation instructions provided by the model authors. \
Both models include specifications on dataset installation additional information can be found here: https://github.com/lichengunc/refer \
For this thesis the Refcoco, Refcoco+ and Refcocog were used. The Referit dataset was not used and does not need to be installed.

#### TransVG:
https://github.com/djiajunustc/TransVG  
The pretrained models with the ResNet-101 backbone were used for these thesis

#### ReSC: 
https://github.com/zyang-ur/ReSC  
The large model was used for this thesis

### Incremental preprocessing

After setting up the pretrained models the datasets need to be changed to an incremental version. To do this the data needs to be parsed. 
Please install the parsers 1a) Stanford Parser and 1b) Attention Parser provided here: https://github.com/lichengunc/refer-parser2.\
Alternatively instead of using the Stanford parser the script parse_sents_model in the preprocessing folder can be used. A version of CoreNLP still needs to be installed.\
Then follow the steps in the streamline data notebook to create the incremental dataset and obtain the incremental bounding boxes from the models. 

# Directory Listings
* evaluation metrics: contains scripts for implementing the incremental metrics discussed in the thesis
* evaluation bounding boxes: contains scripts to analyse bounding box behaviour with progressing increments
* evaluation sentences: contains scripts analysing the linguistic information provided
* data preprocessing: contains script to bring data into an incremental format

  




