
# Read Me
This repository hold contains code and documentation regarding my master thesis: "Evaluating the Incremental Performance of Pre-Trained Models for Reference Resolution", 2023 superivsed by Brielen Madueira. It contains offers scripts to replicate the results found in my thesis. And guidelines on setting up the datastrucutres.

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
To obtain the incremental datasets used for this thesis. Please follow the steps provided in the data preprocessing folder.\
To do this the parsers 1a) (Stanford Parser) and 1b) (Attention Parser) provided here: https://github.com/lichengunc/refer-parser2 need to be installed.\
Alternativley instead of using the Stanford parser the script " XYZ " in the preprocessing folder can be used.

# Directory Listings




