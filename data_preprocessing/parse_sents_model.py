"""
This code parses sentences into dependencies, parse tree, text, and words using Stanford-CoreNLP-Parser.

The parsed sentences are saved in 'cache/parsed_sents/dataset_splitBy/sents.json'.
The 'sents.json' file contains a list of dictionaries, where each dictionary has keys 'sent_id', 'sent', 'parse', 'raw', 'tokens'.
The 'parse' key holds information about the parse tree, dependencies, text, and words.
"""

import os
import os.path as osp

import time
import argparse
import json

import pickle

from corenlp import StanfordCoreNLP


def parse_sents(sents):
    """
    Parses a list of sentences using Stanford CoreNLP.

    Args:
        sents: List of sentences to parse.

    Returns:
        List of parsed results, including parse tree, dependencies, text, and words.
    """
    full_list = []
    corenlp_dir = "/home/users/fschreiber/project/refer-parser2/stanford-corenlp-full-2014-08-27"
    corenlp = StanfordCoreNLP(corenlp_dir)
    
    print("CoreNLP loaded")
    print("Parsing begins...")

    counter = 0
    for entry in sents:
        counter = counter + 1
        
        if counter % 100 == 0:
            print(counter)
        parsed = corenlp.raw_parse(entry)
        
        full_list.append(parsed)
    return full_list


def main(params):
    """
    Main function to parse sentences and save the results to a JSON file.

    Args:
        params: A dictionary containing 'splitBy' and 'partition' keys.

    Returns:
        None
    """
    if not osp.isdir("/home/users/fschreiber/project/parses/" + params["splitBy"] + params["partition"]):
        os.makedirs("/home/users/fschreiber/project/parses/" + params["splitBy"] + params["partition"])

    with open(r"/home/users/fschreiber/project/incremental_pickles/" + params["splitBy"] + params["partition"] + ".p", "rb") as input_file:
        sents = pickle.load(input_file)

    print(sents[0])
    print(len(sents))

    parsed_results = parse_sents(sents)

    # Save the parsed results to a JSON file
    with open(osp.join("/home/users/fschreiber/project/parses/" + params["splitBy"] + params["partition"], 'sents.json'), 'w') as io:
        json.dump(parsed_results, io)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--splitBy', default='unc', help='Split By')
    parser.add_argument("--partition", default="val", help="Test or validation part")
    args = parser.parse_args()
    params = vars(args)

    # Call the main function
    main(params)
