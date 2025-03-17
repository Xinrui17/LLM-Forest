import numpy as np
import os
import pickle
import argparse
import inspect
import logging
import sys
import random
import torch
from graph_search import neighbor_search 
from llm_imputation import llm_imputation
from preprocess import preprocess
from postprocess import postprocess


def parse_global_args(parser):
    parser.add_argument("--data_path", type = str, default = "/data/", help="path to dataset")
    parser.add_argument("--model_name", type = str, default="gpt", help="gpt or mixtral")
    parser.add_argument("--dataset", type=str, default="house", help="dataset name")
    parser.add_argument("--num_round", type=int, default=3, help="number of trees in the forest")
    parser.add_argument("--group_size", type=2, help="size to merge graphs")
    parser.add_argument("--num_neighbors", type=str, help="number of neighbors in the context")
    return parser
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_global_args(parser)
    args = parser.parse_args()
    preprocess(args.dataset, args.data_path)
    neighbor_search(args)
    for round_id in range(args.num_round):
        llm_imputation(args.model_name, args.dataset, round_id)
    postprocess(args.dataset, args.data_path, args.num_round)
    
    