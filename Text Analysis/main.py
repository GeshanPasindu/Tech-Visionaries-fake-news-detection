import pandas as pd
import os
import subprocess
import nltk

os.system("python preprocessing.py")
os.system("python feature_extraction.py")
os.system("python sentiment_analysis.py")
os.system("python bert_training.py")
os.system("python evaluate_model.py")
os.system("python predict.py")


