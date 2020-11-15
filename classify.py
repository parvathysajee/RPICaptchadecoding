#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import string
import random
import argparse
import tflite_runtime.interpreter as tflite
from collections import OrderedDict 

class my_dictionary(dict): 
  
    # __init__ function 
    def __init__(self): 
        self = dict() 
          
    # Function to add key:value 
    def add(self, key, value): 
        self[key] = value 

def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=1)
    return ''.join([characters[x] for x in y])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")
    dict_obj = my_dictionary()


    with open(args.output, 'w',newline='\n') as output_file:
        json_file = open(args.model_name+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        tf_interpreter = tflite.Interpreter(args.model_name+".tflite")
        tf_interpreter.allocate_tensors()

        input_tf = tf_interpreter.get_input_details()
        output_tf = tf_interpreter.get_output_details()
        

        for x in os.listdir(args.captcha_dir):
            # load image and preprocess it
            raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
            rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
            image = numpy.array(rgb_data, dtype=numpy.float32) / 255.0
            (c, h, w) = image.shape
            image = image.reshape([-1, c, h, w])
            tf_interpreter.set_tensor(input_tf[0]['index'],image)
            tf_interpreter.invoke()
            prediction = []
            for output_node in output_tf:
                prediction.append(tf_interpreter.get_tensor(output_node['index']))
            prediction = numpy.reshape(prediction,(len(output_tf),-1))
            output_file.write(x + "," + decode(captcha_symbols, prediction) + "\n")

            print('Classified ' + x)

if __name__ == '__main__':
    main()
