#!/usr/bin/env python3

import os
import numpy
import random
import string
import cv2
import argparse
import captcha.image
import re
import codecs
import logging

def main():
    logging.basicConfig(filename='generate.log', filemode='w',level=logging.INFO,format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logging.info('This will get logged to a file')
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument('--length', help='Length of captchas in characters', type=int)
    parser.add_argument('--count', help='How many captchas to generate', type=int)
    parser.add_argument('--output-dir', help='Where to store the generated captchas', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.width is None:
        print("Please specify the captcha image width")
        exit(1)

    if args.height is None:
        print("Please specify the captcha image height")
        exit(1)

    if args.length is None:
        print("Please specify the captcha length")
        exit(1)

    if args.count is None:
        print("Please specify the captcha count to generate")
        exit(1)

    if args.output_dir is None:
        print("Please specify the captcha output directory")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    captcha_generator = captcha.image.ImageCaptcha(width=args.width, height=args.height)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Generating captchas with symbol set {" + captcha_symbols + "}")
    numberOfExistingFiles = 0
    if not os.path.exists(args.output_dir):
        print("Creating output directory " + args.output_dir)
        os.makedirs(args.output_dir)
    else:
        existingFilesList = os.listdir(args.output_dir)
        numberOfExistingFiles = len(existingFilesList)

    NoOfFilesToGen = args.count - numberOfExistingFiles

    if NoOfFilesToGen <= 0 :
        print('Files are already available in the directory')

    for i in range(NoOfFilesToGen):
        random_str = ''.join([random.choice(captcha_symbols) for j in range(args.length)])
        file_name = codecs.encode(random_str.encode(), "hex")
        image_path = os.path.join(args.output_dir, file_name.decode("ASCII")  +'.png')
        if os.path.exists(image_path):
            version = 1
            while os.path.exists(os.path.join(args.output_dir, file_name.decode("ASCII") + '_' + str(version) + '.png')):
                version += 1
            image_path = os.path.join(args.output_dir, file_name.decode("ASCII") + '_' + str(version) + '.png')

        image = numpy.array(captcha_generator.generate_image(random_str))
        logging.info('Generating image '+random_str)
        cv2.imwrite(image_path, image)

if __name__ == '__main__':
    main()
