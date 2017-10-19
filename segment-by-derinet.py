#!/usr/bin/env python3

import sys
import unittest
import argparse
from time import strftime

from log import perr
from lexeme import Lexeme
from segmentshandler import SegmentedLoader, SegmentedStorer
from segmentace import Segmentace


def parse_args():
	parser = argparse.ArgumentParser(description="Extract possible segmentations from dictionaries of derivations and inflections and segment corpora from STDIN.", epilog="By default, only lemmas from DeriNet are loaded. Since segmentation of lemmas only is too limited for most applications, you can optionally enable support for segmenting inflected forms by using the --analyzer or --morfflex options. Loading MorfFlex produces the most detailed segmentation, but it is very memory intensive. Using the MorphoDiTa analyzer is cheaper, but requires you to install the 'ufal.morphodita' package prom PyPI and doesn't segment all forms reliably.")
	
	parser.add_argument("derinet", metavar="DERINET.tsv.gz", help="a path to the compressed DeriNet dictionary.")
	#parser.add_argument("morfflex", metavar="MORFFLEX.tab.csv.xz", help="a path to the compressed MorfFlex dictionary.")
	parser.add_argument("-a", "--analyzer", metavar="DICTIONARY.tagger", help="a path to the MorphoDiTa tagger data. When used, will lemmatize the input data before segmenting, thus supporting segmentation of inflected forms.")
	parser.add_argument("-m", "--morfflex", metavar="MORFFLEX.tab.csv.xz", help="a path to the compressed MorfFlex dictionary. When used, will enrich the dictionary with forms in addition to lemmas, thus supporting segmentation of inflected forms. Beware, this makes the program very memory intensive.")
	parser.add_argument("-f", "--from", metavar="FORMAT", dest="from_format", help="the format to read. Available: vbpe, hbpe, spl, hmorph. Default: spl.", default="spl", choices=["vbpe", "hbpe", "spl", "hmorph"])
	parser.add_argument("-t", "--to", metavar="FORMAT", dest="to_format", help="the format to write. Available: vbpe, hbpe, spl, hmorph. Default: vbpe.", default="vbpe", choices=["vbpe", "hbpe", "spl", "hmorph"])
	#parser.add_argument("morpho", metavar="MORPHO", help="a path to the MorphoDiTa morphological resource.")
	
	args = parser.parse_args()
	return args



def process_input(loader, storer, segmenter):
	
	for input_sentence in loader:
		output_sentence = segmenter.segment_sentence(input_sentence)
		
		#perr(output_sentence)
		
		storer.print_sentence(output_sentence)
	
	#perr(Lexeme.morph_change_types)



if __name__ == '__main__':
	#unittest.main()
	perr("Started at %s." % strftime("%c"))
	
	args = parse_args()
	
	derinet_file_name = args.derinet
	morfflex_file_name = args.morfflex
	morpho_file_name = args.analyzer
	
	segmenter = Segmentace(derinet_file_name, morfflex_file_name, morpho_file_name)
	
	perr("Ready to split STDIN at %s." % strftime("%c"))
	
	loader = SegmentedLoader(args.from_format, filehandle=sys.stdin)
	storer = SegmentedStorer(args.to_format, filehandle=sys.stdout)
	
	process_input(loader, storer, segmenter)
	
	perr("Finished at %s." % strftime("%c"))
