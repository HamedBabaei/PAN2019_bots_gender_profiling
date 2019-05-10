import argparse
import os  
import re
import codecs
from xml.dom import minidom

def read_text(path):
	with codecs.open(path , 'r' , encoding="utf-8") as f:
		return f.read()

def prepare_dataset(input_dir , truth_path):
	truth = read_text(truth_path)
	train_bot , train_human_male , train_human_female = [], [] , [] 
	for xml_file in truth.split('\n'):
		if len(xml_file) <= 1:
			continue
		xml_path = os.path.join(input_dir , xml_file.split(":::")[0] + '.xml')
		content = open(xml_path).read()
		tweets = []
		i = 0
		while True:
			i += 1
			start_documents = content.find('<document>')
			end_documents = content.find('</document>')
			tweets.append(' '.join(content[start_documents + 19 : end_documents-3].split()))
			content = content[end_documents+10:]
			if i == 100:
				break
		if xml_file.split(":::")[1] == 'human':
			if xml_file.split(":::")[2] == 'male':
				train_human_male.append(' '.join(tweets))
			else:
				train_human_female.append(' '.join(tweets))
		else:
			train_bot.append(' '.join(tweets))
	return train_bot , train_human_male , train_human_female

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', help="path to input dataset")
	parser.add_argument('-o', '--output', help="path to output directory", default='prepared_dataset')
	args = parser.parse_args()
	if args.input is None:
		parser.print_usage()
		exit()
	return args

def mkdir(path):
	if not os.path.isdir(path):
		os.mkdir(path)

def main():
	args = get_args()
	input_dir = os.path.normpath(args.input)
	out = os.path.normpath(args.output)
	mkdir(out)
	for dir in os.listdir(input_dir):
		print("Working on Language:" , dir )
		out_dir = os.path.join(out , dir)
		mkdir(out_dir)
		truth = os.path.join(input_dir, dir , 'truth.txt')
		bot , male , female  = prepare_dataset( os.path.join(input_dir,dir) , truth)
		with codecs.open(os.path.join(out_dir,'human_male.txt'),'w' , encoding='utf-8' ) as f:
			f.write('\n'.join(male))
		with codecs.open(os.path.join(out_dir,'human_female.txt'),'w' , encoding='utf-8' ) as f:
			f.write('\n'.join(female))
		with codecs.open(os.path.join(out_dir,'bot.txt'),'w' , encoding='utf-8' ) as f:
			f.write('\n'.join(bot))
		print("Bot train-set size: ", len(bot))
		print("Human-male train-set size: ", len(male))
		print("Human-female train-set size: ", len(female))
		print("Dataset saved to ", str(out_dir))
		print("--------------------------------------------------")

main()
