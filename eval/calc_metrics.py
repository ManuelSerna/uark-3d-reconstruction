""" Calculate metrics:

- PSNR
- SSIM
"""
import argparse
import cv2
import numpy
import os
import skimage



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-r", "--readdir", required=True, help="directory containing files (MP4 files)")
	parser.add_argument("-o", "--outfile", required=True, help="output file to write results to")
	args = parser.parse_args()
	
	# Process inputs
	read_dir = args.readdir
	out_file = args.outfile
	
	# Go through input directory
	files = os.listdir(read_dir)

	for file in files:
		ext = file[-3:]
		
		if ext == "mp4":
			print(file)
			#psnr_score = skimage.metrics.peak_signal_noise_ratio(image_true=, image_test=, data_range=)
			#ssim_score = skimage.metrics.structural_similarity(im1=, im2=, data_range=1.0)
		else:
			print(f'[INFO] Ignoring: {file}')

	# Write results



if __name__ == "__main__":
	main()
