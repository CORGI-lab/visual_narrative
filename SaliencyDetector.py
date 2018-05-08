# Boilerplate imports.
import tensorflow as tf
import numpy as np
import PIL.Image
from matplotlib import pylab as P
import pickle
import os
import sys
from PIL import Image
import PIL
import cv2
slim=tf.contrib.slim

if not os.path.exists('models/research/slim'):
    print('Please fire: git clone https://github.com/tensorflow/models/ on the command prompt')
old_cwd = os.getcwd()
sys.path.append('models/research/slim/')

# Use either wget or curl depending on your OS.
if not os.path.exists('inception_v3.ckpt'):
    print('You must get the inception_v3 model. Fire the following commands on the terminal')
    #!wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
    print('curl -O http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz')
    print('tar -xvzf inception_v3_2016_08_28.tar.gz')

ckpt_file = './inception_v3.ckpt'

from nets import inception_v3
os.chdir(old_cwd)

import saliency

#boilerplate methods
def LoadImage(image_path):
	img = Image.open(image_path)
	newImg = img.resize((299, 299), PIL.Image.BILINEAR).convert("RGB")
	data = np.array(newImg.getdata())
	ret_img = 2 * (data.reshape((newImg.size[0], newImg.size[1], 3)).astype(np.float32) / 255) - 1
	return ret_img


def ShowImage(im, title='', ax=None):
	if ax is None:
		P.figure()
	P.axis('off')
	im = ((im + 1) * 127.5).astype(np.uint8)
	P.imshow(im)
	P.title(title)
	P.show()


def ShowGrayscaleImage(im, title='', ax=None):
	if ax is None:
		P.figure()
	P.axis('off')

	P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
	P.title(title)
	P.show()


def ShowDivergingImage(grad, title='', percentile=99, ax=None):
	if ax is None:
		fig, ax = P.subplots()
	else:
		fig = ax.figure

	P.axis('off')
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	im = ax.imshow(grad, cmap=P.cm.coolwarm, vmin=-1, vmax=1)
	fig.colorbar(im, cax=cax, orientation='vertical')
	P.title(title)

def scale_things(img):
	vmax = np.percentile(img, 99)
	vmin = np.min(img)

	return np.clip((img - vmin) / (vmax - vmin), 0, 1)

# get cropped image based on maximum saliency region
def GetBoundingBox(gray, color_orig):
	# remove the isolated pixels, it simply runs s 5x5 filter
	def EliminateIsolatedPixels(img):
		# count the number of pixel(i, j) in given neighborhood size sz
		def get_nxn_count(gray, sz, i, j):
			sz = (sz // 2)
			row_min = max(0, i - sz)
			row_max = min(len(gray), i + sz)
			col_min = max(0, j - sz)
			col_max = min(len(gray[0]), j + sz)
			w = np.logical_and(gray[row_min:row_max, col_min:col_max],
							   np.ones((row_max - row_min, col_max - col_min)))
			return sum(sum(w))

		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		for i in range(len(gray)):
			for j in range(len(gray[0])):
				if gray[i][j] != 0:
					s = get_nxn_count(gray, 5, i, j)
					if s < 5: gray[i][j] = 0
		return np.logical_and(gray, np.ones(gray.shape))
	# find the corners of image where saliency starts and ends
	def get_box_coord(img):
		ShowGrayscaleImage(img)
		row_sums = np.sum(img, axis=1)
		col_sums = np.sum(img, axis=0)

		row_b, row_e, col_b, col_e = 0, 0, 0, 0
		for i in range(len(row_sums)):
			if abs(row_sums[i]) > 0:
				row_b = i
				break
		for i in range(len(row_sums)-1, -1, -1):
			if abs(row_sums[i]) > 0:
				row_e = i
				break
		for i in range(len(col_sums)):
			if abs(col_sums[i]) > 0:
				col_b = i
				break
		for i in range(len(col_sums)-1, -1, -1):
			if abs(col_sums[i]) > 0:
				col_e = i
				break
		return row_b, row_e, col_b, col_e

	color = color_orig[:]
	for row in range(color.shape[0]):
		for col in range(color.shape[1]):
			if gray[row][col] < 0.1:
				color[row][col][0] = 0.0
				color[row][col][1] = 0.0
				color[row][col][2] = 0.0

	gray = EliminateIsolatedPixels(color)
	row_b, row_e, col_b, col_e = get_box_coord(gray)
	box = color_orig[row_b:row_e, col_b:col_e]
	return box


with graph.as_default():
	images = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))

	with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
		_, end_points = inception_v3.inception_v3(images, is_training=False, num_classes=1001)

		# Restore the checkpoint
		sess = tf.Session(graph=graph)
		saver = tf.train.Saver()
		saver.restore(sess, ckpt_file)

	# Construct the scalar neuron tensor.
	logits = graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
	neuron_selector = tf.placeholder(tf.int32)
	y = logits[0][neuron_selector]

	# Construct tensor for predictions.
	prediction = tf.argmax(logits, 1)

def GetSalientImage(imagePath):
	im = LoadImage(imagePath)

	# Show the image
	# ShowImage(im)

	# Make a prediction.
	prediction_class = sess.run(prediction, feed_dict={images: [im]})[0]

	#print("Prediction class: " + str(prediction_class))
	# Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
	gradient_saliency = saliency.GradientSaliency(graph, sess, y, images)

	# Compute the vanilla mask and the smoothed mask.
	# vanilla_mask_3d = gradient_saliency.GetMask(im, feed_dict = {neuron_selector: prediction_class})
	smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(im, feed_dict={neuron_selector: prediction_class})

	# Call the visualization methods to convert the 3D tensors to 2D grayscale.
	# vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
	smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
	output = GetBoundingBox(smoothgrad_mask_grayscale, im)
	#ShowImage(output, title='output')

	return output

