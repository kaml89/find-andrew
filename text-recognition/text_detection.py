from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2



def decode_predictions(scores, geometry):

	"""Returns tuple containing the bounding box locations of the text (rects)
	 and the probability of that box containing text (confidences)

    Arguments:
    scores -- probabilities of given area containing text
    geometry -- bounding boxes of text region
    """
	
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities) and geometrical data
		# of boxes that may contain text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		
		for x in range(0, numCols):
			# ignore score if it has low probability
			if scoresData[x] < args["min_confidence"]:
				continue
			
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			
			# calculate width and height of the box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
		

			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	
	return (rects, confidences)


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-i", "--image", type=str,
	help="path to input image")
arg_parser.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector")
arg_parser.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
arg_parser.add_argument("-w", "--width", type=int, default=320,
	help="nearest multiple of 32 for resized width")
arg_parser.add_argument("-e", "--height", type=int, default=320,
	help="nearest multiple of 32 for resized height")
arg_parser.add_argument("-p", "--padding", type=float, default=0.0,
	help="amount of padding to add to each border of ROI")
args = vars(arg_parser.parse_args())

# load the input image and grab the image dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(origH, origW) = image.shape[:2]


(newW, newH) = (args["width"], args["height"])
rW = origW / float(newW)
rH = origH / float(newH)

image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]


layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]
# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)


(rects, confidences) = decode_predictions(scores, geometry)

# remove overlapping bounding boxes 
boxes = non_max_suppression(np.array(rects), probs=confidences)

# initialize the list of results
results = []
# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
	# scale the bounding box coordinates based on the respective
	# ratios
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)
	
	# add padding to bounding boxes
	dX = int((endX - startX) * args["padding"])
	dY = int((endY - startY) * args["padding"])

	startX = max(0, startX - dX)
	startY = max(0, startY - dY)
	endX = min(origW, endX + (dX * 2))
	endY = min(origH, endY + (dY * 2))
	
	roi = orig[startY:endY, startX:endX]
	config = ("-l pol --oem 3 --psm 7")
	text = pytesseract.image_to_string(roi, config=config)

	results.append(((startX, startY, endX, endY), text))

# sort the results bounding box coordinates from top to bottom
results = sorted(results, key=lambda r:r[0][1])

for ((startX, startY, endX, endY), text) in results:
	print("OCR TEXT")
	print("========")
	print("{}\n".format(text))

	#text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
	output = orig.copy()
	cv2.rectangle(output, (startX, startY), (endX, endY),
		(0, 255, 0), 2)
	#cv2.putText(output, text, (startX, startY - 20),
	#	cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
	# show the output image
	cv2.imshow("Text Detection", output)
	cv2.waitKey(0)
