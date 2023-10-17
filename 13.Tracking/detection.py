import numpy as np
import os

from PIL import Image
from skimage.transform import resize
from skimage import io
from config import model, VOC_CLASSES, bbox_util
from utils import get_color

import skimage
import torch


def detection_cast(detections):
    """Helper to cast any array to detections numpy array.
    Even empty.
    """
    return np.array(detections, dtype=np.int32).reshape((-1, 5))


def rectangle(shape, ll, rr, line_width=5):
    """Draw rectangle on numpy array.

    rr, cc = rectangle(frame.shape, (ymin, xmin), (ymax, xmax))
    frame[rr, cc] = [0, 255, 0] # Draw green bbox
    """
    ll = np.minimum(np.array(shape[:2], dtype=np.int32) - 1, np.maximum(ll, 0))
    rr = np.minimum(np.array(shape[:2], dtype=np.int32) - 1, np.maximum(rr, 0))
    result = []

    for c in range(line_width):
        for i in range(ll[0] + c, rr[0] - c + 1):
            result.append((i, ll[1] + c))
            result.append((i, rr[1] - c))
        for j in range(ll[1] + c + 1, rr[1] - c):
            result.append((ll[0] + c, j))
            result.append((rr[0] - c, j))

    return tuple(zip(*result))


@torch.no_grad()
def extract_detections(frame, min_confidence=0.6, labels=None):
    """Extract detections from frame.

    frame: numpy array WxHx3
    returns: numpy int array Cx5 [[label_id, xmin, ymin, xmax, ymax]]
    """
    if len(frame.shape) != 3:
        return []
    # Write code here
    # First, convert frame to float and resize to 300x300, convert RGB to BGR
    # then center it with respect to imagenet means
    H = frame.shape[0]
    W = frame.shape[1]
    # imagenet means for BGR
    mean = np.array([103.939, 116.779, 123.68]).reshape(1, 1, 3)

    def image2tensor(image):
        image = image.astype(float)  # convert frame to float
        image = resize(image, (300, 300))
        # image = ... # resize image to 300x300
        image = image[:, :, ::-1]
        # image = ... # convert RGB to BGR
        image -= mean
        image = np.transpose(image, (2, 0, 1))  # torch works with CxHxW images
        tensor = torch.tensor(image.copy()).unsqueeze(0)
        # tensor.shape == (1, channels, height, width)
        return tensor
    input_tensor = image2tensor(frame)

    # Then use image2tensor, model(input_tensor), convert output to numpy
    # and bbox_util.detection_out
    # Use help(...) function to help
    predictions = model(input_tensor.float()).detach().numpy()
    results = bbox_util.detection_out(
        predictions, confidence_threshold=min_confidence)

    # Select detections with confidence > min_confidence
    # hint: you can make it passing min_confidence as
    # a confidence_threshold argument of bbox_util.detection_out

    # If label set is known, use it
    if labels is not None:
        result_labels = results[:, 0].astype(np.int32)
        indeces = [i for i, l in enumerate(
            result_labels) if VOC_CLASSES[l - 1] in labels]
        results = results[indeces]

    # Remove confidence column from result
    if len(results[0]):
        results = np.delete(results[0], 1, axis=1)
        n, m = results.shape
        for i in range(n):
            for j in range(1, m):
                if j % 2 == 1:
                    results[i][j] *= W
                else:
                    results[i][j] *= H

    # Resize detection coords to input image shape.
    # Didn't you forget to save it before resize?

    # Return result
    return detection_cast(results)


def draw_detections(frame, detections):
    """Draw detections on frame.

    Hint: help(rectangle) would help you.
    Use get_color(label) to select color for detection.
    """
    frame = frame.copy()
    for i in range(detections.shape[0]):
        xmin = detections[i][1]
        ymin = detections[i][2]

        xmax = detections[i][3]
        ymax = detections[i][4]

        rr, cc = rectangle(frame.shape, (ymin, xmin), (ymax, xmax))
        frame[rr, cc] = [0, 255, 0]
    return frame


def main():
    dirname = os.path.dirname(__file__)
    frame = Image.open(os.path.join(dirname, 'data', 'test2.jpg'))
    frame = np.array(frame)
    detections = extract_detections(frame)

    frame = draw_detections(frame, detections)

    io.imshow(frame)
    io.show()


if __name__ == '__main__':
    main()
