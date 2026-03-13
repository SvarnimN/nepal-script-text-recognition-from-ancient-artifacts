import cv2
import numpy as np
import onnxruntime
import supervision as sv
import matplotlib.pyplot as plt


class ImageLoader:
    """Loads and resizes an image for use in the OCR pipeline."""

    def __init__(self, image_path):
        """
        Initialize the ImageLoader with the path to an image file.

        Args:
            image_path: Path to the image file to load.
        """
        self.TARGET_DIM = 1024

        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.resized_image = None

    def show(self):
        """Display the original image."""
        plt.axis('off')
        plt.imshow(self.image)

    def resize(self):
        """Resize the image to a square using letterboxing to preserve aspect ratio."""
        self.resized_image = sv.letterbox_image(
            image=self.image,
            resolution_wh=(self.TARGET_DIM, self.TARGET_DIM)
        )

    def show_resized(self):
        """Display the resized image. Resizes first if not already done."""
        if self.resized_image is None:
            self.resize()

        plt.axis('off')
        plt.imshow(self.resized_image)


class LineSegmenter:
    """
    Detects and segments individual text lines from an image using a YOLO based
    ONNX line detection model. Each detected line is cropped and letterboxed
    to a fixed size for use in the OCR model.
    """

    def __init__(self, image, model_path):
        """
        Initialize the LineSegmenter with an image and a line detection model.

        Args:
            image     : RGB numpy array of the already resized input image.
            model_path: Path to the YOLO based ONNX line detection model.
        """
        self.CONFIDENCE_THRESHOLD = 0.55

        self.SEGMENT_WIDTH = 508
        self.SEGMENT_HEIGHT = 64
        self.TARGET_DIM = 1024

        self.image = image

        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

        self.line_segments = None

    def _preprocess(self):
        """
        Convert the image to a normalized float32 tensor in NCHW format
        suitable for inference.

        Returns:
            Float32 numpy array of shape (1, 3, TARGET_DIM, TARGET_DIM).
        """
        tensor = self.image.transpose(2, 0, 1)
        tensor = tensor.reshape(1, 3, self.TARGET_DIM, self.TARGET_DIM)
        tensor = (tensor / 255.0).astype(np.float32)
        return tensor

    def _suppress_overlapping_boxes(self, boxes):
        """
        Remove bounding boxes that overlap significantly with the previous
        box. This acts as a simple vertical non-maximum suppression
        to avoid duplicate detections on the same text line.

        Args:
            boxes: List of [x_min, y_min, x_max, y_max] boxes sorted top-to-bottom.

        Returns:
            Filtered list of non-overlapping boxes.
        """
        kept_boxes = [boxes[0]]
        previous_box = boxes[0]

        for current_box in boxes[1:]:
            previous_height = previous_box[3] - previous_box[1]
            vertical_overlap = previous_box[3] - current_box[1]

            if vertical_overlap < 0.5 * previous_height:
                kept_boxes.append(current_box)
                previous_box = current_box

        return kept_boxes

    def _decode_bbox(self, row):
        """
        Convert a line detection model output row from center format (xc, yc, w, h)
        to corner format (x_min, y_min, x_max, y_max).

        Args:
            row: Array-like with at least 4 elements [xc, yc, w, h].

        Returns:
            List [x_min, y_min, x_max, y_max].
        """
        xc, yc, w, h = row[:4]
        return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]

    def _calculate_iou(self, box_a, box_b):
        """
        Calculate the Intersection over Union (IoU) between two bounding boxes
        given in center format (xc, yc, w, h).

        Args:
            box_a: [xc, yc, w, h] for the first box.
            box_b: [xc, yc, w, h] for the second box.

        Returns:
            IoU score as a float in [0, 1].
        """
        x1, y1 = box_a[0] - box_a[2] / 2, box_a[1] - box_a[3] / 2
        w1, h1 = box_a[2], box_a[3]

        x2, y2 = box_b[0] - box_b[2] / 2, box_b[1] - box_b[3] / 2
        w2, h2 = box_b[2], box_b[3]

        inter_x1, inter_y1 = max(x1, x2), max(y1, y2)
        inter_x2, inter_y2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)

        intersection_area = max(0, inter_x2 - inter_x1) * \
            max(0, inter_y2 - inter_y1)
        union_area = w1 * h1 + w2 * h2 - intersection_area

        return intersection_area / union_area if union_area > 0 else 0

    def segment_lines(self):
        """
        Run the line detection model on the image, filter and sort the resulting
        bounding boxes, then crop each detected line into a fixed-size grayscale
        segment.
        """
        tensor = self._preprocess()
        raw_output = self.session.run(
            [self.output_name], {self.input_name: tensor})
        predictions = raw_output[0].transpose()

        detected_boxes = [
            self._decode_bbox(row)
            for row in predictions
            if row[4] >= self.CONFIDENCE_THRESHOLD
        ]

        detected_boxes.sort(key=lambda box: box[1])
        detected_boxes = self._suppress_overlapping_boxes(detected_boxes)

        segments = []
        for box in detected_boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            crop = self.image[y_min:y_max, x_min:x_max]

            crop = sv.letterbox_image(
                image=crop,
                resolution_wh=(self.SEGMENT_WIDTH, self.SEGMENT_HEIGHT)
            )
            crop = crop[:, :, 0]   # convert to single-channel grayscale

            segments.append(crop)

        self.line_segments = segments

    def show_segments(self):
        """Display each detected line segment."""
        if self.line_segments is None:
            self.segment_lines()

        for segment in self.line_segments:
            plt.figure()
            plt.axis('off')
            plt.imshow(segment, cmap='gray')

    def get_segments(self):
        """
        Return the list of cropped line segment arrays.
        Runs segmentation first if it has not been done yet.

        Returns:
            List of grayscale images, one per detected line.
        """
        if self.line_segments is None:
            self.segment_lines()

        return self.line_segments
