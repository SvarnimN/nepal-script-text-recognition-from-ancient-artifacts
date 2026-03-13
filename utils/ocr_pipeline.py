import numpy as np
import pandas as pd
import cv2
import onnxruntime


class OCRPipeline:
    """
    End-to-end OCR pipeline that takes pre-segmented line images, runs them
    through an CRNN based ONNX recognition model, and decodes the output into
    Nepal Script Unicode text using CTC decoding.
    """

    def __init__(self, line_segments, model_path, charset_path):
        """
        Initialize the OCR pipeline.

        Args:
            line_segments: List of 2D grayscale images, one per text line.
            model_path   : Path to the ONNX OCR recognition model.
            charset_path : Path to the CSV file containing the character set.
                           Expected to have a column named 'Nepal' with Unicode
                           codepoints for each character.
        """
        self.INPUT_WIDTH = 508
        self.INPUT_HEIGHT = 64

        self.line_segments = line_segments

        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        charset_df = pd.read_csv(charset_path)
        self.codepoints = charset_df["Nepal"].values.tolist()

    def _indices_to_text(self, indices):
        """
        Convert a sequence of character indices to a Unicode string.

        Decoding stops when the CTC blank token is encountered.
        The blank token is defined as the index equal to the vocabulary size.

        Args:
            indices: 1D numpy array of integer character indices.

        Returns:
            Decoded Unicode string.
        """
        characters = []
        blank_token_index = len(self.codepoints)

        for idx in indices.astype(int):
            if idx == blank_token_index:
                break
            characters.append(chr(self.codepoints[idx]))

        return ''.join(characters)

    def _ctc_decode(self, model_output):
        """
        Apply greedy CTC decoding to a single model output sequence.

        Collapses repeated labels and removes blank tokens to produce
        the most likely character sequence.

        Args:
            model_output: Model output array of shape (1, timesteps, vocab_size).

        Returns:
            Decoded string for the input sequence.
        """
        sequence = model_output[0]
        blank_token_index = len(self.codepoints)
        previous_label = blank_token_index

        decoded_indices = []
        for timestep in sequence:
            current_label = np.argmax(timestep)
            if current_label != blank_token_index and current_label != previous_label:
                decoded_indices.append(current_label)
            previous_label = current_label

        return self._indices_to_text(np.array(decoded_indices))

    def _preprocess_segment(self, segment):
        """
        Prepare a single line segment for model inference.

        Transposes the image to match the model's expected input orientation,
        reshapes to (1, width, height, 1), and normalizes pixel values to [0, 1].

        Args:
            segment: 2D grayscale numpy array of shape (height, width).

        Returns:
            Float32 numpy array of shape (1, INPUT_WIDTH, INPUT_HEIGHT, 1).
        """
        tensor = cv2.transpose(segment).astype(np.float32)
        tensor = tensor.reshape(1, self.INPUT_WIDTH, self.INPUT_HEIGHT, 1)
        tensor = tensor / 255.0
        return tensor

    def recognize_text(self):
        """
        Run OCR inference on all line segments and decode the results.

        Each segment is preprocessed, passed through the recognition model,
        and decoded using greedy CTC decoding.

        Returns:
            List of decoded text strings, one per input line segment.
        """
        recognized_lines = []

        for segment in self.line_segments:
            tensor = self._preprocess_segment(segment)
            raw_output = self.session.run(
                [self.output_name], {self.input_name: tensor})
            decoded_text = self._ctc_decode(raw_output[0])
            recognized_lines.append(decoded_text.strip())

        return recognized_lines
