# test.py
import unittest
from custom_model import ImagePreprocessor, ONNXModel
import numpy as np
from labels import label_map



class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = ONNXModel("model.onnx")
        self.preprocessor = ImagePreprocessor()

    def test_tench_image(self):
        input_tensor = self.preprocessor.preprocess("n01440764_tench.jpeg")
        class_id, _ = self.model.predict(input_tensor)
        self.assertEqual(class_id, 0)

    def test_mud_turtle_image(self):
        input_tensor = self.preprocessor.preprocess("n01667114_mud_turtle.JPEG")
        class_id, _ = self.model.predict(input_tensor)
        self.assertEqual(class_id, 35)

    def test_output_shape(self):
        input_tensor = self.preprocessor.preprocess("n01440764_tench.jpeg")
        _, probs = self.model.predict(input_tensor)
        self.assertEqual(len(probs[0]), 1000)
print("Tench image prediction:", label_map.get(class_id, "unknown"))
if __name__ == '__main__':
    unittest.main()