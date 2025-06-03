# convert_to_onnx.py

import torch
import torchvision.models as models
from pytorch_model import Classifier, BasicBlock
import onnx
import os

def convert_model():
    # Step 1: Initialize the custom model with same architecture as ResNet18
    model = Classifier(BasicBlock, [2, 2, 2, 2], num_classes=1000)

    # Step 2: Load pretrained ResNet18 weights from torchvision
    resnet18 = models.resnet18(pretrained=True)
    model.load_state_dict(resnet18.state_dict(), strict=False)  # allow strict=False in case your classifier head differs

    # Step 3: Set to eval mode
    model.eval()

    # Step 4: Dummy input for ONNX export
    dummy_input = torch.randn(1, 3, 224, 224)

    # Step 5: Export the model
    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
        do_constant_folding=True
    )

    print("✅ ONNX model exported to model.onnx")

if __name__ == "__main__":
    convert_model()
