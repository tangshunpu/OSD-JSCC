"""Checks for the shared inference interface; no model download or GPU required."""

import unittest
from unittest.mock import patch

import torch

import inference
import osd_jscc_p.inference as progressive_entry
from models import OSDJSCC, OSDJSCCP
from osd_jscc_p.models import OSDJSCCP as compatibility_model


class DummyModel:
    downsample_factor = 128

    def __init__(self):
        self.seen = None

    def __call__(self, image, snr):
        self.seen = image.shape
        return image

    def progressive(self, image, snr, rates):
        self.seen = image.shape
        return {rate: image for rate in rates}


class SharedInterfaceTest(unittest.TestCase):
    def test_shared_model_identity(self):
        self.assertTrue(issubclass(OSDJSCCP, OSDJSCC))
        self.assertIs(OSDJSCCP, compatibility_model)

    def test_model_specific_defaults(self):
        with patch("sys.argv", ["inference.py", "--input", "x.png"]):
            standard = inference.parse_args()
            progressive = progressive_entry.parse_args()
        self.assertEqual(standard.model, "osd-jscc")
        self.assertEqual(standard.checkpoint.name, "osd_jscc_cr96_awgn.pt")
        self.assertEqual(progressive.model, "osd-jscc-p")
        self.assertEqual(progressive.checkpoint.name, "osd_jscc_p_awgn.safetensors")
        with patch("sys.argv", ["inference.py", "--input", "x.png", "--model", "osd-jscc-p", "--rates", "96", "48"]):
            selected = inference.parse_args()
        self.assertEqual(selected.rates, [96, 48])
        self.assertEqual(str(selected.output_dir), "outputs/osd_jscc_p")

    def test_both_paths_pad_and_crop(self):
        image = torch.rand(1, 3, 129, 137)
        standard, progressive = DummyModel(), DummyModel()
        reconstructed = inference.transmit(standard, image, 10, False)
        outputs = inference.transmit_progressive(progressive, image, 10, [96, 48])
        self.assertEqual(standard.seen[-2:], (256, 256))
        self.assertEqual(progressive.seen[-2:], (256, 256))
        self.assertTrue(torch.equal(reconstructed, image))
        self.assertEqual(list(outputs), [96, 48])
        for output in outputs.values():
            self.assertTrue(torch.equal(output, image))
        legacy = progressive_entry.transmit(DummyModel(), image, 10, [96])
        self.assertTrue(torch.equal(legacy[96], image))


if __name__ == "__main__":
    unittest.main()
