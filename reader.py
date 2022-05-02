import re

import cv2
import numpy as np
import pandas as pd
from pytesseract import image_to_data


class OCRDocumentReader:

    DEFAULT_OCR_TIMEOUT = 300

    def __init__(
        self,
        margin=2,
        bin_threshold=128,
        psm=11,
        lang="swe",
    ):
        self.margin = margin
        self.bin_threshold = bin_threshold
        self.psm = psm
        self.lang = lang

    def read_document(self, document: str, *args, **kwargs):
        img = self.preprocess_image(document)
        df = self.open_ocr_image_to_data(img)
        df = self.prepare_text_dataframes(df, img)
        df = self.postprocessing_text(df, self.margin)
        return df

    @staticmethod
    def preprocess_image(document: str):
        img = cv2.imread(document, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def open_ocr_image_to_data(self, img, output_format="data.frame"):
        data = image_to_data(
            img,
            lang=self.lang,
            config=f"--dpi 290 --oem 3 --psm {self.psm}",
            output_type=output_format,
            timeout=self.DEFAULT_OCR_TIMEOUT,
            pandas_config=None,
        )
        if data.empty:
            raise Exception("Tesseract failed to reading data from document!")
        return data

    def prepare_text_dataframes(self, df, img):
        df = df.dropna(subset=["text"])

        df.loc[:, "text"] = df["text"].str.replace("\n", "")
        df.loc[:, "text"] = df["text"].str.strip()
        df = df[df["text"] != ""]

        df.loc[:, "x_min"] = df["left"]
        df.loc[:, "x_max"] = df["left"] + df["width"]
        df.loc[:, "y_min"] = df["top"] + df["height"]
        df.loc[:, "y_max"] = df["top"]
        df.loc[:, "lit_size"] = df["width"] / df["text"].str.len()

        empties = df.apply(lambda s: self.is_empty(s, img), axis=1)
        df = df[~empties]

        df.loc[:, ["x_min", "x_max", "lit_size"]] /= img.shape[1]
        df.loc[:, ["y_min", "y_max"]] /= img.shape[0] // 1
        df.loc[:, "x_center"] = (df["x_min"] + df["x_max"]) / 2
        df.loc[:, "y_center"] = (df["y_min"] + df["y_max"]) / 2

        return df.reset_index(drop=True)

    @classmethod
    def is_empty(cls, data, img, th=0.95, upper_border=200, bottom_border=10):
        img = img[data["y_max"] : data["y_min"], data["x_min"] : data["x_max"]]
        area = img.shape[0] * img.shape[1]
        cond1 = (img > upper_border).sum() / area
        cond2 = (img < bottom_border).sum() / area
        return cond1 > th or cond2 > th

    @classmethod
    def postprocessing_text(cls, df, margin):
        groups = df.groupby(["page_num", "block_num", "par_num", "line_num"]).groups
        blocks = []

        for row in groups.values():
            blocks.extend(cls.unite_words(df.iloc[row], margin))

        blocks = pd.DataFrame(blocks)[
            ["x_min", "y_min", "x_max", "y_max", "text", "conf"]
        ]
        return blocks.reset_index(drop=True)

    @classmethod
    def unite_words(cls, row, margin=1, filter_pattern=None):
        blocks = []
        block = row.iloc[0].copy()
        for i in range(1, row.shape[0]):
            curr = row.iloc[i]
            lit_size = np.mean([block["lit_size"], curr["lit_size"]])
            skip = False
            if filter_pattern:
                skip = re.search(filter_pattern, curr["text"])
            if not skip and block["x_max"] + lit_size * margin > curr["x_min"]:
                block["text"] = f'{block["text"]} {curr["text"]}'
                block["x_max"] = curr["x_max"]
                block["lit_size"] = lit_size
            else:
                blocks.append(block)
                block = curr.copy()

        block["conf"] = row["conf"].mean()
        blocks.append(block)
        return blocks
