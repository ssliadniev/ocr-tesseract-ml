import json
import re
import warnings
from string import punctuation, whitespace
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cityblock

from reader import OCRDocumentReader

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


class BaseProcess:
    H = 1400
    W = 1000
    x_margin = 0
    y_margin = 0

    def __init__(self, reader=OCRDocumentReader()):

        path = "keywords.json"
        with open(path) as json_file:
            self.keywords = json.load(json_file)

        self.simple_keys = {
            r"^{}".format(k): v for k, v in self.keywords["general_keys"].items()
        }
        self.header_keys = {
            r"{}".format(k): v for k, v in self.keywords["header_keys"].items()
        }
        self.items_borders_keys = self.keywords["items_borders_keys"]
        self.set_margin()

        self.reader = reader

    def set_margin(self, x=0.3, y=0.02):
        # additional width to area of interest
        self.x_margin = x * self.W
        # additional height to area of interest
        self.y_margin = y * self.H

    def find_border_table(
        self, df: pd.DataFrame, items_borders_keys: dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        border_min_value = df[
            df["text"].str.contains("|".join(items_borders_keys["start"]))
        ]
        border_max_value = df[
            df["text"].str.contains("|".join(items_borders_keys["end"]))
        ]
        return border_min_value, border_max_value

    def find_border_values(self, df: pd.DataFrame) -> np.array:
        border_min_value, border_max_value = self.find_border_table(
            df, self.items_borders_keys
        )
        return border_min_value["y_max"].values, border_max_value["y_max"].values

    def filter_invoice_table_lines(
        self,
        df: pd.DataFrame,
        start: list,
        end: list,
        step_top=0,
        step_bottom=0,
        y_field="y_min",
    ) -> pd.DataFrame:
        items_df = []
        for s, e in zip(start, end):
            items = df[(s + step_top <= df[y_field]) & (df[y_field] <= e + step_bottom)]
            items_df.append(items)
        items_df = pd.concat(items_df, axis=0)
        return items_df

    def extract_invoice_table_lines(
        self, df, start_from_page=1, start_border_index=0, end_border_index=0
    ):
        """
        Extract invoice lines fields.

        parameters:
        df [pd.DataFrame]: DataFrame with fields.
        start_from_page [int]: index of page where item lines are started.
        start_border_index [int]: index of the field that will be used for top border.
        end_border_index [int]: index of the field that will be used for bottom border.

        return:
        df [pd.DataFrame] - dataframe of all elements without table elements.
        items_df [pd.DataFrame] - dataframe of table elements.
        """

        border_min_value, border_max_value = self.find_border_values(df)
        i = start_from_page
        start, end = [], []
        while True:
            min_value = border_min_value[
                (border_min_value < i) & (border_min_value > (i - 1))
            ]
            max_value = border_max_value[
                (border_max_value < i) & (border_max_value > (i - 1))
            ]
            min_value, max_value = self._filter_min_max_border(min_value, max_value)

            if min_value.shape[0] > 0 and max_value.shape[0] > 0:
                start.append(min_value.item(start_border_index))
                end.append(max_value.item(end_border_index))
                i += 1
            else:
                break

        if len(start) == 0 or len(end) == 0:
            raise Exception(
                "Border keys are not found! Update items_borders_keys in keywords.json"
            )

        items_df = self.filter_invoice_table_lines(df, start, end)
        df.drop(items_df.index.values.tolist(), inplace=True)
        items_df.reset_index(drop=True, inplace=True)
        return df, items_df

    def _filter_min_max_border(
        self, min_value: np.array, max_value: np.array
    ) -> Tuple[np.array, np.array]:
        return min_value, max_value

    @staticmethod
    def convert_coords(df, height, width):
        df.loc[:, "y_min"] = (df["y_min"] * height).astype(int)
        df.loc[:, "y_max"] = (df["y_max"] * height).astype(int)
        df.loc[:, "x_min"] = (df["x_min"] * width).astype(int)
        df.loc[:, "x_max"] = (df["x_max"] * width).astype(int)
        df.loc[:, "x_center"] = (df["x_min"] + (df["x_max"] - df["x_min"]) / 2).astype(
            int
        )
        df.loc[:, "y_center"] = (df["y_max"] + (df["y_min"] - df["y_max"]) / 2).astype(
            int
        )
        return df

    def translate_value_to_key(self, df_row, translate_dict):
        df_row["ans"] = df_row.get("ans")
        df_rows = []
        for key, value in translate_dict.items():
            if re.search(key, df_row["text"]):
                df_row.loc["keys"] = value
                value_of_key = re.sub(key, "", df_row["text"]).strip()
                if value_of_key.strip(punctuation + whitespace):
                    df_row.loc["ans"] = value_of_key
                df_rows.append(df_row.copy())

        if len(df_rows) > 1:
            row_width = df_row["x_max"] - df_row["x_min"]
            step = row_width / len(df_rows)
            for i, row in enumerate(df_rows):
                row.loc["x_min"] = row.loc["x_min"] + i * step
                row.loc["x_max"] = row["x_min"] + step
                row.loc["x_center"] = (row["x_min"] + row["x_max"]) / 2

            return df_rows

        return [df_row]

    def remove_long_spaces(self, values, space_len=3):
        if space_len:
            pattern = fr"[ ]{{{space_len},}}"
            return self.split_by(values, pattern)
        return values

    def split_by(self, values, pattern):
        values_list = []
        for i, value in values.iterrows():
            splitted = re.split(pattern, value["text"])
            spaces = re.findall(pattern, value["text"])
            if len(splitted) == 1:
                values_list.append(value)
            else:
                values_list.extend(self.create_new_fields(value, splitted, spaces))
        values = (
            pd.concat(values_list, axis=1, sort=False)
            .transpose()
            .reset_index(drop=True)
        )
        return values

    def extract_additional_table_lines(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return df, pd.DataFrame()

    def convert_text_dataframes(self, df: pd.DataFrame):
        df = self.remove_long_spaces(df)
        df.loc[:, "text"] = df["text"].str.replace("  ", " ")
        df, items_df = self.extract_invoice_table_lines(df)
        df_array = []
        for i, row in df.iterrows():
            value = self.translate_value_to_key(row, self.simple_keys)[0]
            df_array.append(value)
        df = pd.concat(df_array, axis=1, sort=False).transpose()
        df = df[df["text"] != "Momsreg.nummer"]
        items_df["keys"] = None
        df = self.convert_coords(df, self.H, self.W)
        items_lines = self.convert_coords(items_df, self.H, self.W).reset_index(
            drop=True
        )
        keys = df.dropna(subset=["keys"]).copy()
        keys = keys.reset_index(drop=True)
        values = df[df["keys"].isna()].copy()
        values = values.reset_index(drop=True)
        values.loc[:, "text"] = values["text"].str.strip(
            punctuation.replace("-", "").replace("+", "") + whitespace
        )
        values = values[values["text"] != ""]
        return keys, values, items_lines

    def find_border_coord(self, coords, keys, fields):
        """
        Find bottom right coordinates of area of interest (area where we are finding nearest keys).

        parameters:
        coords: coordinates of base key.
        keys: dataframe with all keys.
        fields: coordinates type (min, max, center) which will be used to compare keys.

        return:
        bottom right coordinates of area of interest
        """

        idxs = []
        dist = []
        for idx, row in keys.iterrows():
            if (
                int(coords[0]) <= row[fields[0]]
                and int(coords[-1]) <= row[fields[-1]]
                and (
                    int(coords[0]) != row[fields[0]]
                    or int(coords[-1]) != row[fields[1]]
                )
            ):
                d = cityblock(coords, row[fields])
                idxs.append(idx)
                dist.append(d)

        if len(dist) == 0:
            bottom_y = coords[-1] + self.y_margin
            right_x = coords[0] + self.x_margin
        else:
            min_idx = idxs[dist.index(min(dist))]
            if (
                keys.loc[min_idx, fields[1]] < coords[-1] + self.y_margin
                or keys.loc[min_idx, fields[1]] > coords[-1] + self.H * 0.15
            ):
                bottom_y = coords[-1] + self.y_margin
            else:
                bottom_y = keys.loc[min_idx, "y_max"]
            assert (
                keys.loc[min_idx, fields[0]] < coords[0] + self.x_margin
                or keys.loc[min_idx, fields[0]] > coords[0] + self.W * 0.3
            )
            right_x = coords[0] + self.x_margin

        return right_x, bottom_y

    def find_min_dist_idx(self, coords, df, fields):
        idxs = []
        dist = []
        if not isinstance(coords, tuple):
            coords = [coords]
        for idx, row in df.iterrows():
            if (
                int(coords[0]) - self.W * 0.1 <= row[fields[0]]
                and int(coords[-1]) - self.W * 0.01 <= row[fields[-1]]
            ):
                d = cityblock(coords, row[fields])
                idxs.append(idx)
                dist.append(d)
        if len(dist) == 0:
            return -1
        return idxs[dist.index(min(dist))]

    def _filter_data(self, values, border):
        return values[
            (values["x_min"] < border[0] + 0.01 * self.W)
            & (values["y_min"] < border[-1] + 0.01 * self.W)
        ]

    def match_key_value(
        self,
        keys,
        values,
        fields=("x_min", "y_max"),
        border_by_fields=("x_min", "y_min"),
        page=None,
    ):
        """
        Matching keys of simple fields and closest values.

        parameters:
        keys: dataframe with keys.
        values: dataframe with values.
        fields: coords which will be compared.
        border_by_fields: coords of base key which will be used in border calculation.

        return:
        DataFrame with keys and associated values
        """

        keys = self.filter_keys_on_page(keys, page)

        for index, row in keys.iterrows():
            if row["ans"] is None and row["keys"] != "pass":
                coords = (row[fields[0]], row[fields[1]])
                # Looking for the closest keys and finding area of interest
                border = self.find_border_coord(
                    (row[border_by_fields[0]], row[border_by_fields[1]]),
                    keys,
                    list(fields),
                )
                # Filter values that are not in aoi
                values_aoi = self._filter_data(values, border)
                min_idx = self.find_min_dist_idx(coords, values_aoi, list(fields))
                if min_idx != -1:
                    key_value_text = values.loc[min_idx, "text"]
                    values.drop(min_idx, inplace=True)
                else:
                    # If no value found in aoi
                    key_value_text = ""
                keys.loc[index, "ans"] = key_value_text

        keys = keys[keys["keys"] != "pass"].copy()
        keys.index = keys["keys"]
        return keys

    def filter_keys_on_page(self, keys, page=None, field="y_min"):
        keys = keys.copy()
        if page is None:
            pass
        elif page == -1:
            keys = keys.loc[(keys[field] > keys[field].max() - self.H)]
        else:
            keys = keys.loc[
                (keys[field] < (page + 1) * self.H) & (keys[field] > page * self.H)
            ]

        return keys

    def process_duplicates(self, keys, keep="first"):
        result = keys["ans"].to_dict()
        result.update(keys.loc[~keys.index.duplicated(keep=keep), "ans"].to_dict())
        return result

    def _fields_processing(self, keys, values, fields=("x_min", "y_min"), page=None):
        """
        Parsing of simple fields, matching keys and values, preparing the dict result.

        parameters:
        keys: dataframe with keys.
        values: dataframe with values.
        fields: coords which will be compared.

        return:
        dict with matched key-value
        """

        keys = self.match_key_value(keys, values, fields, page=page)
        result = self.process_duplicates(keys)
        return result

    def _find_addr_footer(self, keys, addr_header_coord, default_footer):
        no_addr = keys[keys["keys"] != "invoice.delivery.address"]
        no_addr = no_addr[
            (no_addr["x_center"] <= (addr_header_coord[0] + 0.1 * self.W))
            & (no_addr["x_center"] >= (addr_header_coord[0] - 0.1 * self.W))
            & (no_addr["y_center"] > addr_header_coord[1])
        ].reset_index(drop=True)
        min_idx = self.find_min_dist_idx(addr_header_coord[1], no_addr, ["y_center"])

        if min_idx == -1:
            return default_footer

        return no_addr.loc[min_idx]

    def _get_header_keys(self, values, header_keys=None, reset_index=True):
        if header_keys is None:
            header_keys = self.header_keys

        items_header = []
        for i, value in values.iterrows():
            translated = self.translate_value_to_key(value, header_keys)
            for item in translated:
                if item["keys"] is not None:
                    items_header.append(item)
        items_header = (
            pd.concat(items_header, axis=1)
            .transpose()
            .rename(columns={"keys": "items_header"})
            .sort_values(by="x_center", ascending=True)
            .assign(ans=None)
        )
        if reset_index:
            items_header.reset_index(drop=True, inplace=True)

        return items_header

    def footer_keys_filter(self, page_num, keys, header_edge_y, keys_footer_th):
        footer = keys[keys["y_max"] > header_edge_y + keys_footer_th]
        return footer

    def get_table_fields(
        self,
        keys,
        values,
        start_from_page=1,
        header_keys=None,
        filter_footer_by_x=False,
        footer_th=30,
        keys_footer_th=0,
        pages=None,
    ):
        """
        Process each page of document and extract line items, header and footer.

        parameters:
        keys: keys DataFrame.
        values: values DataFrame.
        start_from_page: page number where table starts.
        header_keys: header of table, default is invoice_line header.
        filter_footer_by_x: filter keys by x to find footer.

        return:
        tuple of items, header and footer
        """

        items_header = self._get_header_keys(values, header_keys)
        if pages is None:
            pages = items_header["y_min"].max() // self.H + 1

        items = []
        header1 = None
        for i in range(pages):
            if i < start_from_page - 1:
                continue
            # Find header
            header = items_header[
                (items_header["y_min"] < self.H * (i + 1))
                & (items_header["y_min"] > self.H * i)
            ]
            x_min = header["x_min"].min()
            x_max = header["x_max"].max()
            # Find footer coords
            footer = None
            for j, item in header.iterrows():
                if abs(header["y_min"].median() - item["y_min"]) > footer_th:
                    footer = item
                    header = header.drop(index=j)
            header_edge_y = header["y_min"].max()
            if footer is None:
                footer = self.footer_keys_filter(i, keys, header_edge_y, keys_footer_th)
                if filter_footer_by_x:
                    footer = footer[
                        (footer["x_max"] > x_min - self.H * 0.01)
                        & (footer["x_min"] < x_max)
                    ]
                footer.reset_index(inplace=True, drop=True)
                if header.shape[0] == 0 and footer.shape[0] == 0:
                    continue
                min_idx = self.find_min_dist_idx(header_edge_y, footer, ["y_center"])
                if min_idx == -1:
                    footer = header.iloc[-1]
                else:
                    footer = footer.loc[min_idx]
            # Extract items
            cond1 = values["y_min"] > header_edge_y
            cond2 = values["y_min"] < footer["y_center"]
            items.append(values[cond1 & cond2])
            header1 = header.reset_index(drop=True)

        items = pd.concat(items, axis=0)
        items.loc[:, "text"] = items["text"].str.strip()
        items = items[items["text"] != ""]

        return items, header1, footer

    def _split_values(
        self,
        items,
        filter_pattern=r"[^\d\., %-]",
        match_pattern=r"(-?\d+(?:[ \.,]\d+)*(?:[ ]?%)?)",
    ):
        """
        Split fields into subfields if they are too close.

        parameters:
        items: source items DataFrame.
        filter_pattern: regexp for values that shouldn't be divided.
        match_pattern: regexp for values that have to be divided.

        return:
        updated items DataFrame.
        """

        items_list = []
        for i, item in items.iterrows():
            if len(re.findall(filter_pattern, item["text"])) > 0 or re.match(
                rf"{match_pattern}[-]{match_pattern}", item["text"]
            ):
                items_list.append(item)
            else:
                fields = re.findall(match_pattern, item["text"])
                cleaned = re.sub(match_pattern, "x", item["text"]).strip()
                spaces = re.findall("[ ]+", cleaned)
                items_list.extend(self.create_new_fields(item, fields, spaces))

        items = pd.concat(items_list, axis=1).transpose().reset_index(drop=True)

        return items

    def create_new_fields(self, field, subfields, spaces=None):
        """
        Create from one fields multiple subfields.

        parameters:
        field: series object that have to be divided.
        subfields: list with text strings.
        spaces: list with spaces between the subfields.

        return:
        list with series objects.
        """

        if spaces is None:
            spaces = [""] * len(subfields)
        else:
            spaces.extend([""] * (len(subfields) - len(spaces)))

        fields_list = []
        space = 0
        step = field["x_min"]
        field_width = field["x_max"] - field["x_min"]
        for j, subfield in enumerate(subfields):
            new_field = field.copy()
            new_field.loc["text"] = subfield
            new_field.loc["x_min"] = step + space
            new_field.loc["x_max"] = new_field["x_min"] + len(
                subfield
            ) * field_width / len(field["text"])
            new_field.loc["x_center"] = (new_field["x_min"] + new_field["x_max"]) / 2
            fields_list.append(new_field)

            step = new_field.loc["x_max"]
            space = len(spaces[j]) * field_width / len(field["text"])

        return fields_list

    def get_product_label(self, header, id_label=None):
        """
        Get header field that associated with item id.

        parameters:
        header: DataFrame with header fields.
        id_label: name of the id column.

        return:
        Series of the field that associated with item id
        """

        if id_label is None:
            return None
        return header.loc[header["items_header"] == id_label].iloc[0]

    def find_indexes(
        self,
        items,
        x_step=5,
        y_step=5,
        min_items=4,
        product_label=None,
        fields=("x_min", "y_min"),
        ascend=True,
    ):
        """
        Find main line for each item.

        parameters:
        items: items DataFrame.
        step: variance.
        min_items: minimum number of fields that can be in the main line.
        product_label: id label from header.
        fields: coords will be compared.

        return:
        DataFrame with index lines.
        """

        cond = True
        items_ids = pd.DataFrame()
        for idx, item in items.iterrows():
            line = items[abs(items[fields[1]] - item[fields[1]]) < y_step].sort_values(
                "x_min", ascending=ascend
            )
            if product_label is not None:
                cond = not line.loc[
                    abs(line[fields[0]] - (product_label[fields[0]])) < x_step
                ].empty
            if (
                cond
                and line.shape[0] >= min_items
                and line.index[0] not in items_ids.index
            ):
                items_ids = items_ids.append(line.iloc[0])

        return items_ids.reset_index(drop=True)

    def separate_items(
        self, items, footer, items_ids, step=5, step_bottom=0, field="y_min"
    ):
        """
        Assign to each line item number.

        parameters:
        items: DataFrame with items
        footer: DataFrame with coords of footer
        items_ids: DataFrame with index lines for each item
        step: extended area of interest
        """

        items.loc[:, "item_number"] = np.nan

        for index, row in items_ids.iterrows():
            cond1 = items[field] >= items_ids.loc[index, field] - step
            try:
                cond2 = items[field] < items_ids.loc[index + 1, field] - step_bottom
            except (KeyError, ValueError):
                cond2 = items["y_min"] < footer["y_center"]
            mask = items[cond1 & cond2].index
            items.loc[mask, "item_number"] = index

        items.sort_values(by=["item_number", "x_min"], inplace=True)

    def match_items_headers(self, item, items_header, coords, abs_coords=False):
        headers = []
        for idx, field in item.iterrows():
            coord1 = items_header[coords[0]] - field[coords[0]]
            coord2 = items_header[coords[1]] - field[coords[1]]
            if abs_coords:
                dist = coord1.abs() + coord2.abs()
            else:
                dist = (coord1 + coord2).abs()
            idx = dist.values.argmin()
            headers.append(items_header.loc[idx, "items_header"])

        item["items_header"] = headers

    def match_items(
        self, items, items_header, items_ids, coords=("x_center", "x_max")
    ) -> list:
        """
        Match values from invoice lines to header keys.

        parameters:
        items [pd.DataFrame]: dataframe with invoice line (coords, text).
        items_header [pd.DataFrame]: dataframe with header keys and coords.
        items_ids [pd.DataFrame]: DataFrame with index lines for each item.
        coords [tuple]: coords fields for comparing fields and matching them with keys.

        return:
        items_list [list]: list of dictionaries for each item.
        """

        items.dropna(subset=["item_number"], inplace=True)
        items_list = []
        for n in items["item_number"].unique():
            item = items[items["item_number"] == n].reset_index(drop=True)
            self.match_items_headers(item, items_header, coords)

            line_1, line_2 = self.item_to_lines(item, items_ids, n)
            line_1["items_header"].replace("pass", np.nan, inplace=True)
            line_1.dropna(subset=["items_header"], inplace=True)

            item_dict = self.lines_processing(line_1, line_2)
            items_list.append(item_dict)

        return items_list

    def item_to_lines(self, item: pd.Series, items_ids: pd.DataFrame, i: int):
        y_max_coord = items_ids.loc[i, "y_max"]
        y_min_coord = items_ids.loc[i, "y_min"]
        field_width = y_min_coord - y_max_coord

        cond = item["y_max"] <= y_max_coord + field_width / 3
        line_1 = item.loc[cond].sort_values("x_center").reset_index(drop=True)
        line_2 = (
            item.loc[~cond]
            .sort_values("y_center", ascending=True)
            .reset_index(drop=True)
        )
        return line_1, line_2

    def lines_processing(self, line_1: pd.DataFrame, line_2: pd.DataFrame) -> dict:
        """
        Converting a dataframe to a dictionary.
        """

        item_dict = {
            name: " ".join(line_1.loc[idxs, "text"])
            for name, idxs in line_1[["items_header"]]
            .groupby(by="items_header")
            .groups.items()
        }
        return item_dict

    def _filter_keys_items(self, keys, items_lines):
        return keys, items_lines

    def parse_table_lines(
        self,
        keys: pd.DataFrame,
        values: pd.DataFrame,
        header_keys: dict,
        filter_footer_by_x=False,
    ) -> list:
        """
        Here the table is found and its elements are processed.

        parameters:
        keys [pd.DataFrame] - dataframe of general keys.
        values [pd.DataFrame] - dataframe of all values.
        header_keys [dict] - a dictionary of headers keys from keywords.json file.
        filter_footer_by_x [bool] - whether to filter the footer by x or not.

        return:
        items_list [list] - a list of all invoice lines.
        """

        keys, values = self._filter_keys_items(keys, values)
        items, header, footer = self.get_table_fields(
            keys, values, header_keys=header_keys, filter_footer_by_x=filter_footer_by_x
        )
        items = self._split_values(items)
        product_label = self.get_product_label(header, id_label=None)
        items_ids = self.find_indexes(
            items, x_step=5, y_step=5, product_label=product_label
        )
        self.separate_items(items, footer, items_ids)
        items_list = self.match_items(items, header, items_ids)
        return items_list

    def parse_invoice_lines(self, keys, values):
        items_list = self.parse_table_lines(keys, values, self.header_keys)
        return {"invoice.invoice_lines": items_list}

    def parse_invoice(self, dataProcess: pd.DataFrame) -> dict:
        """
        Here is the step-by-step processing and extraction of data.
        We get a dataframe of all invoice data and return a dictionary with the required data.

        parameters:
        dataProcess [pd.DataFrame] - dataframe of invoice elements obtained by reading the reader.

        return:
        data [dict] - the dictionary with keys from the keywords.json file and data from the invoice.
        """

        keys, values, items_lines = self.convert_text_dataframes(dataProcess)
        data_simple = self._fields_processing(keys, values)
        data_il = self.parse_invoice_lines(keys, items_lines)
        data = {**data_simple, **data_il}
        return data

    def read_dataframe(self, path: str, *args, **kwargs) -> pd.DataFrame:
        df = self.reader.read_document(path, *args, **kwargs)
        return df

    def process_document(self, path: str) -> dict:
        df = self.read_dataframe(path)
        data = self.parse_invoice(df)
        return data
