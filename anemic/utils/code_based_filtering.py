import logging
import os
import sys
from collections import Counter

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from anemic.utils.file_loaders import load_json, save_json
from anemic.utils.text_loggers import get_logger

logger = get_logger(__name__)


class TopKCodes:
    def __init__(
        self,
        k,
        labels_save_path,
        labels_freq_save_path=None,
        mode="top",
        rare_split_params=None,
    ):
        logger.debug(
            f"Finding {mode}-k codes with the following args: k = {k}, "
            f"labels_save_path = {labels_save_path}, "
            f"labels_freq_save_path = {labels_freq_save_path}"
        )
        self.k = k
        self.mode = mode
        self.top_k_codes = []
        self.labels_save_path = labels_save_path
        self.labels_freq_save_path = labels_freq_save_path
        self.rare_split_params = rare_split_params

    def __call__(self, label_col_name, code_df):
        label_counts = self.find_top_k_codes(label_col_name, code_df)

        save_json(
            {v: k for k, v in enumerate(self.top_k_codes)},
            self.labels_save_path,
        )

        if self.labels_freq_save_path is not None:
            save_json(label_counts, self.labels_freq_save_path)

        if self.k == 0:
            return code_df
        indices_to_delete = []
        top_k_codes_set = set(self.top_k_codes)
        for idx, row in code_df.iterrows():
            filtered_indices = set(row[label_col_name].split(";")).intersection(
                top_k_codes_set
            )
            if len(filtered_indices) > 0:
                row[label_col_name] = ";".join(filtered_indices)
            else:
                indices_to_delete.append(idx)
        code_df.drop(indices_to_delete, inplace=True)
        return code_df

    def find_top_k_codes(self, label_col_name, code_df):
        counts = Counter()

        if self.mode == "rare" and self.rare_split_params is not None:
            train_ids = load_json(
                os.path.join(
                    self.rare_split_params.hadm_dir,
                    self.rare_split_params.train_hadm_ids_name,
                )
            )
            test_ids = load_json(
                os.path.join(
                    self.rare_split_params.hadm_dir,
                    self.rare_split_params.test_hadm_ids_name,
                )
            )

            train_df = code_df[
                code_df[self.rare_split_params.hadm_id_col].isin(train_ids)
            ]
            test_df = code_df[
                code_df[self.rare_split_params.hadm_id_col].isin(test_ids)
            ]

            train_counts = Counter()
            for _, row in train_df.iterrows():
                for label in row[label_col_name].split(";"):
                    train_counts[label] += 1

            test_counts = Counter()
            for _, row in test_df.iterrows():
                for label in row[label_col_name].split(";"):
                    test_counts[label] += 1

            rare_codes = [c for c, cnt in train_counts.items() if 0 < cnt <= 5]
            ranked = [
                (
                    c,
                    test_counts.get(c, 0) / train_counts[c],
                    train_counts[c] + test_counts.get(c, 0),
                )
                for c in rare_codes
            ]
            ranked.sort(key=lambda x: x[1], reverse=True)
            self.top_k_codes = [c for c, _, _ in ranked[: self.k]]

            for c, _, total in ranked:
                if c in self.top_k_codes:
                    counts[c] = total
        else:
            for _, row in code_df.iterrows():
                for label in row[label_col_name].split(";"):
                    counts[label] += 1
            if self.k == 0:
                self.top_k_codes = [code for code, _ in counts.items()]
            else:
                if self.mode == "rare":
                    sorted_counts = sorted(counts.items(), key=lambda x: x[1])
                    self.top_k_codes = [c for c, _ in sorted_counts[: self.k]]
                    counts = dict(sorted_counts[: self.k])
                else:
                    self.top_k_codes = [
                        code for code, _ in counts.most_common(self.k)
                    ]
                    counts = dict(counts.most_common(self.k))

        logger.debug("selected codes: {}".format(self.top_k_codes))
        return counts
