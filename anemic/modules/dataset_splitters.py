import logging
import os

from anemic.utils.file_loaders import load_json
from anemic.utils.mapper import ConfigMapper
from anemic.utils.text_loggers import get_logger

logger = get_logger(__name__)


@ConfigMapper.map("dataset_splitters", "caml_official_split")
class CamlOfficialSplit:
    def __init__(self, config):
        logger.debug(
            "Using CAML official split to split data into train-test-val with "
            "the following config: {}".format(config.as_dict())
        )
        self.train_split = load_json(
            os.path.join(config.hadm_dir, config.train_hadm_ids_name)
        )
        self.val_split = load_json(
            os.path.join(config.hadm_dir, config.val_hadm_ids_name)
        )
        self.test_split = load_json(
            os.path.join(config.hadm_dir, config.test_hadm_ids_name)
        )
        self.val_size = getattr(config, "val_size", None)

    def __call__(self, df, hadm_id_col_name):
        train_df = df[df[hadm_id_col_name].isin(self.train_split)]
        val_df = df[df[hadm_id_col_name].isin(self.val_split)]
        test_df = df[df[hadm_id_col_name].isin(self.test_split)]
        if self.val_size is not None:
            if self.val_size < len(val_df):
                val_df = val_df.sample(n=self.val_size, random_state=0)
        return (train_df, val_df, test_df)


@ConfigMapper.map("dataset_splitters", "caml_official_split_limited")
class CamlOfficialSplitLimited(CamlOfficialSplit):
    """Like :class:`CamlOfficialSplit` but optionally limit validation size."""

    def __init__(self, config):
        super().__init__(config)
