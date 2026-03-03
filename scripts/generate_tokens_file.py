import os
from core.tokenizer import save_tokens_to_file
from utils.datasets import load_data

data_dict_parents = ["data", "train", "PERFECT_CUT_a_z_1_9"]
data_dict_parents_aug = ["data", "train_augmented", "PERFECT_CUT_a_z_1_9_aug_SYNTHETIC"]


data_dict_parents = os.path.join(*data_dict_parents)
data_dict_parents_aug = os.path.join(*data_dict_parents_aug)

data_dict_file = "0annotation.json"
data_dict_file_aug = "synthetic_annotation.json"

data_dict_aug_path = os.path.join(data_dict_parents_aug, data_dict_file_aug)
data_dict_path = os.path.join(data_dict_parents, data_dict_file)

datadict = load_data(data_dict_path)
datadict_aug = load_data(data_dict_aug_path)

words = [word.lower() for word in datadict.data.values()]
words.extend([word.lower() for word in datadict_aug.data.values()])

out_dir_parents = "include"
out_file = "tokens.json"
save_tokens_to_file(words, file_name=out_file, out_dir=out_dir_parents)
