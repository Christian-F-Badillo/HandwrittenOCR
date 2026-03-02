import os
import click
from utils.datasets import HandwrittenData, load_data
from utils.preprocessing import process_img


@click.command()
@click.option("-i", default=0, type=int, help="Set the index of the data sample")
def main(i):
    data_dict_parents = ["data", "train", "PERFECT_CUT_a_z_1_9"]
    data_dict_parents = os.path.join(*data_dict_parents)
    data_dict_file = "0annotation.json"
    data_dict_path = os.path.join(data_dict_parents, data_dict_file)

    datadict = load_data(data_dict_path)

    dataset = HandwrittenData(
        datadict=datadict, data_path=data_dict_parents, transform=process_img
    )

    img_tensor, label = dataset.__getitem__(i)

    print("Tensor Img:\n", img_tensor)
    print("\nLabel: ", label)


if __name__ == "__main__":
    main()
