from PIL import Image


def _resize_data(img: Image.Image):
    factor = 32 / img.height
    new_width = int(img.width * factor)

    safe_width = min(128, new_width)

    return img.resize((safe_width, 32))


def _add_padding(img: Image.Image):
    new_width = 128
    new_height = 32

    temp_img = Image.new(img.mode, (new_width, new_height), 255)

    temp_img.paste(img, (0, 0))

    return temp_img


def process_img(file: str):
    img = Image.open(file).convert("L")

    temp_out = _resize_data(img)

    if temp_out.height == 32 and temp_out.width == 128:
        return temp_out
    else:
        return _add_padding(temp_out)


if __name__ == "__main__":
    img = process_img(
        "/home/quichoarch/Desktop/HandwrittenOCR/data/datos_entrenamiento/PERFECT_CUT_a_z_1_9/pre_conzuelo_A.0.jpg"
    )
    img.show()
