def format_filename(filepath, addon):

    filepath = filepath.split(".")
    filepath[-2] = filepath[-2] + addon
    filepath[-1] = "." + filepath[-1]
    filepath[0] = "." + filepath[0]
    new_filepath = "".join(filepath)

    print(new_filepath)

    return new_filepath


def get_file_extension(filepath):

    ext_arr = ["png", "jpg", "jpeg"]

    ext = filepath.split(".")

    if ext[-1] in ext_arr:
        return ext[-1]
    
    return 'not supported'