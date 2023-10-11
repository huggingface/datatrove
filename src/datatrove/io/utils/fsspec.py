import fsspec


def valid_fsspec_path(path):
    try:
        fsspec.get_filesystem_class(path.split("://")[0])
    except ValueError:
        return False
    return True
