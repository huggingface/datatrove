import xxhash


def xxhash32(data: str):
    return xxhash.xxh32_intdigest(data)


def xxhash64(data: str):
    return xxhash.xxh64_intdigest(data)
