import xxhash


def xxhash32(data: str | bytes):
    return xxhash.xxh32_intdigest(data)


def xxhash64(data: str | bytes):
    return xxhash.xxh64_intdigest(data)
