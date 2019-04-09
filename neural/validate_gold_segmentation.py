#!/usr/bin/python3

import sys

for line in sys.stdin:
    line = line.rstrip("\n")
    fields = line.split("\t")
    assert len(fields) == 3

    for field in fields:
        assert len(field) > 0

    assert len(fields[1]) == len(fields[2])

    assert set(fields[2]) == {"B", "C"}
