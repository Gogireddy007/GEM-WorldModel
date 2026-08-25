#!/usr/bin/env python
"""Pull gRodon/Madin growth-rate data + GTDB tree/taxonomy, cross-reference
them, and print the validation report (species count, growth-rate coverage,
doubling-time distribution above/below the 5h split).
"""

from gem_worldmodel.data import consolidate, validate
from gem_worldmodel.utils.config import load_config


def main():
    cfg = load_config("data")
    result = consolidate.run(cfg)
    validate.validate_labeled_corpus(result["labeled"], cfg)


if __name__ == "__main__":
    main()
