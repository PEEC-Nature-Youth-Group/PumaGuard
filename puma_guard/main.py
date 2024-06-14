import argparse


def parse_commandline():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def main():
    options = parse_commandline()
