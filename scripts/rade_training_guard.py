"""CPU-only guards for the generic COLMAP path in the IGS RaDe fork."""
import argparse


def validate_rade_arguments(argv):
    # Match argparse's accepted abbreviated long options as well as -l/--loader.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-l', '--loader', default='')
    parser.add_argument('--eval', action='store_true')
    options, _ = parser.parse_known_args(argv)
    if options.loader not in ('', 'colmap'):
        raise ValueError('Use --loader colmap: IGS dataset-specific loaders contain '
                         'unvalidated absolute-world-z pruning.')
    if options.eval:
        raise ValueError('This input is already train-only; --eval would silently '
                         'withhold additional training cameras.')
    return options.loader
