from .compare_visualizer import NPZComparer
import argparse
def argparser():
    parser = argparse.ArgumentParser(description='Compare two npz files')
    parser.add_argument('target', type=str, help='Path to the target npz file')
    parser.add_argument('reference', type=str, help='Path to the reference npz file')
    parser.add_argument('-t', '--tolerance', type=str, default="0.99,0.90", help='Tolerance for the comparison, cosine and euclid. Default is 0.99,0.90')
    parser.add_argument('-q', '--quiet', action='store_true', help='If set, only return the result code')
    parser.add_argument('-v', '--verbose', action='store_true', help='If set, dump the top 10 differences if failure')
    parser.add_argument('-s', '--summary', action='store_true', help='If set, only print the summary of the comparison')
    return parser.parse_args()

if __name__ == "__main__":
    args = argparser()
    tolerance = list(map(float, args.tolerance.split(',')))
    comparer = NPZComparer(args.target, args.reference)
    if args.quiet and args.verbose:
        print("Cannot set both --quiet and --verbose")
        exit(1)
    verbose = 0 if args.quiet else (2 if args.verbose else 1)
    ret = comparer.compare(tolerance=tolerance, verbose=verbose, summary=args.summary)
    exit(0 if ret else 1)