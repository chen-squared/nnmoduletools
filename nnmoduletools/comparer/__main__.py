from .compare_visualizer import NPZComparer
import argparse
def argparser():
    parser = argparse.ArgumentParser(description='Compare two npz files')
    parser.add_argument('target', type=str, help='Path to the target npz file')
    parser.add_argument('reference', type=str, help='Path to the reference npz file')
    parser.add_argument('-t', '--tolerance', type=str, default="0.99,0.90", help='Tolerance for the comparison, cosine and euclid. Default is 0.99,0.90')
    parser.add_argument('-q', '--quiet', action='store_true', help='If set, only return the result code')
    parser.add_argument('-s', '--summary', action='store_true', help='If set, only print the summary of the comparison')
    return parser.parse_args()

if __name__ == "__main__":
    args = argparser()
    tolerance = list(map(float, args.tolerance.split(',')))
    comparer = NPZComparer(args.target, args.reference)
    ret = comparer.compare(tolerance=tolerance, verbose=not args.quiet, summary=args.summary)
    exit(ret)