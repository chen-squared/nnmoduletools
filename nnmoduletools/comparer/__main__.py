from .compare_visualizer import NPZComparer
import argparse

def argparser():
    parser = argparse.ArgumentParser(description='Compare two npz files and generate a report')
    parser.add_argument('target', type=str, help='Path to the target npz file')
    parser.add_argument('reference', type=str, help='Path to the reference npz file')
    parser.add_argument('tensor', type=str, nargs="*", help='The tensor(s) to compare. Leave blank to compare all. If multiple, use space to split: tensor1 tensor2')
    parser.add_argument('-t', '--tolerance', type=str, default="0.99,0.90", help='Tolerance for the comparison, cosine and euclid. Default is 0.99,0.90')
    parser.add_argument('-a', "--abs_tol", type=float, default=1e-8, help="The absolute tolerance for plot. Default is 1e-8")
    parser.add_argument('-r', "--rel_tol", type=float, default=1e-3, help="The relative tolerance for plot. Default is 1e-3")
    parser.add_argument('-v', '--verbose', type=int, default=3, help="Set the verbose level. 0: quiet, 1: normal, 2: dump failed, 3: dump and plot failed, 4: dump and plot all.  Default is 3.")
    parser.add_argument('-s', '--summary', action='store_true', help='If set, only print the summary of the comparison')
    parser.add_argument('-o', '--output_dir', type=str, default="compare_report", help='The output dir to save the comparison result. Default is compare_report')
    parser.add_argument('-f', '--output_fn', type=str, default="compare_report.md", help='The output filename of markdown report. Default is compare_report.md')
    parser.add_argument('-i', '--info', action='store_true', help='Print the shape info of the npz file or given tensors instead of generating a report')
    parser.add_argument('-d', '--dump', nargs='?', const=-1, type=int, help='Dump the tensor data instead of generating a report. If a number is given, only dump the top-N errors. Will not affect dump in comparing and plotting. Default is dump all.')
    return parser.parse_args()

def main():
    args = argparser()
    comparer = NPZComparer(args.target, args.reference)
    tensors = args.tensor if args.tensor else None
    if args.info:
        comparer.info(tensor=tensors, show_isolated=True)
        return True
    tolerance = list(map(float, args.tolerance.split(',')))
    if args.dump is not None:
        if args.dump == -1: args.dump = None
        comparer.dump_vs(tensor=tensors, abs_tol=args.abs_tol, rel_tol=args.rel_tol, verbose=True, top_k=args.dump)
        return True
    return comparer.report(tensor=tensors, tolerance=tolerance,
                           abs_tol=args.abs_tol, rel_tol=args.rel_tol,
                           verbose=args.verbose, summary=args.summary, output_dir=args.output_dir,
                           title=f"{args.target} vs {args.reference}", output_fn=args.output_fn)

if __name__ == "__main__":
    ret = main()
    exit(0 if ret else 1)