#!/usr/bin/env python3
import os
import argparse

def tree_summary(root, max_depth, threshold, sample_count):
    root = os.path.abspath(root)
    for current_root, dirs, files in os.walk(root):
        depth = current_root[len(root):].count(os.sep)
        if depth > max_depth:
            dirs[:] = []  # don’t recurse deeper
            continue

        indent = '    ' * depth
        print(f"{indent}{os.path.basename(current_root) or current_root}")

        # print subdirectories
        for d in dirs:
            print(f"{indent}    {d}/")

        # group files by extension
        ext_groups = {}
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            ext_groups.setdefault(ext, []).append(f)

        for ext, flist in sorted(ext_groups.items()):
            count = len(flist)
            if ext in ['.mp4', '.ip'] and count > threshold:
                print(f"{indent}    [many {ext} files: {count} files]")
                for sample in flist[:sample_count]:
                    print(f"{indent}        - {sample}")
            else:
                for f in flist:
                    print(f"{indent}    {f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize directory tree up to a given depth."
    )
    parser.add_argument("directory", help="Root directory for the tree")
    parser.add_argument("-d", "--depth", type=int, default=4,
                        help="Maximum tree depth (default: 4)")
    parser.add_argument("-t", "--threshold", type=int, default=50,
                        help="Threshold for 'many' files per extension (default: 50)")
    parser.add_argument("-s", "--sample", type=int, default=5,
                        help="Number of sample file names to show when threshold exceeded (default: 5)")
    args = parser.parse_args()

    tree_summary(args.directory, args.depth, args.threshold, args.sample)

