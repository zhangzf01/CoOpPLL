"""
Goal
---
1. Read test results from log.txt files
2. Compute mean and std across different folders (seeds)

Usage
---
Assume the output files are saved under output/my_experiment,
which contains results of different seeds, e.g.,

my_experiment/
    seed1/
        log.txt
    seed2/
        log.txt
    seed3/
        log.txt

Run the following command from the root directory:

$ python tools/parse_test_res.py output/my_experiment

Add --ci95 to the argument if you wanna get 95% confidence
interval instead of standard deviation:

$ python tools/parse_test_res.py output/my_experiment --ci95

If my_experiment/ has the following structure,

my_experiment/
    exp-1/
        seed1/
            log.txt
            ...
        seed2/
            log.txt
            ...
        seed3/
            log.txt
            ...
    exp-2/
        ...
    exp-3/
        ...

Run

$ python tools/parse_test_res.py output/my_experiment --multi-exp
"""
import re
import numpy as np
import os.path as osp
import argparse
from collections import OrderedDict, defaultdict

from dassl.utils import check_isfile, listdir_nohidden


def compute_ci95(res):
    return 1.96 * np.std(res) / np.sqrt(len(res))


def parse_function(*metrics, directory="", args=None, end_signal=None):
    # print(f"Parsing files in {directory}")
    subdirs = listdir_nohidden(directory, sort=True)

    outputs = []

    for subdir in subdirs:
        fpath = osp.join(directory, subdir, "log.txt")
        assert check_isfile(fpath)
        good_to_go = False
        output = OrderedDict()

        with open(fpath, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()

                if line == end_signal:
                    good_to_go = True

                for metric in metrics:
                    match = metric["regex"].search(line)
                    if match and good_to_go:
                        if "file" not in output:
                            output["file"] = fpath
                        num = float(match.group(1))
                        name = metric["name"]
                        output[name] = num

        if output:
            outputs.append(output)

    assert len(outputs) > 0, f"Nothing found in {directory}"

    metrics_results = defaultdict(list)

    for output in outputs:
        msg = ""
        for key, value in output.items():
            if isinstance(value, float):
                msg += f"{key}: {value:.2f}%. "
            else:
                msg += f"{key}: {value}. "
            if key != "file":
                metrics_results[key].append(value)
        # print(msg)

    output_results = OrderedDict()

    print("===")
    print(f"Summary of directory: {directory}")
    for key, values in metrics_results.items():
        avg = np.mean(values)
        std = compute_ci95(values) if args.ci95 else np.std(values)
        print(f"* {key}: {avg:.2f}% +- {std:.2f}%")
        output_results[key] = [avg,std]
        print(values)
    print("===")

    return output_results


def main(args, end_signal):
    metric = {
        "name": args.keyword,
        "regex": re.compile(fr"\* {args.keyword}: ([\.\deE+-]+)%"),
    }

    if args.multi_exp:
        final_results = defaultdict(list)

        for directory in listdir_nohidden(args.directory, sort=True):
            directory = osp.join(args.directory, directory)
            results = parse_function(
                metric, directory=directory, args=args, end_signal=end_signal
            )

            for key, value in results.items():
                final_results[key].append(value)

        print("Average performance")
        for key, values in final_results.items():
            avg = np.mean(values)
            print(f"* {key}: {avg:.2f}%")
        

    else:
        avg=parse_function(
            metric, directory=args.directory, args=args, end_signal=end_signal
        )
        return avg



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("pr", type=str)
    # parser.add_argument("directory", type=str, help="path to directory")
    parser.add_argument(
        "--ci95", action="store_true", help=r"compute 95\% confidence interval"
    )
    parser.add_argument("--test-log", action="store_true", help="parse test-only logs")
    parser.add_argument(
        "--multi-exp", action="store_true", help="parse multiple experiments"
    )
    parser.add_argument(
        "--keyword", default="accuracy", type=str, help="which keyword to extract"
    )
    args = parser.parse_args()
    end_signal = "Finish training"
    args.keyword='accuracy'
    if args.test_log:
        end_signal = "=> result"
#########################################
    dataset = args.dataset
    pr = args.pr



    directories=[

        '/home/zzf/doc/linearprobe_sel/LP_RC/'+dataset+'/rn50/lr0.01/partial_rate_' + pr, 
        '/home/zzf/doc/linearprobe_sel/LP_RC/'+dataset+'/rn50/lr0.025/partial_rate_' + pr, 
        '/home/zzf/doc/linearprobe_sel/LP_RC/'+dataset+'/rn50/lr0.5/partial_rate_' + pr, 
        '/home/zzf/doc/linearprobe_sel/LP_RC/'+dataset+'/rn50/lr0.1/partial_rate_' + pr, 
        '/home/zzf/doc/linearprobe_sel/LP_RC/'+dataset+'/rn50/lr0.25/partial_rate_' + pr, 
        '/home/zzf/doc/linearprobe_sel/LP_RC/'+dataset+'/rn50/lr0.5/partial_rate_' + pr, 
        '/home/zzf/doc/linearprobe_sel/LP_RC/'+dataset+'/rn50/lr1/partial_rate_' + pr, 
        '/home/zzf/doc/linearprobe_sel/LP_RC/'+dataset+'/rn50/lr2.5/partial_rate_' + pr, 
        '/home/zzf/doc/linearprobe_sel/LP_RC/'+dataset+'/rn50/lr5/partial_rate_' + pr, 
        '/home/zzf/doc/linearprobe_sel/LP_RC/'+dataset+'/rn50/lr10/partial_rate_' + pr, 
        '/home/zzf/doc/linearprobe_sel/LP_RC/'+dataset+'/rn50/lr25/partial_rate_' + pr, 
        '/home/zzf/doc/linearprobe_sel/LP_RC/'+dataset+'/rn50/lr50/partial_rate_' + pr, 

    ]


    best = [-1,-1]
    best_dir=''
    for d in directories: 
        args.directory = d
        a = main(args, end_signal)
        if a['accuracy'][0]>best[0]: 
            best=a['accuracy']
            best_dir=d
    print('BEST----------------')
    print(best_dir)
    print(best)