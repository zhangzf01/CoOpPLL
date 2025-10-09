dataset=$1
pr=$2
trainer=$3


# linear probe
python analysis/parse_test_res.py /home/zzf/doc/linearprobe_sel/${trainer}/${dataset}/rn50/lr0.01/partial_rate_${pr}
python analysis/parse_test_res.py /home/zzf/doc/linearprobe_sel/${trainer}/${dataset}/rn50/lr0.025/partial_rate_${pr}
python analysis/parse_test_res.py /home/zzf/doc/linearprobe_sel/${trainer}/${dataset}/rn50/lr0.05/partial_rate_${pr}
python analysis/parse_test_res.py /home/zzf/doc/linearprobe_sel/${trainer}/${dataset}/rn50/lr0.1/partial_rate_${pr}
python analysis/parse_test_res.py /home/zzf/doc/linearprobe_sel/${trainer}/${dataset}/rn50/lr0.25/partial_rate_${pr}
python analysis/parse_test_res.py /home/zzf/doc/linearprobe_sel/${trainer}/${dataset}/rn50/lr0.5/partial_rate_${pr}
python analysis/parse_test_res.py /home/zzf/doc/linearprobe_sel/${trainer}/${dataset}/rn50/lr1/partial_rate_${pr}
python analysis/parse_test_res.py /home/zzf/doc/linearprobe_sel/${trainer}/${dataset}/rn50/lr2.5/partial_rate_${pr}
python analysis/parse_test_res.py /home/zzf/doc/linearprobe_sel/${trainer}/${dataset}/rn50/lr5/partial_rate_${pr}
python analysis/parse_test_res.py /home/zzf/doc/linearprobe_sel/${trainer}/${dataset}/rn50/lr10/partial_rate_${pr}
python analysis/parse_test_res.py /home/zzf/doc/linearprobe_sel/${trainer}/${dataset}/rn50/lr25/partial_rate_${pr}
python analysis/parse_test_res.py /home/zzf/doc/linearprobe_sel/${trainer}/${dataset}/rn50/lr50/partial_rate_${pr}




