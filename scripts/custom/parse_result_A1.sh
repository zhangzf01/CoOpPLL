dataset=$1
trainer1=$2
trainer2=$3
task=$4




python analysis/parse_test_res.py /home/zzf/doc/${task}/${trainer1}/${dataset}/rn50_16shots_nctx16/partial_rate_0.1
python analysis/parse_test_res.py /home/zzf/doc/${task}/${trainer1}/${dataset}/rn50_16shots_nctx16/partial_rate_0.3
python analysis/parse_test_res.py /home/zzf/doc/${task}/${trainer1}/${dataset}/rn50_16shots_nctx16/partial_rate_0.5
python analysis/parse_test_res.py /home/zzf/doc/${task}/${trainer1}/${dataset}/rn50_16shots_nctx16/partial_rate_0.7

python analysis/parse_test_res.py /home/zzf/doc/${task}/${trainer2}/${dataset}/rn50_16shots_nctx16/partial_rate_0.1
python analysis/parse_test_res.py /home/zzf/doc/${task}/${trainer2}/${dataset}/rn50_16shots_nctx16/partial_rate_0.3
python analysis/parse_test_res.py /home/zzf/doc/${task}/${trainer2}/${dataset}/rn50_16shots_nctx16/partial_rate_0.5
python analysis/parse_test_res.py /home/zzf/doc/${task}/${trainer2}/${dataset}/rn50_16shots_nctx16/partial_rate_0.7

