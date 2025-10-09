name=$1
g1=$2
g2=$3
g3=$4
g4=$5
g5=$6
g6=$7
g7=$8




nohup bash scripts/coopv2_single.sh ${name} rn50 end 16 16 False 0.0 ${g1} &
nohup bash scripts/coopv2_single.sh ${name} rn50 end 16 16 False 0.1 ${g1} &
nohup bash scripts/coopv2_single.sh ${name} rn50 end 16 16 False 0.2 ${g2} &
nohup bash scripts/coopv2_single.sh ${name} rn50 end 16 16 False 0.3 ${g3} &
nohup bash scripts/coopv2_single.sh ${name} rn50 end 16 16 False 0.4 ${g4} &
nohup bash scripts/coopv2_single.sh ${name} rn50 end 16 16 False 0.5 ${g5} &


