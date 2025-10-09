# search from seed 1




# #  COOP
# nohup bash scripts/custom/single.sh eurosat rn50 16 0.1 C1_naive CoOp_NAIVE 7 &
# nohup bash scripts/custom/single.sh eurosat rn50 16 0.2 C1_naive CoOp_NAIVE 6 &
# nohup bash scripts/custom/single.sh eurosat rn50 16 0.3 C1_naive CoOp_NAIVE 5 &
# nohup bash scripts/custom/single.sh eurosat rn50 16 0.4 C1_naive CoOp_NAIVE 4 &
# nohup bash scripts/custom/single.sh eurosat rn50 16 0.5 C1_naive CoOp_NAIVE 3 &
# nohup bash scripts/custom/single.sh eurosat rn50 16 0.1 C1_RC_coop CoOp_RC 0 &
# nohup bash scripts/custom/single.sh eurosat rn50 16 0.2 C1_RC_coop CoOp_RC 1 &
# nohup bash scripts/custom/single.sh eurosat rn50 16 0.3 C1_RC_coop CoOp_RC 2 &
# nohup bash scripts/custom/single.sh eurosat rn50 16 0.4 C1_RC_coop CoOp_RC 3 &
# nohup bash scripts/custom/single.sh eurosat rn50 16 0.5 C1_RC_coop CoOp_RC 4 &






# Linear Probe + RC

## first we'll  test linear probe's performance under clean labels for eurosat. 


# nohup bash scripts/custom/lp_sel.sh caltech101 rn50 0.1 LP_PiCO 0&
# nohup bash scripts/custom/lp_sel.sh caltech101 rn50 0.2 LP_PiCO 0&
# nohup bash scripts/custom/lp_sel.sh caltech101 rn50 0.3 LP_PiCO 1&
# nohup bash scripts/custom/lp_sel.sh caltech101 rn50 0.4 LP_PiCO 1&
# nohup bash scripts/custom/lp_sel.sh caltech101 rn50 0.5 LP_PiCO 2&
# nohup bash scripts/custom/lp_sel2.sh caltech101 rn50 0.1 LP_PiCO 1&
# nohup bash scripts/custom/lp_sel2.sh caltech101 rn50 0.2 LP_PiCO 1&
# nohup bash scripts/custom/lp_sel2.sh caltech101 rn50 0.3 LP_PiCO 1&
# nohup bash scripts/custom/lp_sel2.sh caltech101 rn50 0.4 LP_PiCO 3&
# nohup bash scripts/custom/lp_sel2.sh caltech101 rn50 0.5 LP_PiCO 4&
# nohup bash scripts/custom/lp_sel3.sh caltech101 rn50 0.1 LP_PiCO 0&
# nohup bash scripts/custom/lp_sel3.sh caltech101 rn50 0.2 LP_PiCO 0&
# nohup bash scripts/custom/lp_sel3.sh caltech101 rn50 0.3 LP_PiCO 1&
# nohup bash scripts/custom/lp_sel3.sh caltech101 rn50 0.4 LP_PiCO 1&
# nohup bash scripts/custom/lp_sel3.sh caltech101 rn50 0.5 LP_PiCO 3&
# nohup bash scripts/custom/lp_sel4.sh caltech101 rn50 0.1 LP_PiCO 3&
# nohup bash scripts/custom/lp_sel4.sh caltech101 rn50 0.2 LP_PiCO 4&
# nohup bash scripts/custom/lp_sel4.sh caltech101 rn50 0.3 LP_PiCO 6&
# nohup bash scripts/custom/lp_sel4.sh caltech101 rn50 0.4 LP_PiCO 7&
# nohup bash scripts/custom/lp_sel4.sh caltech101 rn50 0.5 LP_PiCO 7&

# nohup bash scripts/custom/lp_sel.sh ucf101 rn50 0.1 LP_PiCO 6&
# nohup bash scripts/custom/lp_sel.sh ucf101 rn50 0.2 LP_PiCO 6&
# nohup bash scripts/custom/lp_sel.sh ucf101 rn50 0.3 LP_PiCO 6&
# nohup bash scripts/custom/lp_sel.sh ucf101 rn50 0.4 LP_PiCO 7&
# nohup bash scripts/custom/lp_sel.sh ucf101 rn50 0.5 LP_PiCO 7&
# nohup bash scripts/custom/lp_sel2.sh ucf101 rn50 0.1 LP_PiCO 7&
# nohup bash scripts/custom/lp_sel2.sh ucf101 rn50 0.2 LP_PiCO 0&
# nohup bash scripts/custom/lp_sel2.sh ucf101 rn50 0.3 LP_PiCO 1&
# nohup bash scripts/custom/lp_sel2.sh ucf101 rn50 0.4 LP_PiCO 2&
# nohup bash scripts/custom/lp_sel2.sh ucf101 rn50 0.5 LP_PiCO 4&
# nohup bash scripts/custom/lp_sel3.sh ucf101 rn50 0.1 LP_PiCO 5&
# nohup bash scripts/custom/lp_sel3.sh ucf101 rn50 0.2 LP_PiCO 5&
# nohup bash scripts/custom/lp_sel3.sh ucf101 rn50 0.3 LP_PiCO 0&
# nohup bash scripts/custom/lp_sel3.sh ucf101 rn50 0.4 LP_PiCO 6&
# nohup bash scripts/custom/lp_sel3.sh ucf101 rn50 0.5 LP_PiCO 6&
# nohup bash scripts/custom/lp_sel4.sh ucf101 rn50 0.1 LP_PiCO 5&
# nohup bash scripts/custom/lp_sel4.sh ucf101 rn50 0.2 LP_PiCO 5&
# nohup bash scripts/custom/lp_sel4.sh ucf101 rn50 0.3 LP_PiCO 6&
# nohup bash scripts/custom/lp_sel4.sh ucf101 rn50 0.4 LP_PiCO 7&
# nohup bash scripts/custom/lp_sel4.sh ucf101 rn50 0.5 LP_PiCO 7&

# nohup bash scripts/custom/lp_sel.sh caltech101 rn50 0.0 LP_RC 2&
# nohup bash scripts/custom/lp_sel2.sh caltech101 rn50 0.0 LP_RC 2&
# nohup bash scripts/custom/lp_sel3.sh caltech101 rn50 0.0 LP_RC 2&
# nohup bash scripts/custom/lp_sel4.sh caltech101 rn50 0.0 LP_RC 2&
# nohup bash scripts/custom/lp_sel.sh ucf101 rn50 0.0 LP_RC 2&
# nohup bash scripts/custom/lp_sel2.sh ucf101 rn50 0.0 LP_RC 2&
# nohup bash scripts/custom/lp_sel3.sh ucf101 rn50 0.0 LP_RC 2&
# nohup bash scripts/custom/lp_sel4.sh ucf101 rn50 0.0 LP_RC 2&






# nohup bash scripts/custom/single.sh ucf101 rn50 16 0.1 C1_naive CoOp_NAIVE 2 &
# nohup bash scripts/custom/single.sh ucf101 rn50 16 0.2 C1_naive CoOp_NAIVE 2 &
# nohup bash scripts/custom/single.sh ucf101 rn50 16 0.3 C1_naive CoOp_NAIVE 7 &
# nohup bash scripts/custom/single.sh ucf101 rn50 16 0.4 C1_naive CoOp_NAIVE 2 &
# nohup bash scripts/custom/single.sh ucf101 rn50 16 0.5 C1_naive CoOp_NAIVE 2 &

# nohup bash scripts/custom/single.sh caltech101 rn50 16 0.1 C1_naive CoOp_NAIVE 2 &
# nohup bash scripts/custom/single.sh caltech101 rn50 16 0.2 C1_naive CoOp_NAIVE 2 &
# nohup bash scripts/custom/single.sh caltech101 rn50 16 0.3 C1_naive CoOp_NAIVE 7 &
# nohup bash scripts/custom/single.sh caltech101 rn50 16 0.4 C1_naive CoOp_NAIVE 2 &
# nohup bash scripts/custom/single.sh caltech101 rn50 16 0.5 C1_naive CoOp_NAIVE 2 &




# nohup bash scripts/custom/single.sh ucf101 rn50 16 0.0 C1_RC_coop CoOp_RC 4 &
# nohup bash scripts/custom/single.sh ucf101 rn50 16 0.1 C1_RC_coop CoOp_RC 4 &
# nohup bash scripts/custom/single.sh ucf101 rn50 16 0.2 C1_RC_coop CoOp_RC 6 &
# nohup bash scripts/custom/single.sh ucf101 rn50 16 0.3 C1_RC_coop CoOp_RC 6 &
# nohup bash scripts/custom/single.sh ucf101 rn50 16 0.4 C1_RC_coop CoOp_RC 2 &
# nohup bash scripts/custom/single.sh ucf101 rn50 16 0.5 C1_RC_coop CoOp_RC 6 &

# nohup bash scripts/custom/single.sh caltech101 rn50 16 0.0 C1_RC_coop CoOp_RC 0 &
# nohup bash scripts/custom/single.sh caltech101 rn50 16 0.1 C1_RC_coop CoOp_RC 3 &
# nohup bash scripts/custom/single.sh caltech101 rn50 16 0.2 C1_RC_coop CoOp_RC 3 &
# nohup bash scripts/custom/single.sh caltech101 rn50 16 0.3 C1_RC_coop CoOp_RC 5 &
# nohup bash scripts/custom/single.sh caltech101 rn50 16 0.4 C1_RC_coop CoOp_RC 6 &
# nohup bash scripts/custom/single.sh caltech101 rn50 16 0.5 C1_RC_coop CoOp_RC 1 &


# python analysis/parse_test_res.py /home/zzf/doc/C1_naive/CoOp_NAIVE/caltech101/rn50_16shots_nctx16/partial_rate_0.1
# python analysis/parse_test_res.py /home/zzf/doc/C1_naive/CoOp_NAIVE/caltech101/rn50_16shots_nctx16/partial_rate_0.2
# python analysis/parse_test_res.py /home/zzf/doc/C1_naive/CoOp_NAIVE/caltech101/rn50_16shots_nctx16/partial_rate_0.3
# python analysis/parse_test_res.py /home/zzf/doc/C1_naive/CoOp_NAIVE/caltech101/rn50_16shots_nctx16/partial_rate_0.4
# python analysis/parse_test_res.py /home/zzf/doc/C1_naive/CoOp_NAIVE/caltech101/rn50_16shots_nctx16/partial_rate_0.5
# python analysis/parse_test_res.py /home/zzf/doc/C1_RC_coop/CoOp_RC/caltech101/rn50_16shots_nctx16/partial_rate_0.1
# python analysis/parse_test_res.py /home/zzf/doc/C1_RC_coop/CoOp_RC/caltech101/rn50_16shots_nctx16/partial_rate_0.2
# python analysis/parse_test_res.py /home/zzf/doc/C1_RC_coop/CoOp_RC/caltech101/rn50_16shots_nctx16/partial_rate_0.3
# python analysis/parse_test_res.py /home/zzf/doc/C1_RC_coop/CoOp_RC/caltech101/rn50_16shots_nctx16/partial_rate_0.4
# python analysis/parse_test_res.py /home/zzf/doc/C1_RC_coop/CoOp_RC/caltech101/rn50_16shots_nctx16/partial_rate_0.5
# python analysis/parse_test_res.py /home/zzf/doc/C1_RC_coop/CoOp_RC/caltech101/rn50_16shots_nctx16/partial_rate_0.0
