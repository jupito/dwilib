cmd=/home/jussi/src/dwilib/compare_auc.py
scans=$1
d=pmap
nmap=4
#mfiles="$d/*_Mono.txt $d/*_MonoN.txt $d/*_Kurt.txt $d/*_KurtN.txt $d/*_Stretched.txt $d/*_StretchedN.txt $d/*_Biexp.txt $d/*_BiexpN.txt $d/*_SiN.txt"
mfiles="$d/*_MonoN.txt $d/*_KurtN.txt $d/*_StretchedN.txt $d/*_BiexpN.txt"
nboot=100000

outfile=out_compare_auc_${scans}_cancer.txt
c1="${cmd} -v -s scans_${scans}.txt -l cancer -b ${nboot} -n ${nmap} -m $mfiles > $outfile"

outfile=out_compare_auc_${scans}_3+3_vs_others.txt
g="3+3"
c2="${cmd} -v -s scans_${scans}.txt -l score -g ${g} -b ${nboot} -n ${nmap} -m $mfiles > $outfile"

outfile=out_compare_auc_${scans}_3+3,3+4_vs_others.txt
g="3+3 3+4"
c3="${cmd} -v -s scans_${scans}.txt -l score -g ${g} -b ${nboot} -n ${nmap} -m $mfiles > $outfile"

parallel -- "$c1" "$c2" "$c3"
