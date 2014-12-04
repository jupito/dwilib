cmd=/home/jussi/src/dwilib/correlation.py
scans=$1 # all, cz, pz
d=pmap
#models="Mono MonoN Kurt KurtN Stretched StretchedN Biexp BiexpN SiN"
models="MonoN KurtN StretchedN BiexpN"

outfile=out_correlation_${scans}_each.txt
rm -f ${outfile}
for model in ${models}; do
    ${cmd} -s scans_${scans}.txt -l score -m ${d}/*_${model}.txt >> ${outfile}
done

outfile=out_correlation_${scans}_3+3,3+4,others.txt
rm -f ${outfile}
for model in ${models}; do
    ${cmd} -s scans_${scans}.txt -l score -g 3+3 3+4 -m ${d}/*_${model}.txt >> ${outfile}
done
