cmd=/home/jussi/src/dwilib/reproducibility.py
scans=$1
d=pmap
nboot=100000
#nboot=10
#models="Mono MonoN Kurt KurtN Stretched StretchedN Biexp BiexpN SiN"
models="MonoN KurtN StretchedN BiexpN"

outfile=out_reproducibility_${scans}.txt
echo ${scans} > ${outfile}
for model in ${models}; do
    ${cmd} -s scans_${scans}.txt -b ${nboot} -m ${d}/*_${model}.txt >> ${outfile}
done
