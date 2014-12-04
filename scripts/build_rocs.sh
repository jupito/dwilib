cmd=/home/jussi/src/dwilib/draw_roc.py
#dir=SI_05_03_2014_GS_highB_pmap2
dir=$1
scans=scans.txt
models="Mono MonoN Kurt KurtN Stretched StretchedN Biexp BiexpN"

for model in ${models}; do
    echo
    echo ${model}
    ${cmd} roc_${model} f bin ${scans} ${dir}_pmap/*_${model}.txt
done

model=SI
echo
echo ${model}
${cmd} roc_${model} f bin ${scans} ${dir}/*
