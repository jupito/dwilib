cmd=/home/jussi/src/dwilib/draw_boxplot.py
#dir=SI_05_03_2014_GS_highB_pmap3
dir=$1
scans=scans.txt
models="SiN Mono MonoN Kurt KurtN Stretched StretchedN Biexp BiexpN"

for model in ${models}; do
    echo ${model}
    ${cmd} boxplot_${model} ${scans} ${dir}/*_${model}.txt > boxplot_${model}.txt
done
