cmd=/home/jussi/src/dwilib/test-rls.py
dir=SI_05_03_2014_GS_highB_pmap2
scans=scans.txt
models="Mono MonoN Kurt KurtN Stretched StretchedN Biexp"

for model in ${models}; do
    echo
    echo ${model}
    ${cmd} f bin ${scans} ${dir}/*_${model}.txt
    ${cmd} t bin ${scans} ${dir}/*_${model}.txt
    ${cmd} t cancer ${scans} ${dir}/*_${model}.txt
done

echo
echo "SI"
${cmd} f bin ${scans} SI_05_03_2014_GS_highB/*
${cmd} t bin ${scans} SI_05_03_2014_GS_highB/*
${cmd} t cancer ${scans} SI_05_03_2014_GS_highB/*
