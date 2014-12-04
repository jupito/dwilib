cmd=/home/jussi/src/dwilib/make_ascii.py
dir=$1
models="Mono MonoN Kurt KurtN Stretched StretchedN Biexp BiexpN"

for model in ${models}; do
    echo
    echo ${model}
    ${cmd} -v -m ${dir}_pmap/*_${model}.txt
done

echo SI
echo
echo ${model}
${cmd} -v -m ${dir}/*
