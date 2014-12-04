cmd=/home/jussi/src/dwilib/print_all.py
models="MonoN KurtN StretchedN BiexpN"

#for m in ${models}; do
#    outfile=medians_roi1.txt
#    ${cmd} -v -r 1 -s scans_az.txt -g 3+3 3+4 -m pmap/*_$m.txt >> ${outfile}
#    ${cmd} -v -r 1 -s scans_pz.txt -g 3+3 3+4 -m pmap/*_$m.txt >> ${outfile}
#    ${cmd} -v -r 1 -s scans_cz.txt -g 3+3 3+4 -m pmap/*_$m.txt >> ${outfile}
#    outfile=medians_roi2.txt
#    ${cmd} -v -r 2 -s scans_az.txt -g 3+3 3+4 -m pmap/*_$m.txt >> ${outfile}
#    ${cmd} -v -r 2 -s scans_pz.txt -g 3+3 3+4 -m pmap/*_$m.txt >> ${outfile}
#    ${cmd} -v -r 2 -s scans_cz.txt -g 3+3 3+4 -m pmap/*_$m.txt >> ${outfile}
#done

for roi in 1 2; do
    outfile=medians_roi${roi}.txt
    echo ROI${roi} > ${outfile}
    for area in az pz cz; do
        echo ${area} >> ${outfile}
        for m in ${models}; do
            echo ${m} >> ${outfile}
            ${cmd} -r ${roi} -s scans_${area}.txt -g 3+3 3+4 -m pmap/*_${m}.txt >> ${outfile}
        done
    done
done
