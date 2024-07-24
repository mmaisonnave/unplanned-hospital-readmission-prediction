echo "Joining back interv_conf_2.dv.vectors.npy. Four parts required"
# File originally splitted in 50MB chunks using "split" comand as follows
# $ split interv_conf_2.dv.vectors.npy.tgz -b 50m interv_conf_2.dv.vectors.npy.tgz
interv_conf_2.dv.vectors.npy

cat interv_conf_2.dv.vectors.npy.tgz.part1 \
	interv_conf_2.dv.vectors.npy.tgz.part2 \
	interv_conf_2.dv.vectors.npy.tgz.part3 \
	interv_conf_2.dv.vectors.npy.tgz.part4 > interv_conf_2.dv.vectors.npy.tgz


tar zxvf interv_conf_2.dv.vectors.npy.tgz 
rm interv_conf_2.dv.vectors.npy.tgz
