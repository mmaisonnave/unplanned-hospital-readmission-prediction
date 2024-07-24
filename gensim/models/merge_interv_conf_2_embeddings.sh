
echo "Joining back interv_conf_2_embeddings.npy. Six parts required"
# File originally splitted in 50MB chunks using "split" comand as follows
# $ split interv_conf_2_embeddings.npy.tgz -b 50m interv_conf_2_embeddings.npy.tgz

cat interv_conf_2_embeddings.npy.tgz.part1 \
	interv_conf_2_embeddings.npy.tgz.part2 \
	interv_conf_2_embeddings.npy.tgz.part3 \
	interv_conf_2_embeddings.npy.tgz.part4 \
	interv_conf_2_embeddings.npy.tgz.part5 \
	interv_conf_2_embeddings.npy.tgz.part6 > interv_conf_2_embeddings.npy.tgz


tar zxvf interv_conf_2_embeddings.npy.tgz
rm interv_conf_2_embeddings.npy.tgz
