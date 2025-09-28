# # compiling 
# mkdir build bin 
# cd build 
# cmake ..
# make 

# # Download the dataset
# wget -P ./data/gist ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
# tar -xzvf ./data/gist/gist.tar.gz -C ./data/gist

# Create output directories
mkdir -p logger result

# Create logger directories for each dataset
for dataset in "${datasets[@]}"; do
    mkdir -p logger/${dataset}
done

datasets=(
    "glove-200-angular"
    "sift-128-euclidean"
    "msong-420"
    "contriever-768"
    "gist-960-euclidean"
    "deep-image-96-angular"
    "instructorxl-arxiv-768"
    "openai-1536-angular"
)
hnsw_C=16

for dataset in "${datasets[@]}"; do
    if [ $dataset == "contriever-768" ]; then
        ivf_C=1990
    elif [ $dataset == "glove-200-angular" ]; then
        ivf_C=2176
    elif [ $dataset == "sift-128-euclidean" ]; then
        ivf_C=2000
    elif [ $dataset == "msong-420" ]; then
        ivf_C=1984
    elif [ $dataset == "gist-960-euclidean" ]; then
        ivf_C=2000
    elif [ $dataset == "deep-image-96-angular" ]; then
        ivf_C=12643
    elif [ $dataset == "instructorxl-arxiv-768" ]; then
        ivf_C=3002
    elif [ $dataset == "openai-1536-angular" ]; then
        ivf_C=1999
    fi
    echo "Processing dataset: $dataset"
    dataset_base="./data/${dataset}/${dataset}_base.fvecs"

    ivf_centroids_path="./data/${dataset}/${dataset}_centroids_${ivf_C}.fvecs"
    ivf_cluster_id_path="./data/${dataset}/${dataset}_clusterids_${ivf_C}.ivecs"
    hnsw_centroids_path="./data/${dataset}/${dataset}_centroids_${hnsw_C}.fvecs"
    hnsw_cluster_id_path="./data/${dataset}/${dataset}_clusterids_${hnsw_C}.ivecs"

    dataset_query="./data/${dataset}/${dataset}_query.fvecs"
    dataset_groundtruth="./data/${dataset}/${dataset}_groundtruth.ivecs"

    for bits in 7; do
        dataset_ivf_index="./data/${dataset}/${dataset}_ivf_${bits}_test.index"
        python3 ./python/ivf.py ${dataset_base} ${ivf_C} ${ivf_centroids_path} ${ivf_cluster_id_path} 
        ./bin/ivf_rabitq_indexing ${dataset_base} ${ivf_centroids_path} ${ivf_cluster_id_path} ${bits} ${dataset_ivf_index}
        # res3="./result/${dataset}/${dataset}_ad_ivf_perf_result.txt"
        # /usr/bin/time -v \
        # perf stat \
        #     -e instructions,cycles,branches,branch-misses \
        #     -e cache-misses,cache-references \
        #     -e L1-dcache-load-misses,L1-dcache-loads \
        #     -e L1-dcache-store-misses,L1-dcache-stores \
        #     -e LLC-load-misses,LLC-loads \
        #     -e LLC-store-misses,LLC-stores \
        #     -e dTLB-load-misses,dTLB-loads \
        #     -e dTLB-store-misses,dTLB-stores \
        #     -e page-faults \
        #     -o ${res3} \
        ./bin/ivf_rabitq_querying ${dataset_ivf_index} ${dataset_query} ${dataset_groundtruth}  >> result/${dataset}/hnsw_rabitq_querying_${bits}_pruning.log 2>&1
    done

    indexing and querying for RabitQ+ with hnsw, do clustering first
    python3 ./python/ivf.py ${dataset_base} 16 ${hnsw_centroids_path} ${hnsw_cluster_id_path} >> logger/${dataset}/hnsw_clustering.log 2>&1
    for bits in 7; do
        dataset_hnsw_index="./data/${dataset}/${dataset}_hnsw_${bits}.index"
        python3 ./python/ivf.py ${dataset_base} 16 ${hnsw_centroids_path} ${hnsw_cluster_id_path} >> logger/${dataset}/hnsw_clustering.log 2>&1
        ./bin/hnsw_rabitq_indexing ${dataset_base} ${hnsw_centroids_path} ${hnsw_cluster_id_path} 16 500 ${bits} ${dataset_hnsw_index} >> logger/${dataset}/hnsw_rabitq_indexing_${bits}.log 2>&1
        # res3="./result/${dataset}/${dataset}_ad_hnsw_perf_result.txt"
        # /usr/bin/time -v \
        # perf stat \
        #     -e instructions,cycles,branches,branch-misses \
        #     -e cache-misses,cache-references \
        #     -e L1-dcache-load-misses,L1-dcache-loads \
        #     -e L1-dcache-store-misses,L1-dcache-stores \
        #     -e LLC-load-misses,LLC-loads \
        #     -e LLC-store-misses,LLC-stores \
        #     -e dTLB-load-misses,dTLB-loads \
        #     -e dTLB-store-misses,dTLB-stores \
        #     -e page-faults \
        #     -o ${res3} \
        ./bin/hnsw_rabitq_querying ${dataset_hnsw_index} ${dataset_query} ${dataset_groundtruth} >> result/${dataset}/hnsw_rabitq_querying_${bits}_pruning.log 2>&1
    done

done
