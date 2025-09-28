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


datasets=(
    # "glove-25-angular_100k"
    # "glove-100-angular_100k"
    # "glove-200-angular_100k"
    # "glove-50-angular_100k"
    # "glove-200-angular_1k"
    # "glove-200-angular_10k"
    # "glove-200-angular"
    # "sift-128-euclidean"
    # "msong-420"
    # "contriever-768"
    # "gist-960-euclidean"
    # "deep-image-96-angular"
    # "instructorxl-arxiv-768"
    # "openai-1536-angular"
    "instructorxl-arxiv-768_1k"
    "instructorxl-arxiv-768_10k"
    "instructorxl-arxiv-768_1000k"
)
hnsw_C=16
for dataset in "${datasets[@]}"; do
    mkdir -p logger/${dataset}
    mkdir -p result/${dataset}
done
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
    elif [ $dataset == "glove-25-angular_100k" ]; then
        ivf_C=316
    elif [ $dataset == "glove-100-angular_100k" ]; then
        ivf_C=316
    elif [ $dataset == "glove-200-angular_100k" ]; then
        ivf_C=316
    elif [ $dataset == "glove-50-angular_100k" ]; then
        ivf_C=316
    elif [ $dataset == "glove-200-angular_1k" ]; then
        ivf_C=31
    elif [ $dataset == "glove-200-angular_10k" ]; then
        ivf_C=100
    elif [ $dataset == "instructorxl-arxiv-768_1k" ]; then      
        ivf_C=31
    elif [ $dataset == "instructorxl-arxiv-768_10k" ]; then
        ivf_C=100
    elif [ $dataset == "instructorxl-arxiv-768_1000k" ]; then
        ivf_C=1000
    fi

    echo "Processing dataset: $dataset"
    dataset_base="./data/${dataset}/${dataset}_base.fvecs"
    dataset_query="./data/${dataset}/${dataset}_query.fvecs"
    dataset_groundtruth="./data/${dataset}/${dataset}_groundtruth.ivecs"

    ivf_centroids_path="./data/${dataset}/${dataset}_centroids_${ivf_C}.fvecs"
    ivf_cluster_id_path="./data/${dataset}/${dataset}_clusterids_${ivf_C}.ivecs"
    hnsw_centroids_path="./data/${dataset}/${dataset}_centroids_${hnsw_C}.fvecs"
    hnsw_cluster_id_path="./data/${dataset}/${dataset}_clusterids_${hnsw_C}.ivecs"



    for bits in 7; do
        dataset_ivf_index="./data/${dataset}/${dataset}_ivf_${bits}.index"
        python3 ./python/ivf.py ${dataset_base} ${ivf_C} ${ivf_centroids_path} ${ivf_cluster_id_path} >> logger/${dataset}/ivf_clustering.log 2>&1
        ./bin/ivf_rabitq_indexing ${dataset_base} ${ivf_centroids_path} ${ivf_cluster_id_path} ${bits} ${dataset_ivf_index} >> logger/${dataset}/ivf_rabitq_indexing_${bits}.log 2>&1
        ./bin/ivf_rabitq_querying ${dataset_ivf_index} ${dataset_query} ${dataset_groundtruth}  >> result/${dataset}/ivf_rabitq_querying_${bits}.log 2>&1
    done

    for bits in 7; do
        dataset_hnsw_index="./data/${dataset}/${dataset}_hnsw_${bits}.index"
        python3 ./python/ivf.py ${dataset_base} 16 ${hnsw_centroids_path} ${hnsw_cluster_id_path} >> logger/${dataset}/hnsw_clustering.log 2>&1
        ./bin/hnsw_rabitq_indexing ${dataset_base} ${hnsw_centroids_path} ${hnsw_cluster_id_path} 16 500 ${bits} ${dataset_hnsw_index} >> logger/${dataset}/hnsw_rabitq_indexing_${bits}.log 2>&1
        ./bin/hnsw_rabitq_querying ${dataset_hnsw_index} ${dataset_query} ${dataset_groundtruth} >> result/${dataset}/hnsw_rabitq_querying_${bits}.log 2>&1
    done

done
