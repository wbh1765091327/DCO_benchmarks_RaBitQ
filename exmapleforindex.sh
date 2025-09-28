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
C_values=(158 316 632)
M_values=(8 32)
efConstruction_values=(250 500 750)

dataset="instructorxl-arxiv-768_100k"
hnsw_C=16
mkdir -p logger/${dataset}
mkdir -p result/${dataset}

for C in "${C_values[@]}"; do

    echo "Processing dataset: $dataset"
    dataset_base="./data/${dataset}/${dataset}_base.fvecs"
    dataset_query="./data/${dataset}/${dataset}_query.fvecs"
    dataset_groundtruth="./data/${dataset}/${dataset}_groundtruth.ivecs"

    ivf_centroids_path="./data/${dataset}/${dataset}_centroids_${C}.fvecs"
    ivf_cluster_id_path="./data/${dataset}/${dataset}_clusterids_${C}.ivecs"



    for bits in 7; do
        dataset_ivf_index="./data/${dataset}/${dataset}_ivf_${C}_${bits}.index"
        python3 ./python/ivf.py ${dataset_base} ${C} ${ivf_centroids_path} ${ivf_cluster_id_path} 
        ./bin/ivf_rabitq_indexing ${dataset_base} ${ivf_centroids_path} ${ivf_cluster_id_path} ${bits} ${dataset_ivf_index} 
        ./bin/ivf_rabitq_querying ${dataset_ivf_index} ${dataset_query} ${dataset_groundtruth}  >> result/${dataset}/ivf_rabitq_querying_${C}_${bits}.log 2>&1
    done
done

for ef in "${efConstruction_values[@]}"; do
    M=16
    echo "Processing dataset: $dataset"
    dataset_base="./data/${dataset}/${dataset}_base.fvecs"
    dataset_query="./data/${dataset}/${dataset}_query.fvecs"
    dataset_groundtruth="./data/${dataset}/${dataset}_groundtruth.ivecs"

    hnsw_centroids_path="./data/${dataset}/${dataset}_centroids_${hnsw_C}_ef${ef}_M${M}.fvecs"
    hnsw_cluster_id_path="./data/${dataset}/${dataset}_clusterids_${hnsw_C}_ef${ef}_M${M}.ivecs"


    for bits in 7; do
        dataset_hnsw_index="./data/${dataset}/${dataset}_hnsw_${bits}_ef${ef}_M${M}.index"
        python3 ./python/ivf.py ${dataset_base} ${M} ${hnsw_centroids_path} ${hnsw_cluster_id_path} 
        ./bin/hnsw_rabitq_indexing ${dataset_base} ${hnsw_centroids_path} ${hnsw_cluster_id_path} ${M} ${ef} ${bits} ${dataset_hnsw_index}
        ./bin/hnsw_rabitq_querying ${dataset_hnsw_index} ${dataset_query} ${dataset_groundtruth} >> result/${dataset}/hnsw_rabitq_querying_ef${ef}_M${M}_${bits}.log 2>&1
    done

done

for M in "${M_values[@]}"; do
    ef=500
    echo "Processing dataset: $dataset"
    dataset_base="./data/${dataset}/${dataset}_base.fvecs"
    dataset_query="./data/${dataset}/${dataset}_query.fvecs"
    dataset_groundtruth="./data/${dataset}/${dataset}_groundtruth.ivecs"

    hnsw_centroids_path="./data/${dataset}/${dataset}_centroids_${hnsw_C}_ef${ef}_M${M}.fvecs"
    hnsw_cluster_id_path="./data/${dataset}/${dataset}_clusterids_${hnsw_C}_ef${ef}_M${M}.ivecs"

    for bits in 7; do
        dataset_hnsw_index="./data/${dataset}/${dataset}_hnsw_${bits}_ef${ef}_M${M}.index"
        python3 ./python/ivf.py ${dataset_base} ${M} ${hnsw_centroids_path} ${hnsw_cluster_id_path} 
        ./bin/hnsw_rabitq_indexing ${dataset_base} ${hnsw_centroids_path} ${hnsw_cluster_id_path} ${M} ${ef} ${bits} ${dataset_hnsw_index} 
        ./bin/hnsw_rabitq_querying ${dataset_hnsw_index} ${dataset_query} ${dataset_groundtruth} >> result/${dataset}/hnsw_rabitq_querying_ef${ef}_M${M}_${bits}.log 2>&1
    done

done
