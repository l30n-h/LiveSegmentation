unsigned int BUFFER_SIZE = 128*1024*1024;

void *buffer;
cudaMallocManaged(&buffer, BUFFER_SIZE);

cudaFree(buffer);



template<typename tT>
__host__
void execBuffered(uint fromSize, const std::function<void(const thurst::device_vector<tT>&, uint start, uint n)> &task){
    thrust::device_vector<tT> toList(buffer, buffer+BUFFER_SIZE/sizeof(tT));
    uint start=0;
    while(true){
        uint n = min(fromSize-start, toList.size());
        if(n<=0) break;
        task(toList, start, end);
        start+=n;
    }
}

struct Key {
    Vector3int16_t block_coordinates;
    uint voxel_index;
    __host__ __device__
    Key(Vector3int16_t block_coordinates, 
        uint voxel_index):
        block_coordinates(block_coordinates), 
        voxel_index(voxel_index){}
};

struct IndexData {
    Vector3int16_t block_coordinate;
    uint16_t voxel_index;
    Voxel* voxel;
    Vector2uint32_t pixel;
    float weight;
};
thrust::device_vector<thrust::pair<Vector3int16_t, Voxel*>> voxelBlockList(voxelList.device_begin(), voxelList.device_begin()+voxelList.size());
auto getIndexDataBuffered = [&voxelBlockList](const thurst::device_vector<IndexData> out&, uint start, uint n){
    struct getIndexData {
        __device__
        IndexData operator()(uint i) {
            uint blockIndex = i/voxels_per_block;
            uint16_t voxelIndex = (uint16_t)(i%voxels_per_block);
            thrust::pair<Vector3int16_t, Voxel*> voxelBlock = voxelBlockList[blockIndex];
            auto block_coordinate = voxelBlock.first;
            auto voxel_coordinates = VoxelHelper::index2voxel(voxelIndex, size_voxel);
            Vector3float voxel_local = pose.global2localPoint(VoxelHelper::voxel2pointcenter(block_coordinate, voxel_coordinates, size_voxel));
            if(intrinsics.isPointInsideScreen(voxel_local)){
                return IndexData(block_coordinate, voxelIndex, voxelBlock.second, static_cast<Vector2uint32_t>(intrinsics.projectPointExact(voxel_local)), voxelBlock.second[voxelIndex].weight)
            }
            return IndexData(block_coordinate, voxelIndex, voxelBlock.second, Vector2uint32_t(), -1)
        }
    };
    thrust::transform(thrust::counting_iterator<uint> counter_start(start),
                      thrust::counting_iterator<uint> counter_start(start+n),
                      out.start(),
                      getIndexData);
    thurst::host_vector<IndexData> indexDataList(out);
    //TODO do something
});
execBuffered(list.size()*voxels_per_block, getIndexDataBuffered);

struct MaxData {
    uint16_t voxel_index;
    Voxel* voxel;
    thrust::pair<uint, float> max;
};
thurst::host_vector<IndexData> indexDataList(out);
struct getMaxAfterUpdatingProbabilities {
    __host__
    MaxData operator()(const IndexData &d) {
        //TODO voxelMap synchronized or only get and later map update
        if(weight==0){
            voxelMap.erase(Key(d.block_coordinate, d.voxel_index));
        }
        if(d.weight<=0) return MaxData(d.voxel_index, d.voxel, thrust::pair(0, 0));
        auto probabilities = voxelMap[Key(d.block_coordinate, d.voxel_index)];
        probabilities.update(predictions, d.pixel.x, d.pixel.y);
        return MaxData(d.voxel_index, d.voxel, probabilities.getMax());
    }
};
thurst::host_vector<MaxData> maxList(out);
thrust::transform(indexDataList.begin(),
                  indexDataList.end(),
                  maxList.begin(),
                  getMaxAfterUpdatingProbabilities);



auto updateMaxBuffered = [&maxList](const thurst::device_vector<MaxData> out&, uint start, uint n){
    thurst::copy(maxList.begin()+start, maxList.begin()+start+n, out.start());
    struct updateMax {
        __device__
        void operator()(const MaxData &d) {
            d.voxel[d.voxel_index].classification = d.max;
        }
    };
    thrust::for_each(out.begin(), out.begin()+n, updateMax());
});
execBuffered(maxList.size(), updateMaxBuffered);