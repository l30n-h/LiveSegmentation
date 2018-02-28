#include <LiveSegmentation/BA/Classifier.hpp>
#include <LiveSegmentation/BA/ClassifierQueue.hpp>
#include <LiveSegmentation/BA/Probabilities.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/unordered_map.hpp>

#include <iostream>
#include <chrono>

#include <cstdlib>

// template <typename K, typename V> using HashMap = boost::unordered_map<K, V>;
template <typename K, typename V> using HashMap = std::unordered_map<K, V>;

int main(int argc, char** argv) {
    int framesCount = 10;
    int labelsCount = 20;
    if(argc>1) framesCount = std::atoi(argv[1]);
    if(argc>2) labelsCount = std::atoi(argv[2]);
    int width = 640;
    int height = 480;
    HashMap<int, ls::Probabilities> voxelMap;
    std::vector<cv::Mat> predictions;
    for(int i=0;i<labelsCount;i++){
        predictions.push_back(cv::Mat::zeros(width, height, CV_8UC1));
    }
    // voxelMap.reserve(width*height*framesCount);
    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<framesCount; ++i){
        for(int w=0;w<width;++w){
            for(int h=0;h<height;++h){
                int key = (int)((float)std::rand()/RAND_MAX*(640*480+i*640*(480/3)))+i*640*(480/3);//std::rand();//(int)((float)std::rand()/RAND_MAX*10000);
                // auto it_voxel = voxelMap.find(key);
                // if(it_voxel == voxelMap.end()){
                //     it_voxel = voxelMap.insert(it_voxel, std::pair<int, ls::Probabilities>(key, ls::Probabilities()));   
                //     // it_voxel = voxelMap.emplace_hint(it_voxel, key, ls::Probabilities());
                // }
                // auto probs = it_voxel->second;

                auto probs = voxelMap[key];
                
                probs.update(predictions, w, h);
                ls::LPPair lppair = probs.getMax();
            } 
        }
       
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    std::cout << voxelMap.size() << "\n";
    std::cout << duration << "\n";
    std::cout << duration/framesCount << "\n";
}