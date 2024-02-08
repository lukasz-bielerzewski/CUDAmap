#include <cuda_runtime.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/cuda/point_cloud.h>
#include <thrust/device_vector.h>
#include <pcl/point_types.h>
#include <omp.h>
#include <iostream>

#include "CUDAmap.h"


__host__ SimpleMat34 convertToSimpleMat(walkers::Mat34 &eigenMatrix)
{
    SimpleMat34 simpleMat;
    // Assuming row-major order in Eigen
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            simpleMat.data[row * 4 + col] = eigenMatrix(row, col);
        }
    }
    return simpleMat;
}

__device__ SimpleMat34 operator*(const SimpleMat34& a, const SimpleMat34& b)
{
    SimpleMat34 result;

    // Matrix multiplication logic for 3x4 matrices
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            result.data[row * 4 + col] = 0; // Initialize the result element
            for (int k = 0; k < 3; ++k) { // Iterate over the columns of 'a' and rows of 'b'
                // Adjust the calculation if your matrices are not 3x4 or if they have different rules
                result.data[row * 4 + col] += a.data[row * 4 + k] * b.data[k * 4 + col];
            }
        }
    }

    return result;
}

__global__ void transfromPointsKernel(pcl::PointXYZ *inputPointCloud, SimpleMat34 pointCL, int numPoints, SimpleMat34 camPose)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints)
    {
        // Apply the transformation
        pointCL.data[4 * 0 + 3] = inputPointCloud[idx].x;
        pointCL.data[4 * 1 + 3] = inputPointCloud[idx].y;
        pointCL.data[4 * 2 + 3] = inputPointCloud[idx].z;
        pointCL = camPose * pointCL;
        pointCL.data[4 * 1 + 3] = -pointCL.data[4 * 1 + 3];

        // Update the point
        inputPointCloud[idx].x = pointCL.data[4 * 0 + 3];
        inputPointCloud[idx].y = pointCL.data[4 * 1 + 3];
        inputPointCloud[idx].z = pointCL.data[4 * 2 + 3];
    }
}

__host__ void transformAndUploadPointCloud(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &input_cloud,
                                           std::vector<SimpleColor> &colorVec,
                                           pcl::gpu::DeviceArray<pcl::PointXYZ> &output_device_array)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);

    //std::cout << "3.1" << std::endl;

    cloud_xyz->resize(input_cloud->size());

    //std::cout << "input_cloud->size(): " << input_cloud->size() << std::endl;
    //std::cout << "cloud_xyz->size(): " << cloud_xyz->size() << std::endl;

    //std::cout << "3.2" << std::endl;

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(input_cloud->size()); ++i)
    {
        cloud_xyz->points[i].x = input_cloud->points[i].x;
        cloud_xyz->points[i].y = input_cloud->points[i].y;
        cloud_xyz->points[i].z = input_cloud->points[i].z;
        colorVec[i].r = input_cloud->points[i].r;
        colorVec[i].g = input_cloud->points[i].g;
        colorVec[i].b = input_cloud->points[i].b;
        colorVec[i].a = input_cloud->points[i].a;
    }

    //std::cout << "3.3" << std::endl;

    // Now upload the XYZ data to the GPU
    //output_device_array.create(cloud_xyz->size());

    // Allocate memory on the device
    pcl::PointXYZ* device_ptr;
    cudaMalloc(&device_ptr, cloud_xyz->size() * sizeof(pcl::PointXYZ));

    // Copy data from host to device
    cudaMemcpy(device_ptr, cloud_xyz->points.data(), cloud_xyz->size() * sizeof(pcl::PointXYZ), cudaMemcpyHostToDevice);

    // Store the device pointer in output_device_array
    output_device_array = pcl::gpu::DeviceArray<pcl::PointXYZ>(device_ptr, cloud_xyz->size());

    cudaDeviceSynchronize();

    //cudaMalloc(&(cloud_xyz->points[0]), cloud_xyz->size() * sizeof(pcl::PointXYZ));
    //cudaMemcpy(cloud_xyz->data(), &(cloud_xyz->points[0]), cloud_xyz->size() * sizeof(pcl::PointXYZ), cudaMemcpyDeviceToDevice);
    //cudaDeviceSynchronize();

    //output_device_array.upload(&(cloud_xyz->points[0]), cloud_xyz->size());

    //std::cout << "3.4" << std::endl;
}

__host__ void downloadFromDeviceArray(pcl::gpu::DeviceArray<pcl::PointXYZ>& device_array, std::vector<SimpleColor> &colorVec, std::vector<mapping::Point3D>& host_vector)
{
    //std::cout << "7.1" << std::endl;

    // Resize host vector to fit the data from the device array
    host_vector.resize(device_array.size());

    // Download data from the device to the host
    //std::vector<pcl::PointXYZ> temp_host_vector(device_array.size());
    //device_array.download(temp_host_vector);

    // Allocate host memory for temporary storage
    pcl::PointXYZ* temp_host_array = new pcl::PointXYZ[device_array.size()];

    //std::cout << "7.2" << std::endl;

    // Download data from the device to the host
    cudaMemcpy(temp_host_array, device_array.ptr(), device_array.size() * sizeof(pcl::PointXYZ), cudaMemcpyDeviceToHost);

    //std::cout << "7.3" << std::endl;

    // Convert pcl::PointXYZ to Point3D
    #pragma omp parallel for
    for (size_t i = 0; i < device_array.size(); ++i)
    {
        const auto& pcl_point = temp_host_array[i];
        const auto& color = colorVec[i];

        // Create a new Point3D instance with position and color
        host_vector[i] = mapping::Point3D(pcl_point.x, pcl_point.y, pcl_point.z, color.r, color.g, color.b, color.a);
    }

    //std::cout << "7.4" << std::endl;

    delete[] temp_host_array;

    //std::cout << "7.5" << std::endl;

    device_array.release();

    //std::cout << "7.6" << std::endl;
}

__host__ void transformPoints(grabber::PointCloud &cloud1, walkers::Mat34 pointCL, mapping::PointCloud &cloudMap1, walkers::Mat34 &camPose)
{
    // Init data structures
    pcl::gpu::DeviceArray<pcl::PointXYZ> output_device_array;

    //std::cout << "1" << std::endl;

    std::vector<SimpleColor> colorVec;
    colorVec.resize(cloud1.size());

    //std::cout << "2" << std::endl;S

    // Get shared pointer
    std::shared_ptr<pcl::PointCloud<pcl::PointXYZRGBA>> cloudPtr = std::make_shared<pcl::PointCloud<pcl::PointXYZRGBA>>(cloud1);

    //std::cout << "3" << std::endl;

    // Convert pcl::PointCloud<pcl::PointXYZRGBA> to pcl::gpu::DeviceArray<pcl::PointXYZ>
    transformAndUploadPointCloud(cloudPtr, colorVec, output_device_array);

    //std::cout << "4" << std::endl;

    // Convert matrices
    SimpleMat34 t_pointCL = convertToSimpleMat(pointCL);
    SimpleMat34 t_camPose = convertToSimpleMat(camPose);

    //std::cout << "5" << std::endl;

    // Define kernel launch parameters
    int blockSize = 256;
    int numBlocks = (output_device_array.size() + blockSize - 1) / blockSize;

    //std::cout << "6" << std::endl;

    // Launch the kernel
    transfromPointsKernel<<<numBlocks, blockSize>>>(output_device_array.ptr(), t_pointCL, output_device_array.size(), t_camPose);
    cudaDeviceSynchronize();

    //std::cout << "7" << std::endl;

    // Copy back the transformed points
    cloudMap1.clear();
    downloadFromDeviceArray(output_device_array, colorVec, cloudMap1);

    //std::cout << "8" << std::endl;

   // Release GPU memory
    output_device_array.release();
}

