/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2020, OPEN AI LAB
 * Author: qtang@openailab.com
 */

/*
 * Parts of the following code in this file refs to
 * https://github.com/Tencent/ncnn/blob/master/examples/yolact.cpp
 * Tencent is pleased to support the open source community by making ncnn
 * available.
 *
 * Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this
 * file except in compliance with the License. You may obtain a copy of the
 * License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"
#define TARGET_SIZE 544
#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1
#define VXDEVICE  "TIMVX"

const char* class_names[] = {"background",
                            "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                            "train", "truck", "boat", "traffic light", "fire hydrant",
                            "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                            "skis", "snowboard", "sports ball", "kite", "baseball bat",
                            "baseball glove", "skateboard", "surfboard", "tennis racket",
                            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                            "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                            "hot dog", "pizza", "donut", "cake", "chair", "couch",
                            "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                            "toaster", "sink", "refrigerator", "book", "clock", "vase",
                            "scissors", "teddy bear", "hair drier", "toothbrush"};
const float mean_vals[3] = {123.68f, 116.78f, 103.94f};
const float norm_vals[3] = {1.0 / 58.40f, 1.0 / 57.12f, 1.0 / 57.38f};
//const float mean_vals[3] = {2.1179,2.0357,1.8114};
// const float mean_vals[3] = {123.68f, 116.78f, 103.94f};
// const float norm_vals[3] = {0.01712,0.017507,0.017428};
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<float> maskdata;
    cv::Mat mask;
};

void get_input_data_cv_uint8(const cv::Mat& sample, uint8_t* input_data, int img_h, int img_w, const float* mean, const float* scale, 
                       float input_scale, int zero_point)
{
    cv::Mat img;
    // img = sample;
    // std::cout << " Leaving bgr image" << std::endl;
    if (sample.channels() == 4)
    {
        cv::cvtColor(sample, img, cv::COLOR_BGRA2RGB);
    }
    else if (sample.channels() == 1)
    {
        cv::cvtColor(sample, img, cv::COLOR_GRAY2RGB);
    }
    else if (sample.channels() == 3)
    {
        cv::cvtColor(sample, img, cv::COLOR_BGR2RGB);
    }
    else
    {
        img = sample;
    }

    cv::resize(img, img, cv::Size(img_h, img_w));
    img.convertTo(img, CV_32FC3);
    float* img_data = (float* )img.data;

    /* nhwc to nchw */
    for (int h = 0; h < img_h; h++)
    {   for (int w = 0; w < img_w; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                int in_index  = h * img_w * 3 + w * 3 + c;
                int out_index = c * img_h * img_w + h * img_w + w;
                float input_fp32 = (img_data[in_index] - mean[c]) * scale[c];
                // std::cout << input_fp32 <<":";
                /* quant to uint8 */
                int udata = (round)(input_fp32 / input_scale + ( float )zero_point);
                // std::cout << udata << " ";
                if (udata > 255)
                    udata = 255;
                else if (udata < 0)
                    udata = 0;

                input_data[out_index] = udata;
            }
        }
    }
    // for (int h = 0; h < img_h; h++)
	// {
	// 	for (int w = 0; w < img_w; w++)
	// 	{
	// 		for (int c = 0; c < 3; c++)
	// 		{
	// 			int udata = *img_data;
	// 			if (udata > 255)
	// 				udata = 255;
	// 			else if (udata < 0)
	// 				udata = 0;
	// 			input_data[c * img_h * img_w + h * img_w + w] = udata; //- mean[c]) ;
    //             //std::cout << input_data[c * img_h * img_w + h * img_w + w];
	// 			img_data++;
	// 		}
	// 	}
	// }
}

struct Box2f
{
    float cx;
    float cy;
    float w;
    float h;
};

static std::vector<Box2f> generate_priorbox(int num_priores)
{
    std::vector<Box2f> priorboxs(num_priores);

    const int conv_ws[5] = {68, 34, 17, 9, 5};
    const int conv_hs[5] = {68, 34, 17, 9, 5};

    const float aspect_ratios[3] = {1.f, 0.5f, 2.f};
    const float scales[5] = {24.f, 48.f, 96.f, 192.f, 384.f};
    //const float scales[5] = {32.f, 64.f, 128.f, 256.f, 512.f};
    int index = 0;

    for (int i = 0; i < 5; i++)
    {
        int conv_w = conv_ws[i];
        int conv_h = conv_hs[i];
        int scale = scales[i];
        for (int ii = 0; ii < conv_h; ii++)
        {
            for (int j = 0; j < conv_w; j++)
            {
                float cx = (j + 0.5f) / conv_w;
                float cy = (ii + 0.5f) / conv_h;

                for (int k = 0; k < 3; k++)
                {
                    float ar = aspect_ratios[k];

                    ar = sqrt(ar);

                    float w = scale * ar / TARGET_SIZE;
                    float h = scale / ar / TARGET_SIZE;

                    h = w;

                    Box2f& priorbox = priorboxs[index];

                    priorbox.cx = cx;
                    priorbox.cy = cy;
                    priorbox.w = w;
                    priorbox.h = h;

                    index += 1;
                }
            }
        }
    }

    return priorboxs;
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void fast_nms(std::vector<std::vector<Object>>& class_candidates, std::vector<Object>& objects,
                     const float iou_thresh, const int nms_top_k, const int keep_top_k)
{
    for (int i = 0; i < ( int )class_candidates.size(); i++)
    {
        std::vector<Object>& candidate = class_candidates[i];
        std::sort(candidate.begin(), candidate.end(), [](const Object& a, const Object& b) { return a.prob > b.prob; });
        if (candidate.size() == 0)
            continue;

        if (nms_top_k != 0 && nms_top_k > candidate.size())
        {
            candidate.erase(candidate.begin() + nms_top_k, candidate.end());
        }

        objects.push_back(candidate[0]);
        const int n = candidate.size();
        std::vector<float> areas(n);
        std::vector<int> keep(n);
        for (int j = 0; j < n; j++)
        {
            areas[j] = candidate[j].rect.area();
        }
        std::vector<std::vector<float>> iou_matrix;
        for (int j = 0; j < n; j++)
        {
            std::vector<float> iou_row(n);
            for (int k = 0; k < n; k++)
            {
                float inter_area = intersection_area(candidate[j], candidate[k]);
                float union_area = areas[j] + areas[k] - inter_area;
                iou_row[k] = inter_area / union_area;
            }
            iou_matrix.push_back(iou_row);
        }
        for (int j = 1; j < n; j++)
        {
            std::vector<float>::iterator max_value;
            max_value = std::max_element(iou_matrix[j].begin(), iou_matrix[j].begin() + j - 1);
            if (*max_value <= iou_thresh)
            {
                objects.push_back(candidate[j]);
            }
        }
    }
    std::sort(objects.begin(), objects.end(), [](const Object& a, const Object& b) { return a.prob > b.prob; });
    if (objects.size() > keep_top_k)
        objects.resize(keep_top_k);
}

static void concatenate(std::vector<float> &vector1, std::vector<float> &vector2){
    vector1.insert(
        vector1.end(),
        std::make_move_iterator(vector2.begin()),
        std::make_move_iterator(vector2.end()));
}
static void softmax(std::vector<float> &tensor){
    double sum = 0;
    double maxcls = 0.1;
    int maxind = 0;
    for (int i = 0; i < tensor.size()/81; i++){
        for (int j = 0; j < 81; j++){
            sum+=exp(tensor[i*81 + j]);
        }
        for (int j = 0; j < 81; j++){
            tensor[i*81 + j] = exp(tensor[i*81 + j]) / sum;
            if (tensor[i*81 + j] > maxcls){
                maxind = j;
                maxcls = tensor[i*81 + j];
            }
        }
        sum = 0;
        if (maxind != 0){
            std::cout << "maxcls: " << maxcls << ", detected: " << class_names[maxind] << ", i: " << i << std::endl;
        }
        maxcls = 0.1;
        maxind = 0;
    }
}

static std::vector<float> dequant_and_transform(tensor_t tensor){
    int out_dim[4];
    float output_scale = 0.f;
    int output_zero_point = 0;
    get_tensor_quant_param(tensor, &output_scale, &output_zero_point, 1);
    int counter = get_tensor_buffer_size(tensor) / sizeof(uint8_t);
    uint8_t* data_uint8 = ( uint8_t* )get_tensor_buffer(tensor);
    std::vector<float> data_fp32(counter);
    get_tensor_shape( tensor, out_dim, 4);
    for (int c = 0; c < counter; c++)
    {
        data_fp32[c] = (( float )data_uint8[c] - ( float )output_zero_point) * output_scale; 
    }  

    // {
    //     data_fp32[c] = (( float )data_uint8[c % out_dim[1] * out_dim[2] * out_dim[3] + (c / out_dim[1])  ] - ( float )output_zero_point) * output_scale; // 243x68x68 -> 13872x81
    // }                                     //c %  243        *  68         *  68      +     c /  243    

    return data_fp32; 
}

// static std::vector<float> summ_tensors(tensor_t tensor0, tensor_t tensor1, tensor_t tensor2, tensor_t tensor3, tensor_t tensor4){
//     int out_dim[5][4];
//     float output_scale[5] = { 0.f, 0.f, 0.f, 0.f, 0.f};
//     int output_zero_point[5] = { 0, 0, 0, 0, 0};
//     int counter[5];
//     uint8_t* data_uint8[5];
//     int size = 0;
//     std::vector<tensor_t> tensors ={ tensor0, tensor1, tensor2, tensor3, tensor4};
//     for (int i = 0; i < 5; i++){
//         get_tensor_quant_param(tensors[i], &output_scale[i], &output_zero_point[i], 1);
//         counter[i] = get_tensor_buffer_size(tensors[i]) / sizeof(uint8_t);
//         data_uint8[i] = ( uint8_t* )get_tensor_buffer(tensors[i]);
//         size += counter[i];
//         get_tensor_shape( tensors[i], &out_dim[i][0], 4);
//     }

//     std::vector<float> data_fp32(size);
    // for (int c = 0; c < size; c++){
    //     for (int i= 0; i < 5; i++)
    //     {
    //         data_fp32[c + i * 243] = (( float )data_uint8[i][c % out_dim[i][1] * out_dim[i][2] * out_dim[i][3] + (c / out_dim[i][1])  ] - ( float )output_zero_point[i]) * output_scale[i]; // 243x68x68 -> 13872x81
    //     }                                     //c %  243        *  68         *  68      +     c /  243    
//     }
//     return data_fp32; 
// }

static int detect_yolact(const cv::Mat& bgr, std::vector<Object>& objects, const char* model_file, int repeat_count,
                         int num_thread)
{
    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_UINT8;
    opt.affinity = 0;

    /* inital tengine */
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    const int target_size = TARGET_SIZE;

    int img_w = bgr.cols;
    int img_h = bgr.rows;


    context_t timvx_context = create_context("timvx", 1);

	int rtt = set_context_device(timvx_context, VXDEVICE, nullptr, 0);
	if (0 > rtt)
	{
		fprintf(stderr, " add_context_device VSI DEVICE failed.\n");
		exit(-1);
	}
    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(timvx_context, "tengine", model_file);
    if (NULL == graph)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }

    /* set the input shape to initial the graph, and prerun graph to infer shape */
    int img_size = target_size * target_size * 3;
    int dims[] = {1, 3, target_size, target_size};    // nchw
    std::vector<uint8_t> input_data(img_size);

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == NULL)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor, input_data.data(), img_size) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }    

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    /* prepare process input data, set the data mem to input tensor */
    float input_scale = 0.f;
    int input_zero_point = 0;
    get_tensor_quant_param(input_tensor, &input_scale, &input_zero_point, 1);    
    get_input_data_cv_uint8(bgr, input_data.data(), target_size, target_size, mean_vals, norm_vals, input_scale, input_zero_point);

    /* run graph */
    double min_time = DBL_MAX;
    double max_time = DBL_MIN;
    double total_time = 0.;
    for (int i = 0; i < repeat_count; i++)
    {
        double start = get_current_time();
        if (run_graph(graph, 1) < 0)
        {
            fprintf(stderr, "Run graph failed\n");
            return -1;
        }
        double end = get_current_time();
        double cur = end - start;
        total_time += cur;
        min_time = std::min(min_time, cur);
        max_time = std::max(max_time, cur);
    }
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", repeat_count,
            num_thread, total_time / repeat_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");
    tensor_t proto   = get_graph_output_tensor(graph, 0, 0); // 1x136x136x32
    tensor_t class_0 = get_graph_output_tensor(graph, 1, 0); // 1x243x68x68
    tensor_t class_1 = get_graph_output_tensor(graph, 4, 0); // 1x243x34x34
    tensor_t class_2 = get_graph_output_tensor(graph, 7, 0);  // 1x243x17x17
    tensor_t class_3 = get_graph_output_tensor(graph, 10, 0); // 1x243x9x9
    tensor_t class_4 = get_graph_output_tensor(graph, 13, 0); // 1x243x5x5
    tensor_t box_0   = get_graph_output_tensor(graph, 2, 0);  // 1x12x68x68
    tensor_t box_1   = get_graph_output_tensor(graph, 5, 0);  // 1x12x34x34
    tensor_t box_2   = get_graph_output_tensor(graph, 8, 0);  // 1x12x17x17
    tensor_t box_3   = get_graph_output_tensor(graph, 11, 0); // 1x12x9x9
    tensor_t box_4   = get_graph_output_tensor(graph, 14, 0); // 1x12x5x5
    tensor_t coef_0  = get_graph_output_tensor(graph, 3, 0);  // 1x96x68x68
    tensor_t coef_1  = get_graph_output_tensor(graph, 6, 0);  // 1x96x34x34
    tensor_t coef_2  = get_graph_output_tensor(graph, 9, 0);  // 1x96x17x17
    tensor_t coef_3  = get_graph_output_tensor(graph, 12, 0); // 1x96x9x9
    tensor_t coef_4  = get_graph_output_tensor(graph, 15, 0); // 1x96x5x5
    
    int out_dim[4];
    for (int i = 0; i < 16; i++){
        get_tensor_shape( get_graph_output_tensor(graph, i, 0), out_dim, 4);
        std::cout << "Shape of " << i << "'th tensor: " << out_dim[0] << " "<< out_dim[1] << " " << out_dim[2] << " "<< out_dim[3] << std::endl;
    }
    // std::vector<float> class_0_fp32 = summ_tensors(class_0, class_1, class_2, class_3, class_4);
    // std::vector<float> box_0_fp32 = summ_tensors(box_0, box_1, box_2, box_3, box_4);
    // std::vector<float> coef_0_fp32 = summ_tensors(coef_0, coef_1, coef_2, coef_3, coef_4);
    std::vector<float> class_0_fp32 = dequant_and_transform(class_0);
    std::vector<float> class_1_fp32 = dequant_and_transform(class_1);
    std::vector<float> class_2_fp32 = dequant_and_transform(class_2);
    std::vector<float> class_3_fp32 = dequant_and_transform(class_3);
    std::vector<float> class_4_fp32 = dequant_and_transform(class_4);
    std::vector<float> box_0_fp32   = dequant_and_transform(box_0);
    std::vector<float> box_1_fp32   = dequant_and_transform(box_1);
    std::vector<float> box_2_fp32   = dequant_and_transform(box_2);
    std::vector<float> box_3_fp32   = dequant_and_transform(box_3);
    std::vector<float> box_4_fp32   = dequant_and_transform(box_4);
    std::vector<float> coef_0_fp32  = dequant_and_transform(coef_0);
    std::vector<float> coef_1_fp32  = dequant_and_transform(coef_1);
    std::vector<float> coef_2_fp32  = dequant_and_transform(coef_2);
    std::vector<float> coef_3_fp32  = dequant_and_transform(coef_3);
    std::vector<float> coef_4_fp32  = dequant_and_transform(coef_4);
    std::vector<float> proto_fp32   = dequant_and_transform(proto);
    // float output_scale = 0.f;
    // int output_zero_point = 0;
    // get_tensor_quant_param(proto, &output_scale, &output_zero_point, 1);
    // int counter = get_tensor_buffer_size(proto) / sizeof(uint8_t);
    // uint8_t* data_uint8 = ( uint8_t* )get_tensor_buffer(proto);
    // std::vector<float> proto_fp32(counter);
    // get_tensor_shape( proto, out_dim, 4);
    // for (int c = 0; c < counter; c++)
    // {
    //     proto_fp32[c] = (( float )data_uint8[c] - ( float )output_zero_point) * output_scale; 
    // }                                     

    concatenate(class_0_fp32, class_1_fp32);
    concatenate(class_0_fp32, class_2_fp32);
    concatenate(class_0_fp32, class_3_fp32);
    concatenate(class_0_fp32, class_4_fp32);
    concatenate(box_0_fp32, box_1_fp32);
    concatenate(box_0_fp32, box_2_fp32);
    concatenate(box_0_fp32, box_3_fp32);
    concatenate(box_0_fp32, box_4_fp32);
    concatenate(coef_0_fp32, coef_1_fp32);
    concatenate(coef_0_fp32, coef_2_fp32);
    concatenate(coef_0_fp32, coef_3_fp32);
    concatenate(coef_0_fp32, coef_4_fp32);
    softmax(class_0_fp32);
    std::vector<float> maskmaps   = proto_fp32;
    std::vector<float> location   = box_0_fp32;
    std::vector<float> mask       = coef_0_fp32;
    std::vector<float> confidence = class_0_fp32;
    /* dequant output data */
    // tensor_t maskmaps_tensor   = get_graph_output_tensor(graph, 1, 0);
    // tensor_t location_tensor   = get_graph_output_tensor(graph, 2, 0);
    // tensor_t mask_tensor       = get_graph_output_tensor(graph, 3, 0);
    // tensor_t confidence_tensor = get_graph_output_tensor(graph, 4, 0);

    // float maskmaps_scale = 0.f;
    // float location_scale = 0.f;
    // float mask_scale     = 0.f;
    // float confidence_scale = 0.f;

    // int maskmaps_zero_point = 0;
    // int location_zero_point = 0;
    // int mask_zero_point     = 0;
    // int confidence_zero_point = 0;
    // // Поменять по аналогу с V5
    // get_tensor_quant_param(maskmaps_tensor, &maskmaps_scale, &maskmaps_zero_point, 1);
    // get_tensor_quant_param(location_tensor, &location_scale, &location_zero_point, 1);
    // get_tensor_quant_param(mask_tensor, &mask_scale, &mask_zero_point, 1);
    // get_tensor_quant_param(confidence_tensor, &confidence_scale, &confidence_zero_point, 1);

    // int maskmaps_count   = get_tensor_buffer_size(maskmaps_tensor) / sizeof(uint8_t);
    // int location_count   = get_tensor_buffer_size(location_tensor) / sizeof(uint8_t);
    // int mask_count       = get_tensor_buffer_size(mask_tensor) / sizeof(uint8_t);
    // int confidence_count = get_tensor_buffer_size(confidence_tensor) / sizeof(uint8_t);

    // uint8_t* maskmaps_u8   = ( uint8_t* )get_tensor_buffer(maskmaps_tensor);
    // uint8_t* location_u8   = ( uint8_t* )get_tensor_buffer(location_tensor);
    // uint8_t* mask_u8       = ( uint8_t* )get_tensor_buffer(mask_tensor);
    // uint8_t* confidence_u8 = ( uint8_t* )get_tensor_buffer(confidence_tensor);

    // std::vector<float> maskmaps(maskmaps_count);
    // std::vector<float> location(location_count);
    // std::vector<float> mask(mask_count);
    // std::vector<float> confidence(confidence_count);

    // for (int c = 0; c < maskmaps_count; c++)
    // {
    //     maskmaps[c] = (( float )maskmaps_u8[c] - ( float )maskmaps_zero_point) * maskmaps_scale;
    // }

    // for (int c = 0; c < location_count; c++)
    // {
    //     location[c] = (( float )location_u8[c] - ( float )location_zero_point) * location_scale;
    // }

    // for (int c = 0; c < mask_count; c++)
    // {
    //     mask[c] = (( float )mask_u8[c] - ( float )mask_zero_point) * mask_scale;
    // }

    // for (int c = 0; c < confidence_count; c++)
    // {
    //     confidence[c] = (( float )confidence_u8[c] - ( float )confidence_zero_point) * confidence_scale;
    // }    

    /* postprocess */
    int num_class = 81;
    int num_priors = 18525;
    std::vector<Box2f> priorboxes = generate_priorbox(num_priors);
    const float confidence_thresh = 0.5f;
    const float nms_thresh = 0.3f;
    const int keep_top_k = 200;

    std::vector<std::vector<Object>> class_candidates;
    class_candidates.resize(num_class);

    for (int i = 0; i < num_priors; i++)
    {
        const float* conf = confidence.data() + i * 81;
        const float* loc = location.data() + i * 4;
        const float* maskdata = mask.data() + i * 32;
        Box2f& priorbox = priorboxes[i];

        int label = 0;
        float score = 0.f;
        for (int j = 1; j < num_class; j++)
        {
            float class_score = conf[j];
            if (class_score > score)
            {
                label = j;
                score = class_score;
            }
        }

        if (label == 0 || score <= confidence_thresh)
            continue;

        float var[4] = {0.1f, 0.1f, 0.2f, 0.2f};

        float bbox_cx = var[0] * loc[0] * priorbox.w + priorbox.cx;
        float bbox_cy = var[1] * loc[1] * priorbox.h + priorbox.cy;
        float bbox_w = ( float )(exp(var[2] * loc[2]) * priorbox.w);
        float bbox_h = ( float )(exp(var[3] * loc[3]) * priorbox.h);

        float obj_x1 = bbox_cx - bbox_w * 0.5f;
        float obj_y1 = bbox_cy - bbox_h * 0.5f;
        float obj_x2 = bbox_cx + bbox_w * 0.5f;
        float obj_y2 = bbox_cy + bbox_h * 0.5f;

        obj_x1 = std::max(std::min(obj_x1 * bgr.cols, ( float )(bgr.cols - 1)), 0.f);
        obj_y1 = std::max(std::min(obj_y1 * bgr.rows, ( float )(bgr.rows - 1)), 0.f);
        obj_x2 = std::max(std::min(obj_x2 * bgr.cols, ( float )(bgr.cols - 1)), 0.f);
        obj_y2 = std::max(std::min(obj_y2 * bgr.rows, ( float )(bgr.rows - 1)), 0.f);

        Object obj;
        obj.rect = cv::Rect_<float>(obj_x1, obj_y1, obj_x2 - obj_x1 + 1, obj_y2 - obj_y1 + 1);
        obj.label = label;
        obj.prob = score;

        obj.maskdata = std::vector<float>(maskdata, maskdata + 32);

        class_candidates[label].push_back(obj);
    }

    objects.clear();
    fast_nms(class_candidates, objects, nms_thresh, 0, keep_top_k);

    for (int i = 0; i < objects.size(); i++)
    {
        Object& obj = objects[i];

        cv::Mat mask1(136, 136, CV_32FC1);
        {
            mask1 = cv::Scalar(0.f);

            for (int p = 0; p < 32; p++)
            {
                const float* maskmap = maskmaps.data() + p;
                float coeff = obj.maskdata[p];
                float* mp = ( float* )mask1.data;

                // mask += m * coeff
                for (int j = 0; j < 136 * 136; j++)
                {
                    mp[j] += maskmap[j * 32] * coeff;
                }
            }
        }

        cv::Mat mask2;
        cv::resize(mask1, mask2, cv::Size(img_w, img_h));

        // crop obj box and binarize
        obj.mask = cv::Mat(img_h, img_w, CV_8UC1);
        {
            obj.mask = cv::Scalar(0);

            for (int y = 0; y < img_h; y++)
            {
                if (y < obj.rect.y || y > obj.rect.y + obj.rect.height)
                    continue;

                const float* mp2 = mask2.ptr<const float>(y);
                uchar* bmp = obj.mask.ptr<uchar>(y);

                for (int x = 0; x < img_w; x++)
                {
                    if (x < obj.rect.x || x > obj.rect.x + obj.rect.width)
                        continue;

                    bmp[x] = mp2[x] > 0.5f ? 255 : 0;
                }
            }
        }
    }

    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const unsigned char colors[81][3] = {
        {56, 0, 255},
        {226, 255, 0},
        {0, 94, 255},
        {0, 37, 255},
        {0, 255, 94},
        {255, 226, 0},
        {0, 18, 255},
        {255, 151, 0},
        {170, 0, 255},
        {0, 255, 56},
        {255, 0, 75},
        {0, 75, 255},
        {0, 255, 169},
        {255, 0, 207},
        {75, 255, 0},
        {207, 0, 255},
        {37, 0, 255},
        {0, 207, 255},
        {94, 0, 255},
        {0, 255, 113},
        {255, 18, 0},
        {255, 0, 56},
        {18, 0, 255},
        {0, 255, 226},
        {170, 255, 0},
        {255, 0, 245},
        {151, 255, 0},
        {132, 255, 0},
        {75, 0, 255},
        {151, 0, 255},
        {0, 151, 255},
        {132, 0, 255},
        {0, 255, 245},
        {255, 132, 0},
        {226, 0, 255},
        {255, 37, 0},
        {207, 255, 0},
        {0, 255, 207},
        {94, 255, 0},
        {0, 226, 255},
        {56, 255, 0},
        {255, 94, 0},
        {255, 113, 0},
        {0, 132, 255},
        {255, 0, 132},
        {255, 170, 0},
        {255, 0, 188},
        {113, 255, 0},
        {245, 0, 255},
        {113, 0, 255},
        {255, 188, 0},
        {0, 113, 255},
        {255, 0, 0},
        {0, 56, 255},
        {255, 0, 113},
        {0, 255, 188},
        {255, 0, 94},
        {255, 0, 18},
        {18, 255, 0},
        {0, 255, 132},
        {0, 188, 255},
        {0, 245, 255},
        {0, 169, 255},
        {37, 255, 0},
        {255, 0, 151},
        {188, 0, 255},
        {0, 255, 37},
        {0, 255, 0},
        {255, 0, 170},
        {255, 0, 37},
        {255, 75, 0},
        {0, 0, 255},
        {255, 207, 0},
        {255, 0, 226},
        {255, 245, 0},
        {188, 255, 0},
        {0, 255, 18},
        {0, 255, 75},
        {0, 255, 151},
        {255, 56, 0},
        {245, 255, 0}
    };

    cv::Mat image = bgr.clone();

    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        if (obj.prob < 0.15)
            continue;

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob, obj.rect.x, obj.rect.y,
                obj.rect.width, obj.rect.height);

        const unsigned char* color = colors[color_index % 81];
        color_index++;

        cv::rectangle(image, obj.rect, cv::Scalar(color[0], color[1], color[2]));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0));

        // draw mask
        for (int y = 0; y < image.rows; y++)
        {
            const uchar* mp = obj.mask.ptr(y);
            uchar* p = image.ptr(y);
            for (int x = 0; x < image.cols; x++)
            {
                if (mp[x] == 255)
                {
                    p[0] = cv::saturate_cast<uchar>(p[0] * 0.5 + color[0] * 0.5);
                    p[1] = cv::saturate_cast<uchar>(p[1] * 0.5 + color[1] * 0.5);
                    p[2] = cv::saturate_cast<uchar>(p[2] * 0.5 + color[2] * 0.5);
                }
                p += 3;
            }
        }
    }

    cv::imwrite("yolact_uint8_out.jpg", image);
}

void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count]\n");
}

int main(int argc, char** argv)
{
    int repeat_count = DEFAULT_REPEAT_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    char* model_file = nullptr;
    char* image_file = nullptr;

    int res;
    while ((res = getopt(argc, argv, "m:i:r:t:h:")) != -1)
    {
        switch (res)
        {
            case 'm':
                model_file = optarg;
                break;
            case 'i':
                image_file = optarg;
                break;
            case 'r':
                repeat_count = std::strtoul(optarg, nullptr, 10);
                break;
            case 't':
                num_thread = std::strtoul(optarg, nullptr, 10);
                break;
            case 'h':
                show_usage();
                return 0;
            default:
                break;
        }
    }

    /* check files */
    if (nullptr == model_file)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (nullptr == image_file)
    {
        fprintf(stderr, "Error: Image file not specified!\n");
        show_usage();
        return -1;
    }

    // if (!check_file_exist(model_file) || !check_file_exist(image_file))
    //     return -1;

    cv::Mat m = cv::imread(image_file, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", image_file);
        return -1;
    }

    std::vector<Object> objects;
    detect_yolact(m, objects, model_file, repeat_count, num_thread);
    draw_objects(m, objects);

    return 0;
}
