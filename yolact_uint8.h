extern "C"{
    int set_graph(int img_h, int img_w, graph_t graph);
    int inference( void* ptr, void* ptr2, int height, int width, graph_t graph, tensor_t tensor);
}