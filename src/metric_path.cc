#include "MetricPath.h"

using namespace tensorflow;

REGISTER_OP("MetricPath")
    .Input("image: float")
    .Input("label: float")
    .Input("map: float")    
    .Input("v_const: float")         
    .Output("accuracy: float")  
    .Output("dissimilarity: float")    
    ;

REGISTER_KERNEL_BUILDER(Name("MetricPath").Device(DEVICE_CPU), MetricPathOp);
