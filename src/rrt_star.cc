#include "RRTStar.h"

using namespace tensorflow;

REGISTER_OP("RRTStar")
    .Input("image: float")
    .Input("cost_map: float")
    .Input("label: float")
    .Input("random_n: float")
    .Output("path: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });        
        
REGISTER_KERNEL_BUILDER(Name("RRTStar").Device(DEVICE_CPU), RRTStarOp);

