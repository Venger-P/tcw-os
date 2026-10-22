/* TCW-Runtime v0.4: TFLite Micro wrapper.
 * Exposes C-API for tcw_ai_init() and tcw_ai_infer_raw().
 * Calibration (alpha, beta) applied by caller in main.c. */

#include "tcw_ai.h"

#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <math.h>

#include "model.hpp"

namespace {
    const tflite::Model *model = nullptr;
    tflite::MicroInterpreter *interpreter = nullptr;
    TfLiteTensor *input  = nullptr;
    TfLiteTensor *output = nullptr;

    constexpr int kTensorArenaSize = 2000;
    uint8_t tensor_arena[kTensorArenaSize];
}

extern "C" void tcw_ai_init(void)
{
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        return;
    }

    static tflite::MicroMutableOpResolver<1> resolver;
    resolver.AddFullyConnected();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        interpreter = nullptr;
        return;
    }

    input  = interpreter->input(0);
    output = interpreter->output(0);
}

extern "C" float tcw_ai_infer_raw(float x)
{
    if (!interpreter) return 0.0f;

    int8_t x_q = (int8_t)roundf(x / input->params.scale)
                 + input->params.zero_point;
    input->data.int8[0] = x_q;

    if (interpreter->Invoke() != kTfLiteOk) return 0.0f;

    int8_t y_q = output->data.int8[0];
    return (y_q - output->params.zero_point) * output->params.scale;
}
