#ifndef TCW_AI_H
#define TCW_AI_H

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize TFLite Micro interpreter (call once at startup) */
void  tcw_ai_init(void);

/* Run model on x, return raw y_model (before calibration).
 * y_final = alpha * y_model + beta is applied by caller. */
float tcw_ai_infer_raw(float x);

#ifdef __cplusplus
}
#endif

#endif /* TCW_AI_H */
