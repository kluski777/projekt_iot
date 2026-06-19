package org.tensorflow.lite.examples.objectdetection

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import java.nio.FloatBuffer

/**
 * Runs model.onnx via ONNX Runtime.
 *
 * Expected model contract:
 *   Input  "images"  : [1, 3, 640, 640]  float32, raw pixel values [0, 255], NCHW
 *   Output "output0" : best plate box [1, 4] — x1 y1 x2 y2 in 640×640 pixel space
 *   Output "output1" : plate characters [1, 8, vocab_size]
 *       Alphabet: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ " (space = pad / unknown)
 */
class ObjectDetectorHelper(
    var threshold: Float = 0.25f,
    var iouThreshold: Float = 0.45f,
    var maxResults: Int = 5,
    var numThreads: Int = 2,
    val context: Context,
    val objectDetectorListener: DetectorListener?
) {
    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null

    private val inputSize = 640
    private val plateAlphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "

    init {
        setupObjectDetector()
    }

    fun clearObjectDetector() {
        ortSession?.close()
        ortSession = null
    }

    fun setupObjectDetector() {
        try {
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(numThreads)
            }
            val bytes = context.assets.open("model.onnx").readBytes()
            ortSession = ortEnv.createSession(bytes, opts)
            Log.d(TAG, "ONNX session ready. inputs=${ortSession?.inputNames} outputs=${ortSession?.outputNames}")
        } catch (e: Exception) {
            objectDetectorListener?.onError("Failed to load model.onnx: ${e.message}")
            Log.e(TAG, "Session creation failed", e)
        }
    }

    fun detect(image: Bitmap, imageRotation: Int) {
        val session = ortSession ?: return

        val start = SystemClock.uptimeMillis()

        val scaled = Bitmap.createScaledBitmap(image, inputSize, inputSize, true)
        val inputBuf = bitmapToNchw(scaled)

        val inputTensor = OnnxTensor.createTensor(
            ortEnv, inputBuf,
            longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        )

        val outputs = try {
            session.run(mapOf(session.inputNames.first() to inputTensor))
        } catch (e: Exception) {
            objectDetectorListener?.onError("Inference error: ${e.message}")
            Log.e(TAG, "Inference failed", e)
            inputTensor.close()
            return
        } finally {
            inputTensor.close()
        }

        val inferenceTime = SystemClock.uptimeMillis() - start
        Log.d(TAG, "Inference done in ${inferenceTime}ms")
        val detections = parseOutputs(outputs, image.width, image.height)
        outputs.close()

        objectDetectorListener?.onResults(detections, inferenceTime, image.height, image.width)
    }

    private fun bitmapToNchw(bitmap: Bitmap): FloatBuffer {
        val n = inputSize * inputSize
        val pixels = IntArray(n)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        val buf = FloatBuffer.allocate(3 * n)
        for (i in 0 until n) {
            val px = pixels[i]
            buf.put(i,       ((px shr 16) and 0xFF).toFloat())
            buf.put(i + n,   ((px shr 8)  and 0xFF).toFloat())
            buf.put(i + n*2, ( px         and 0xFF).toFloat())
        }
        return buf  // position stays 0 due to absolute puts
    }

    private fun parseOutputs(outputs: OrtSession.Result, origW: Int, origH: Int): List<PlateDetection> {
        val names = outputs.map { it.key }.toList()
        if (names.isEmpty()) return emptyList()

        val boxTensor  = outputs[names[0]].get() as? OnnxTensor ?: return emptyList()
        val charTensor = if (names.size > 1) outputs[names[1]].get() as? OnnxTensor else null

        val plateText = charTensor?.let { decodePlateChars(it) } ?: ""

        // output0: [1, 4] — x1 y1 x2 y2 in 640×640 pixel space
        val data = boxTensor.floatBuffer
        Log.d(TAG, "output0 raw box: x1=${data[0]} y1=${data[1]} x2=${data[2]} y2=${data[3]}")
        Log.d(TAG, "output1 plate text: \"$plateText\"")

        val sx = origW.toFloat() / inputSize
        val sy = origH.toFloat() / inputSize

        return listOf(PlateDetection(
            boundingBox = RectF(
                (data[0] * sx).coerceIn(0f, origW.toFloat()),
                (data[1] * sy).coerceIn(0f, origH.toFloat()),
                (data[2] * sx).coerceIn(0f, origW.toFloat()),
                (data[3] * sy).coerceIn(0f, origH.toFloat()),
            ),
            confidence = 1.0f,
            classLabel = "plate",
            plateText  = plateText,
        ))
    }

    // Plate character tensor: [1, 8, vocab_size]
    private fun decodePlateChars(tensor: OnnxTensor): String {
        val shape = tensor.info.shape
        if (shape.size < 3) return ""
        val numPos  = shape[1].toInt()
        val vocabSz = shape[2].toInt()
        val data    = tensor.floatBuffer

        return buildString {
            for (pos in 0 until numPos) {
                var bestIdx = 0
                var bestVal = Float.NEGATIVE_INFINITY
                for (v in 0 until vocabSz) {
                    val score = data[pos * vocabSz + v]
                    if (score > bestVal) { bestVal = score; bestIdx = v }
                }
                append(if (bestIdx < plateAlphabet.length) plateAlphabet[bestIdx] else ' ')
            }
        }.trimEnd()
    }

    interface DetectorListener {
        fun onError(error: String)
        fun onResults(results: List<PlateDetection>, inferenceTime: Long, imageHeight: Int, imageWidth: Int)
    }

    companion object {
        private const val TAG = "ObjectDetectorHelper"
    }
}
