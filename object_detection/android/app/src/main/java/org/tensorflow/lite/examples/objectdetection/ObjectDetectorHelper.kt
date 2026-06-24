package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.os.SystemClock
import android.util.Log
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Runs plate_recognizer.pte (YOLO + OCR Combined) via ExecuTorch 1.3.1.
 *
 * Input  "images"  : [1, 3, 640, 640]  float32, [0-255] raw pixels, NCHW
 * Output 0 (tuple) : best plate box [1, 4] — x1 y1 x2 y2 in 640×640 pixel space
 * Output 1 (tuple) : plate characters [1, 8, 37]
 * Alphabet: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ " (space = pad / unknown)
 */
class ObjectDetectorHelper(
    var threshold: Float = 0.25f,
    var iouThreshold: Float = 0.45f,
    var maxResults: Int = 5,
    var numThreads: Int = 2,
    val context: Context,
    val objectDetectorListener: DetectorListener?
) {
    private var executorchModule: Module? = null

    // Rozmiar wejściowy dopasowany pod natywne 640x640 modelu YOLO v26n
    private val inputSize = 960
    private val plateAlphabet = "ABCDEFGHIJKLMNOPRSTUVWXYZ0123456789"

    init {
        setupObjectDetector()
    }

    fun clearObjectDetector() {
        executorchModule = null
    }

    fun setupObjectDetector() {
        try {
            val modelPath = getAssetFilePath(context, "model.pte")
            executorchModule = Module.load(modelPath)
            Log.d(TAG, "ExecuTorch Combined module ready from path: $modelPath")
        } catch (e: Exception) {
            objectDetectorListener?.onError("Failed to load model.pte: ${e.message}")
            Log.e(TAG, "Module creation failed", e)
        }
    }

    private fun letterboxBitmap(src: Bitmap, size: Int): Bitmap {
        val scale = size.toFloat() / maxOf(src.width, src.height)
        val newW = (src.width * scale).toInt()
        val newH = (src.height * scale).toInt()
        val resized = Bitmap.createScaledBitmap(src, newW, newH, true)

        val padded = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(padded)
        canvas.drawColor(android.graphics.Color.BLACK)  // czarne paski
        canvas.drawBitmap(resized,
            ((size - newW) / 2).toFloat(),
            ((size - newH) / 2).toFloat(),
            null)
        return padded
    }

    fun detect(image: Bitmap, imageRotation: Int) {
        val module = executorchModule ?: return

        val start = SystemClock.uptimeMillis()

        // 1. Prostowanie obrazu w zależności od fizycznej orientacji telefonu
        val rotatedBitmap = if (imageRotation != 0) {
            val matrix = Matrix().apply { postRotate(imageRotation.toFloat()) }
            Bitmap.createBitmap(image, 0, 0, image.width, image.height, matrix, true)
        } else {
            image
        }

        // 2. Skalowanie wyprostowanego obrazu do rozdzielczości 640x640
        val scaled = letterboxBitmap(rotatedBitmap, inputSize)
        val inputBuf = bitmapToNchw(scaled)

        val inputShape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        val inputTensor = Tensor.fromBlob(inputBuf, inputShape)

        // 3. Wywołanie inferencji na silniku ExecuTorch
        val outputs: Array<EValue> = try {
            module.forward(EValue.from(inputTensor))
        } catch (e: Exception) {
            objectDetectorListener?.onError("Inference error: ${e.message}")
            Log.e(TAG, "Inference failed", e)
            return
        }

        val inferenceTime = SystemClock.uptimeMillis() - start
        Log.d(TAG, "Inference done in ${inferenceTime}ms")

        val detections = parseOutputs(outputs, image.width, image.height)

        objectDetectorListener?.onResults(detections, inferenceTime, image.height, image.width)
    }

    // Alokacja pamięci natywnej (Direct Buffer) i wysyłanie surowych wartości [0.0 - 255.0]
    private fun bitmapToNchw(bitmap: Bitmap): FloatBuffer {
        val n = inputSize * inputSize
        val pixels = IntArray(n)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        val byteBuffer = ByteBuffer.allocateDirect(3 * n * 4)
        byteBuffer.order(ByteOrder.nativeOrder())
        val buf = byteBuffer.asFloatBuffer()

        for (i in 0 until n) {
            val px = pixels[i]
            buf.put(i,       (((px shr 16) and 0xFF).toFloat())) // R
            buf.put(i + n,   (((px shr 8)  and 0xFF).toFloat())) // G
            buf.put(i + n*2, (( px         and 0xFF).toFloat())) // B
        }

        buf.rewind()
        return buf
    }

    private fun parseOutputs(outputs: Array<EValue>?, origW: Int, origH: Int): List<PlateDetection> {
        try {
            if (outputs == null || outputs.isEmpty()) return emptyList()

            val boxTensor = outputs[0].toTensor()
            val data = boxTensor.dataAsFloatArray
            if (data.size < 5) return emptyList()

            val rawX1 = data[0]
            val rawY1 = data[1]
            val rawX2 = data[2]
            val rawY2 = data[3]
            val conf = data[4]

            // Filtr confidence
            if (conf < threshold) return emptyList()

            // Odskalowanie letterbox 960×960 → oryginalne wymiary
            val scale = inputSize.toFloat() / maxOf(origW, origH)
            val padX = (inputSize - origW * scale) / 2f
            val padY = (inputSize - origH * scale) / 2f

            val x1 = ((rawX1 - padX) / scale).coerceIn(0f, origW.toFloat())
            val y1 = ((rawY1 - padY) / scale).coerceIn(0f, origH.toFloat())
            val x2 = ((rawX2 - padX) / scale).coerceIn(0f, origW.toFloat())
            val y2 = ((rawY2 - padY) / scale).coerceIn(0f, origH.toFloat())

            val charTensor = if (outputs.size > 1) outputs[1].toTensor() else null
            val plateText = charTensor?.let { decodePlateChars(it) } ?: ""
            val cleanText = plateText.trim().replace(" ", "")

            if (cleanText.length < 3) return emptyList()

            Log.d(TAG, "Plate detected! Box: [$x1, $y1, $x2, $y2] Conf: $conf Text: $cleanText")

            return listOf(PlateDetection(
                boundingBox = RectF(x1, y1, x2, y2),
                confidence = conf,
                classLabel = "plate",
                plateText = cleanText
            ))
        } catch (e: Exception) {
            Log.e(TAG, "Error during parseOutputs", e)
            return emptyList()
        }
    }

    private fun decodePlateChars(tensor: Tensor): String {
        val shape = tensor.shape()
        if (shape.size < 3) return ""
        val numPos  = shape[1].toInt()
        val vocabSz = shape[2].toInt()
        val data    = tensor.dataAsFloatArray

        return buildString {
            for (pos in 0 until numPos) {
                var bestIdx = 0
                var bestVal = Float.NEGATIVE_INFINITY
                for (v in 0 until vocabSz) {
                    val score = data[pos * vocabSz + v]
                    if (score > bestVal) {
                        bestVal = score
                        bestIdx = v
                    }
                }
                append(if (bestIdx < plateAlphabet.length) plateAlphabet[bestIdx] else ' ')
            }
        }.trimEnd()
    }

    private fun getAssetFilePath(context: Context, assetName: String): String {
        val file = File(context.cacheDir, assetName)
        file.delete()
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
        return file.absolutePath
    }

    interface DetectorListener {
        fun onError(error: String)
        fun onResults(results: List<PlateDetection>, inferenceTime: Long, imageHeight: Int, imageWidth: Int)
    }

    companion object {
        private const val TAG = "ObjectDetectorHelper"
    }
}