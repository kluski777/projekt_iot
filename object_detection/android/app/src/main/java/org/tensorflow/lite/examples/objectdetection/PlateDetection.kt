package org.tensorflow.lite.examples.objectdetection

import android.graphics.RectF

data class PlateDetection(
    val boundingBox: RectF,
    val confidence: Float,
    val classLabel: String,
    val plateText: String  // up to 8 characters; space used for empty/pad positions
)
