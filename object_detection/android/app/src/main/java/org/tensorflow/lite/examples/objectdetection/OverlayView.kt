package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import kotlin.math.max

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results: List<PlateDetection> = emptyList()
    private var scaleFactor: Float = 1f

    private val boxPaint        = Paint()
    private val labelBgPaint    = Paint()
    private val labelTextPaint  = Paint()
    private val plateBgPaint    = Paint()
    private val plateTextPaint  = Paint()
    private val bounds          = Rect()

    init { initPaints() }

    fun clear() {
        results = emptyList()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        boxPaint.color       = ContextCompat.getColor(context!!, R.color.bounding_box_color)
        boxPaint.strokeWidth = 8f
        boxPaint.style       = Paint.Style.STROKE

        labelBgPaint.color   = Color.BLACK
        labelBgPaint.style   = Paint.Style.FILL
        labelBgPaint.textSize = 50f

        labelTextPaint.color    = Color.WHITE
        labelTextPaint.style    = Paint.Style.FILL
        labelTextPaint.textSize = 50f

        // Yellow pill behind the plate text
        plateBgPaint.color = Color.argb(220, 30, 30, 30)
        plateBgPaint.style = Paint.Style.FILL

        plateTextPaint.color        = Color.YELLOW
        plateTextPaint.style        = Paint.Style.FILL
        plateTextPaint.textSize     = 64f
        plateTextPaint.isFakeBoldText = true
        plateTextPaint.letterSpacing  = 0.1f
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        for (det in results) {
            val top    = det.boundingBox.top    * scaleFactor
            val bottom = det.boundingBox.bottom * scaleFactor
            val left   = det.boundingBox.left   * scaleFactor
            val right  = det.boundingBox.right  * scaleFactor

            // Bounding box
            canvas.drawRect(RectF(left, top, right, bottom), boxPaint)

            // Label + confidence above the box
            val label = "${det.classLabel} ${"%.2f".format(det.confidence)}"
            labelBgPaint.getTextBounds(label, 0, label.length, bounds)
            val lh = bounds.height().toFloat()
            val lw = bounds.width().toFloat()
            canvas.drawRect(left, top - lh - PAD, left + lw + PAD, top, labelBgPaint)
            canvas.drawText(label, left, top - PAD / 2f, labelTextPaint)

            // Plate text below the box (only when non-blank)
            val plate = det.plateText.trim()
            if (plate.isNotEmpty()) {
                plateTextPaint.getTextBounds(plate, 0, plate.length, bounds)
                val pw = bounds.width().toFloat()
                val ph = bounds.height().toFloat()
                canvas.drawRect(left, bottom, left + pw + PAD * 2, bottom + ph + PAD * 2, plateBgPaint)
                canvas.drawText(plate, left + PAD, bottom + ph + PAD / 2f, plateTextPaint)
            }
        }
    }

    fun setResults(detections: List<PlateDetection>, imageHeight: Int, imageWidth: Int) {
        results = detections
        scaleFactor = max(width * 1f / imageWidth, height * 1f / imageHeight)
    }

    companion object {
        private const val PAD = 8f
    }
}
