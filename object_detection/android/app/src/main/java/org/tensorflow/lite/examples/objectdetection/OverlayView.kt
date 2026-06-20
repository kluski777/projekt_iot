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

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results: List<PlateDetection> = emptyList()

    // Niezależne skale dla osi X i Y
    private var scaleX: Float = 1f
    private var scaleY: Float = 1f

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

        // Ciemne, lekko przezroczyste tło pod rozpoznanym tekstem tablicy
        plateBgPaint.color = Color.argb(220, 30, 30, 30)
        plateBgPaint.style = Paint.Style.FILL

        // Żółty, czytelny tekst
        plateTextPaint.color        = Color.YELLOW
        plateTextPaint.style        = Paint.Style.FILL
        plateTextPaint.textSize     = 64f
        plateTextPaint.isFakeBoldText = true
        plateTextPaint.letterSpacing  = 0.1f
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        for (det in results) {
            val left   = det.boundingBox.left   * scaleX
            val bottom = det.boundingBox.bottom * scaleY

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
        scaleX = width.toFloat() / imageWidth.toFloat()
        scaleY = height.toFloat() / imageHeight.toFloat()
    }

    companion object {
        // Brakująca stała wywołująca błąd
        private const val PAD = 8f
    }
}