package org.tensorflow.lite.examples.objectdetection;

/**
 * Runs plate_recognizer.pte (YOLO + OCR Combined) via ExecuTorch 1.3.1.
 *
 * Input  "images"  : [1, 3, 640, 640]  float32, [0-255] raw pixels, NCHW
 * Output 0 (tuple) : best plate box [1, 4] — x1 y1 x2 y2 in 640×640 pixel space
 * Output 1 (tuple) : plate characters [1, 8, 37]
 * Alphabet: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ " (space = pad / unknown)
 */
@kotlin.Metadata(mv = {1, 8, 0}, k = 1, xi = 48, d1 = {"\u0000j\n\u0002\u0018\u0002\n\u0002\u0010\u0000\n\u0000\n\u0002\u0010\u0007\n\u0002\b\u0002\n\u0002\u0010\b\n\u0002\b\u0002\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0002\b\u0004\n\u0002\u0018\u0002\n\u0002\b\u000e\n\u0002\u0010\u000e\n\u0002\b\u0003\n\u0002\u0018\u0002\n\u0000\n\u0002\u0018\u0002\n\u0000\n\u0002\u0010\u0002\n\u0002\b\u0002\n\u0002\u0018\u0002\n\u0002\b\t\n\u0002\u0010 \n\u0002\u0018\u0002\n\u0000\n\u0002\u0010\u0011\n\u0002\u0018\u0002\n\u0002\b\u0007\u0018\u0000 =2\u00020\u0001:\u0002=>B?\u0012\b\b\u0002\u0010\u0002\u001a\u00020\u0003\u0012\b\b\u0002\u0010\u0004\u001a\u00020\u0003\u0012\b\b\u0002\u0010\u0005\u001a\u00020\u0006\u0012\b\b\u0002\u0010\u0007\u001a\u00020\u0006\u0012\u0006\u0010\b\u001a\u00020\t\u0012\b\u0010\n\u001a\u0004\u0018\u00010\u000b\u00a2\u0006\u0002\u0010\fJ\u0010\u0010\"\u001a\u00020#2\u0006\u0010$\u001a\u00020%H\u0002J\u0006\u0010&\u001a\u00020\'J\u0010\u0010(\u001a\u00020\u001f2\u0006\u0010)\u001a\u00020*H\u0002J\u0016\u0010+\u001a\u00020\'2\u0006\u0010,\u001a\u00020%2\u0006\u0010-\u001a\u00020\u0006J\u0018\u0010.\u001a\u00020\u001f2\u0006\u0010\b\u001a\u00020\t2\u0006\u0010/\u001a\u00020\u001fH\u0002J\u0018\u00100\u001a\u00020%2\u0006\u00101\u001a\u00020%2\u0006\u00102\u001a\u00020\u0006H\u0002J3\u00103\u001a\b\u0012\u0004\u0012\u000205042\u000e\u00106\u001a\n\u0012\u0004\u0012\u000208\u0018\u0001072\u0006\u00109\u001a\u00020\u00062\u0006\u0010:\u001a\u00020\u0006H\u0002\u00a2\u0006\u0002\u0010;J\u0006\u0010<\u001a\u00020\'R\u0011\u0010\b\u001a\u00020\t\u00a2\u0006\b\n\u0000\u001a\u0004\b\r\u0010\u000eR\u0010\u0010\u000f\u001a\u0004\u0018\u00010\u0010X\u0082\u000e\u00a2\u0006\u0002\n\u0000R\u000e\u0010\u0011\u001a\u00020\u0006X\u0082D\u00a2\u0006\u0002\n\u0000R\u001a\u0010\u0004\u001a\u00020\u0003X\u0086\u000e\u00a2\u0006\u000e\n\u0000\u001a\u0004\b\u0012\u0010\u0013\"\u0004\b\u0014\u0010\u0015R\u001a\u0010\u0005\u001a\u00020\u0006X\u0086\u000e\u00a2\u0006\u000e\n\u0000\u001a\u0004\b\u0016\u0010\u0017\"\u0004\b\u0018\u0010\u0019R\u001a\u0010\u0007\u001a\u00020\u0006X\u0086\u000e\u00a2\u0006\u000e\n\u0000\u001a\u0004\b\u001a\u0010\u0017\"\u0004\b\u001b\u0010\u0019R\u0013\u0010\n\u001a\u0004\u0018\u00010\u000b\u00a2\u0006\b\n\u0000\u001a\u0004\b\u001c\u0010\u001dR\u000e\u0010\u001e\u001a\u00020\u001fX\u0082D\u00a2\u0006\u0002\n\u0000R\u001a\u0010\u0002\u001a\u00020\u0003X\u0086\u000e\u00a2\u0006\u000e\n\u0000\u001a\u0004\b \u0010\u0013\"\u0004\b!\u0010\u0015\u00a8\u0006?"}, d2 = {"Lorg/tensorflow/lite/examples/objectdetection/ObjectDetectorHelper;", "", "threshold", "", "iouThreshold", "maxResults", "", "numThreads", "context", "Landroid/content/Context;", "objectDetectorListener", "Lorg/tensorflow/lite/examples/objectdetection/ObjectDetectorHelper$DetectorListener;", "(FFIILandroid/content/Context;Lorg/tensorflow/lite/examples/objectdetection/ObjectDetectorHelper$DetectorListener;)V", "getContext", "()Landroid/content/Context;", "executorchModule", "Lorg/pytorch/executorch/Module;", "inputSize", "getIouThreshold", "()F", "setIouThreshold", "(F)V", "getMaxResults", "()I", "setMaxResults", "(I)V", "getNumThreads", "setNumThreads", "getObjectDetectorListener", "()Lorg/tensorflow/lite/examples/objectdetection/ObjectDetectorHelper$DetectorListener;", "plateAlphabet", "", "getThreshold", "setThreshold", "bitmapToNchw", "Ljava/nio/FloatBuffer;", "bitmap", "Landroid/graphics/Bitmap;", "clearObjectDetector", "", "decodePlateChars", "tensor", "Lorg/pytorch/executorch/Tensor;", "detect", "image", "imageRotation", "getAssetFilePath", "assetName", "letterboxBitmap", "src", "size", "parseOutputs", "", "Lorg/tensorflow/lite/examples/objectdetection/PlateDetection;", "outputs", "", "Lorg/pytorch/executorch/EValue;", "origW", "origH", "([Lorg/pytorch/executorch/EValue;II)Ljava/util/List;", "setupObjectDetector", "Companion", "DetectorListener", "app_debug"})
public final class ObjectDetectorHelper {
    private float threshold;
    private float iouThreshold;
    private int maxResults;
    private int numThreads;
    @org.jetbrains.annotations.NotNull
    private final android.content.Context context = null;
    @org.jetbrains.annotations.Nullable
    private final org.tensorflow.lite.examples.objectdetection.ObjectDetectorHelper.DetectorListener objectDetectorListener = null;
    @org.jetbrains.annotations.Nullable
    private org.pytorch.executorch.Module executorchModule;
    private final int inputSize = 640;
    @org.jetbrains.annotations.NotNull
    private final java.lang.String plateAlphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ";
    @org.jetbrains.annotations.NotNull
    private static final java.lang.String TAG = "ObjectDetectorHelper";
    @org.jetbrains.annotations.NotNull
    public static final org.tensorflow.lite.examples.objectdetection.ObjectDetectorHelper.Companion Companion = null;
    
    public ObjectDetectorHelper(float threshold, float iouThreshold, int maxResults, int numThreads, @org.jetbrains.annotations.NotNull
    android.content.Context context, @org.jetbrains.annotations.Nullable
    org.tensorflow.lite.examples.objectdetection.ObjectDetectorHelper.DetectorListener objectDetectorListener) {
        super();
    }
    
    public final float getThreshold() {
        return 0.0F;
    }
    
    public final void setThreshold(float p0) {
    }
    
    public final float getIouThreshold() {
        return 0.0F;
    }
    
    public final void setIouThreshold(float p0) {
    }
    
    public final int getMaxResults() {
        return 0;
    }
    
    public final void setMaxResults(int p0) {
    }
    
    public final int getNumThreads() {
        return 0;
    }
    
    public final void setNumThreads(int p0) {
    }
    
    @org.jetbrains.annotations.NotNull
    public final android.content.Context getContext() {
        return null;
    }
    
    @org.jetbrains.annotations.Nullable
    public final org.tensorflow.lite.examples.objectdetection.ObjectDetectorHelper.DetectorListener getObjectDetectorListener() {
        return null;
    }
    
    public final void clearObjectDetector() {
    }
    
    public final void setupObjectDetector() {
    }
    
    private final android.graphics.Bitmap letterboxBitmap(android.graphics.Bitmap src, int size) {
        return null;
    }
    
    public final void detect(@org.jetbrains.annotations.NotNull
    android.graphics.Bitmap image, int imageRotation) {
    }
    
    private final java.nio.FloatBuffer bitmapToNchw(android.graphics.Bitmap bitmap) {
        return null;
    }
    
    private final java.util.List<org.tensorflow.lite.examples.objectdetection.PlateDetection> parseOutputs(org.pytorch.executorch.EValue[] outputs, int origW, int origH) {
        return null;
    }
    
    private final java.lang.String decodePlateChars(org.pytorch.executorch.Tensor tensor) {
        return null;
    }
    
    private final java.lang.String getAssetFilePath(android.content.Context context, java.lang.String assetName) {
        return null;
    }
    
    @kotlin.Metadata(mv = {1, 8, 0}, k = 1, xi = 48, d1 = {"\u0000\u0012\n\u0002\u0018\u0002\n\u0002\u0010\u0000\n\u0002\b\u0002\n\u0002\u0010\u000e\n\u0000\b\u0086\u0003\u0018\u00002\u00020\u0001B\u0007\b\u0002\u00a2\u0006\u0002\u0010\u0002R\u000e\u0010\u0003\u001a\u00020\u0004X\u0082T\u00a2\u0006\u0002\n\u0000\u00a8\u0006\u0005"}, d2 = {"Lorg/tensorflow/lite/examples/objectdetection/ObjectDetectorHelper$Companion;", "", "()V", "TAG", "", "app_debug"})
    public static final class Companion {
        
        private Companion() {
            super();
        }
    }
    
    @kotlin.Metadata(mv = {1, 8, 0}, k = 1, xi = 48, d1 = {"\u00000\n\u0002\u0018\u0002\n\u0002\u0010\u0000\n\u0000\n\u0002\u0010\u0002\n\u0000\n\u0002\u0010\u000e\n\u0002\b\u0002\n\u0002\u0010 \n\u0002\u0018\u0002\n\u0000\n\u0002\u0010\t\n\u0000\n\u0002\u0010\b\n\u0002\b\u0002\bf\u0018\u00002\u00020\u0001J\u0010\u0010\u0002\u001a\u00020\u00032\u0006\u0010\u0004\u001a\u00020\u0005H&J.\u0010\u0006\u001a\u00020\u00032\f\u0010\u0007\u001a\b\u0012\u0004\u0012\u00020\t0\b2\u0006\u0010\n\u001a\u00020\u000b2\u0006\u0010\f\u001a\u00020\r2\u0006\u0010\u000e\u001a\u00020\rH&\u00a8\u0006\u000f"}, d2 = {"Lorg/tensorflow/lite/examples/objectdetection/ObjectDetectorHelper$DetectorListener;", "", "onError", "", "error", "", "onResults", "results", "", "Lorg/tensorflow/lite/examples/objectdetection/PlateDetection;", "inferenceTime", "", "imageHeight", "", "imageWidth", "app_debug"})
    public static abstract interface DetectorListener {
        
        public abstract void onError(@org.jetbrains.annotations.NotNull
        java.lang.String error);
        
        public abstract void onResults(@org.jetbrains.annotations.NotNull
        java.util.List<org.tensorflow.lite.examples.objectdetection.PlateDetection> results, long inferenceTime, int imageHeight, int imageWidth);
    }
}