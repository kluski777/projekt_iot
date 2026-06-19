package org.tensorflow.lite.examples.objectdetection.fragments

import android.annotation.SuppressLint
import android.content.res.Configuration
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.navigation.Navigation
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import org.tensorflow.lite.examples.objectdetection.ObjectDetectorHelper
import org.tensorflow.lite.examples.objectdetection.PlateDetection
import org.tensorflow.lite.examples.objectdetection.R
import org.tensorflow.lite.examples.objectdetection.databinding.FragmentCameraBinding

class CameraFragment : Fragment(), ObjectDetectorHelper.DetectorListener {

    private val TAG = "ObjectDetection"

    private var _binding: FragmentCameraBinding? = null
    private val binding get() = _binding!!

    private lateinit var detectorHelper: ObjectDetectorHelper
    private lateinit var bitmapBuffer: Bitmap

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null

    private lateinit var cameraExecutor: ExecutorService

    override fun onResume() {
        super.onResume()
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(requireActivity(), R.id.fragment_container)
                .navigate(CameraFragmentDirections.actionCameraToPermissions())
        }
    }

    override fun onDestroyView() {
        _binding = null
        super.onDestroyView()
        cameraExecutor.shutdown()
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentCameraBinding.inflate(inflater, container, false)
        return binding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        detectorHelper = ObjectDetectorHelper(
            context = requireContext(),
            objectDetectorListener = this
        )

        // Hide model / delegate spinners — not relevant for a fixed ONNX model
        binding.bottomSheetLayout.spinnerModel.visibility    = View.GONE
        binding.bottomSheetLayout.spinnerDelegate.visibility = View.GONE

        cameraExecutor = Executors.newSingleThreadExecutor()

        binding.viewFinder.post { setUpCamera() }

        initBottomSheetControls()
    }

    private fun initBottomSheetControls() {
        binding.bottomSheetLayout.thresholdMinus.setOnClickListener {
            if (detectorHelper.threshold >= 0.1f) {
                detectorHelper.threshold -= 0.1f
                updateControlsUi()
            }
        }
        binding.bottomSheetLayout.thresholdPlus.setOnClickListener {
            if (detectorHelper.threshold <= 0.8f) {
                detectorHelper.threshold += 0.1f
                updateControlsUi()
            }
        }
        binding.bottomSheetLayout.maxResultsMinus.setOnClickListener {
            if (detectorHelper.maxResults > 1) {
                detectorHelper.maxResults--
                updateControlsUi()
            }
        }
        binding.bottomSheetLayout.maxResultsPlus.setOnClickListener {
            if (detectorHelper.maxResults < 5) {
                detectorHelper.maxResults++
                updateControlsUi()
            }
        }
        binding.bottomSheetLayout.threadsMinus.setOnClickListener {
            if (detectorHelper.numThreads > 1) {
                detectorHelper.numThreads--
                updateControlsUi()
            }
        }
        binding.bottomSheetLayout.threadsPlus.setOnClickListener {
            if (detectorHelper.numThreads < 4) {
                detectorHelper.numThreads++
                updateControlsUi()
            }
        }
    }

    private fun updateControlsUi() {
        binding.bottomSheetLayout.maxResultsValue.text = detectorHelper.maxResults.toString()
        binding.bottomSheetLayout.thresholdValue.text  = "%.2f".format(detectorHelper.threshold)
        binding.bottomSheetLayout.threadsValue.text    = detectorHelper.numThreads.toString()
        detectorHelper.clearObjectDetector()
        binding.overlay.clear()
    }

    private fun setUpCamera() {
        val future = ProcessCameraProvider.getInstance(requireContext())
        future.addListener(
            { cameraProvider = future.get(); bindCameraUseCases() },
            ContextCompat.getMainExecutor(requireContext())
        )
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {
        val provider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")
        val selector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK).build()

        preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor) { image ->
                    if (!::bitmapBuffer.isInitialized) {
                        bitmapBuffer = Bitmap.createBitmap(
                            image.width, image.height, Bitmap.Config.ARGB_8888
                        )
                    }
                    detectObjects(image)
                }
            }

        provider.unbindAll()
        try {
            camera = provider.bindToLifecycle(this, selector, preview, imageAnalyzer)
            preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
        } catch (e: Exception) {
            Log.e(TAG, "Use case binding failed", e)
        }
    }

    private fun detectObjects(image: ImageProxy) {
        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }
        detectorHelper.detect(bitmapBuffer, image.imageInfo.rotationDegrees)
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalyzer?.targetRotation = binding.viewFinder.display.rotation
    }

    override fun onResults(
        results: List<PlateDetection>,
        inferenceTime: Long,
        imageHeight: Int,
        imageWidth: Int
    ) {
        results.forEachIndexed { i: Int, d: PlateDetection ->
            Log.d(TAG, "result[$i] box=${d.boundingBox} conf=${d.confidence} text=\"${d.plateText}\"")
        }
        if (results.isEmpty()) Log.d(TAG, "no detections this frame")

        activity?.runOnUiThread {
            binding.bottomSheetLayout.inferenceTimeVal.text = "${inferenceTime} ms"
            binding.overlay.setResults(results, imageHeight, imageWidth)
            binding.overlay.invalidate()
        }
    }

    override fun onError(error: String) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
        }
    }
}
