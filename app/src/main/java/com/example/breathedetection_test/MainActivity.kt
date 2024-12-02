package com.example.breathedetection_test

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.YuvImage
import android.media.Image.Plane
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import org.opencv.core.MatOfRect
import org.opencv.objdetect.CascadeClassifier
import org.opencv.core.Rect
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import java.io.FileOutputStream


class BreathingRateActivity : AppCompatActivity() {
    private val CAMERA_PERMISSION_REQUEST_CODE = 100

    private lateinit var resultTextView: TextView
    private lateinit var previewView: PreviewView
    private var faceDetector: CascadeClassifier? = null


    private val greenValues = mutableListOf<Double>()
    private val cameraExecutor = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        resultTextView = findViewById(R.id.resultTextView)
        previewView = findViewById(R.id.previewView)
        Log.d("MainActivity", "onCreate called")
        // 初始化 OpenCV和人臉檢測
        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "Unable to load OpenCV")
        } else {
            initializeFaceDetector()
            Log.d("OpenCV", "OpenCV successfully loaded")
        }
        checkPermissions()


        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                REQUIRED_PERMISSIONS,
                CAMERA_PERMISSION_REQUEST_CODE
            )
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            val cameraSelector = CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                .build()

            // 添加影像分析器
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { image ->
                        processImageProxy(image) // 傳遞影像幀到 processFrame 處理
                    }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer)
            } catch (exc: Exception) {
                Log.e("Camera", "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // 處理影像幀
    private fun processImageProxy(image: ImageProxy) {
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        // 合併 NV21 格式
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize) // V comes before U in NV21
        uBuffer.get(nv21, ySize + vSize, uSize)

        // 將 NV21 轉換為 Bitmap
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, image.width, image.height), 100, out)
        val bitmap = BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())

        // 調用 processFrame
        processFrame(bitmap)

        // 關閉 image 以釋放資源
        image.close()
    }


    private fun processFrame(bitmap: Bitmap) {
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)

        val file = File("${getExternalFilesDir(null)}/debug_image.jpg")
        FileOutputStream(file).use { fos ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos)
        }
        Log.d("Debug", "Saved debug image at: ${file.absolutePath}")


        if (faceDetector != null) {
            val grayMat = Mat()
            Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_BGR2GRAY)
            Core.flip(grayMat, grayMat, 1) // 翻轉影像
            Imgproc.equalizeHist(grayMat, grayMat) // 增強對比度

            // 縮放影像以提升檢測效率
            val resizedMat = Mat()
            Imgproc.resize(grayMat, resizedMat, org.opencv.core.Size(320.0, 240.0))
            Core.rotate(grayMat, grayMat, Core.ROTATE_90_CLOCKWISE)

            val faces = MatOfRect()
            faceDetector?.detectMultiScale(
                resizedMat, faces,
                1.05, // 調整縮放係數
                3,    // 調整最小鄰居數
                0,
                org.opencv.core.Size(20.0, 20.0) // 減小最小框大小
            )

            Imgcodecs.imwrite("${getExternalFilesDir(null)}/gray_image.jpg", grayMat)
            Log.d("Debug", "Saved gray image for inspection.")

            if (faces.toArray().isNotEmpty()) {
                Log.d("FaceDetection", "Detected ${faces.toArray().size} faces")
                val faceRect = faces.toArray()[0]
                val croppedMat = Mat(mat, faceRect)

                val greenChannel = Mat()
                Core.extractChannel(croppedMat, greenChannel, 1)

                val mean = Core.mean(greenChannel)
                greenValues.add(mean.`val`[0])

                if (greenValues.size > 50) {
                    val breathingRate = calculateBreathingRate(greenValues)
                    runOnUiThread {
                        resultTextView.text = "呼吸率: ${"%.2f".format(breathingRate)} 次/分鐘"
                    }
                }
                else{
                    runOnUiThread {
                        resultTextView.text = "未檢測到人臉"
                    }
                }
            } else {
                Log.d("FaceDetection", "No face detected.")
            }

            grayMat.release()
            resizedMat.release()
        } else {
            Log.e("FaceDetection", "Face detector not initialized.")
        }

        mat.release()
    }




    private fun calculateBreathingRate(values: List<Double>): Double {
        // 高斯平滑
        val smoothedValues = gaussianFilter(values)


        // 找峰值
        val peaks = findPeaks(smoothedValues)

        // 計算呼吸率
        return if (peaks.isNotEmpty()) {
            val avgPeriod = peaks.zipWithNext { a, b -> b - a }.average()
            60.0 / avgPeriod
        } else {
            0.0
        }
    }

    private fun gaussianFilter(values: List<Double>): List<Double> {
        // 簡單的高斯平滑實現
        return values.mapIndexed { index, value ->
            val start = maxOf(0, index - 3)
            val end = minOf(values.size - 1, index + 3)
            values.subList(start, end).average()
        }
    }

    private fun findPeaks(values: List<Double>): List<Int> {
        val threshold = 0.5 // 設置一個合適的閾值
        return values.indices.filter { index ->
            index > 0 && index < values.size - 1 &&
                    values[index] > values[index - 1] &&
                    values[index] > values[index + 1] &&
                    values[index] > threshold
        }
    }


    private fun initializeFaceDetector() {
        try {
            // 載入人臉檢測模型，將 haarcascade_frontalface_alt.xml 放到 res/raw
            //val inputStream = resources.openRawResource(R.raw.haarcascade_frontalface_alt)
            val inputStream = resources.openRawResource(R.raw.lbpcascade_frontalface)
            val cascadeDir = getDir("cascade", MODE_PRIVATE)
            val cascadeFile = File(cascadeDir, "lbpcascade_frontalface.xml")
            val outputStream = FileOutputStream(cascadeFile)

            val buffer = ByteArray(4096)
            var bytesRead: Int
            while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                outputStream.write(buffer, 0, bytesRead)
            }
            inputStream.close()
            outputStream.close()

            faceDetector = CascadeClassifier(cascadeFile.absolutePath)
            if (faceDetector?.empty() == true) {
                Log.e("faceDetector", "Failed to load face detector.")
                faceDetector = null
            } else {
                Log.d("FaceDetection", "Face detector is ready.")
            }


            cascadeFile.delete()
            cascadeDir.delete()
        } catch (e: Exception) {
            Log.e("OpenCV", "Error initializing face detector: ${e.message}")
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun checkPermissions() {
        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, CAMERA_PERMISSION_REQUEST_CODE
            )
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE) {
            if (allPermissionsGranted()) {
                Toast.makeText(this, "相機權限正常運行", Toast.LENGTH_SHORT).show()
                startCamera()
            } else {
                Toast.makeText(this, "需要相機權限才能正常運行", Toast.LENGTH_SHORT).show()
            }
        }
    }





    companion object {
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA
        )
    }


}