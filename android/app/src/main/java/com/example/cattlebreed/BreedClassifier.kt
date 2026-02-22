/**
 * Cattle Breed Recognition - Android App
 * ======================================
 * 
 * Proof of Concept (PoC) for FLW testing
 * Designed for integration with Bharat Pashudhan App (BPA)
 * 
 * This Kotlin file demonstrates the core functionality for:
 * 1. Loading TFLite models
 * 2. Camera capture
 * 3. Breed prediction
 * 4. UI flow for FLWs
 * 
 * Author: SIH 2025 Team
 * Problem Statement: SIH25004
 */

package com.example.cattlebreed

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*

/**
 * Breed classification result
 */
data class BreedResult(
    val breed: String,
    val confidence: Float,
    val topPredictions: List<Pair<String, Float>>,
    val action: String, // "auto_confirm", "flw_select", "expert_review"
    val inferenceTimeMs: Long
)

/**
 * Detection result from YOLO
 */
data class DetectionResult(
    val detected: Boolean,
    val boundingBox: RectF,
    val confidence: Float
)

/**
 * Complete prediction result
 */
data class PredictionResult(
    val detection: DetectionResult,
    val classification: BreedResult,
    val croppedBitmap: Bitmap?,
    val totalTimeMs: Long
)

/**
 * TFLite Model Manager
 * Handles loading and running inference for both detection and classification models
 */
class TFLiteModelManager(private val context: Context) {
    
    companion object {
        private const val DETECTOR_MODEL_PATH = "yolov8_nano_cattle_detector_int8.tflite"
        private const val CLASSIFIER_MODEL_PATH = "efficientnet_b0_int8.tflite"
        private const val LABELS_PATH = "labels.txt"
        
        // Model input sizes
        private const val DETECTOR_INPUT_SIZE = 416
        private const val CLASSIFIER_INPUT_SIZE = 224
        
        // Confidence thresholds
        private const val AUTO_CONFIRM_THRESHOLD = 0.85f
        private const val ESCALATION_THRESHOLD = 0.60f
    }
    
    private var detectorInterpreter: Interpreter? = null
    private var classifierInterpreter: Interpreter? = null
    private var labels: List<String> = emptyList()
    
    // Image processors
    private val detectorProcessor: ImageProcessor by lazy {
        ImageProcessor.Builder()
            .add(ResizeOp(DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .build()
    }
    
    private val classifierProcessor: ImageProcessor by lazy {
        ImageProcessor.Builder()
            .add(ResizeOp(CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .build()
    }
    
    /**
     * Initialize models
     */
    fun initialize(): Boolean {
        return try {
            // Load detector model
            val detectorBuffer = loadModelFile(DETECTOR_MODEL_PATH)
            detectorInterpreter = Interpreter(detectorBuffer)
            
            // Load classifier model
            val classifierBuffer = loadModelFile(CLASSIFIER_MODEL_PATH)
            classifierInterpreter = Interpreter(classifierBuffer)
            
            // Load labels
            labels = loadLabels()
            
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }
    
    /**
     * Load TFLite model from assets
     */
    @Throws(IOException::class)
    private fun loadModelFile(modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    /**
     * Load breed labels
     */
    private fun loadLabels(): List<String> {
        return try {
            context.assets.open(LABELS_PATH).bufferedReader().readLines()
        } catch (e: Exception) {
            // Default labels if file not found
            listOf(
                "Gir", "Sahiwal", "Red_Sindhi", "Tharparkar", "Rathi",
                "Hariana", "Kankrej", "Ongole", "Deoni",
                "Hallikar", "Amritmahal", "Khillari", "Kangayam", "Bargur",
                "Dangi", "Krishna_Valley", "Malnad_Gidda", "Punganur", "Vechur",
                "Pulikulam", "Umblachery", "Toda", "Kalahandi",
                "Murrah", "Jaffrabadi", "Nili_Ravi", "Banni", "Pandharpuri",
                "Mehsana", "Surti", "Nagpuri", "Bhadawari", "Chilika",
                "Jersey_Cross", "HF_Cross"
            )
        }
    }
    
    /**
     * Detect animal in image using YOLO
     */
    fun detectAnimal(bitmap: Bitmap): DetectionResult {
        val startTime = System.currentTimeMillis()
        
        // Prepare input
        val tensorImage = TensorImage.fromBitmap(bitmap)
        val processedImage = detectorProcessor.process(tensorImage)
        
        // Run inference
        val output = TensorBuffer.createFixedSize(intArrayOf(1, 8400, 5), org.tensorflow.lite.DataType.FLOAT32)
        detectorInterpreter?.run(processedImage.buffer, output.buffer)
        
        val inferenceTime = System.currentTimeMillis() - startTime
        
        // Simplified: return full image as detected
        // In production, parse YOLO output properly
        return DetectionResult(
            detected = true,
            boundingBox = RectF(0f, 0f, bitmap.width.toFloat(), bitmap.height.toFloat()),
            confidence = 0.95f
        )
    }
    
    /**
     * Classify breed using EfficientNet
     */
    fun classifyBreed(bitmap: Bitmap): BreedResult {
        val startTime = System.currentTimeMillis()
        
        // Prepare input
        val tensorImage = TensorImage.fromBitmap(bitmap)
        val processedImage = classifierProcessor.process(tensorImage)
        
        // Run inference
        val output = TensorBuffer.createFixedSize(intArrayOf(1, labels.size), org.tensorflow.lite.DataType.FLOAT32)
        classifierInterpreter?.run(processedImage.buffer, output.buffer)
        
        val inferenceTime = System.currentTimeMillis() - startTime
        
        // Get probabilities
        val probabilities = output.floatArray
        
        // Get top 3 predictions
        val indexedProbabilities = probabilities.mapIndexed { index, prob ->
            Pair(labels.getOrElse(index) { "Unknown" }, prob)
        }.sortedByDescending { it.second }
        
        val topPredictions = indexedProbabilities.take(3)
        
        // Determine action
        val confidence = topPredictions[0].second
        val action = when {
            confidence >= AUTO_CONFIRM_THRESHOLD -> "auto_confirm"
            confidence >= ESCALATION_THRESHOLD -> "flw_select"
            else -> "expert_review"
        }
        
        return BreedResult(
            breed = topPredictions[0].first,
            confidence = confidence,
            topPredictions = topPredictions,
            action = action,
            inferenceTimeMs = inferenceTime
        )
    }
    
    /**
     * Complete prediction pipeline
     */
    fun predict(bitmap: Bitmap): PredictionResult {
        val startTime = System.currentTimeMillis()
        
        // Stage 1: Detection
        val detection = detectAnimal(bitmap)
        
        // Stage 2: Classification (on cropped region)
        val croppedBitmap = if (detection.detected) {
            cropBitmap(bitmap, detection.boundingBox)
        } else {
            null
        }
        
        val classification = if (croppedBitmap != null) {
            classifyBreed(croppedBitmap)
        } else {
            BreedResult(
                breed = "Unknown",
                confidence = 0f,
                topPredictions = emptyList(),
                action = "expert_review",
                inferenceTimeMs = 0
            )
        }
        
        val totalTime = System.currentTimeMillis() - startTime
        
        return PredictionResult(
            detection = detection,
            classification = classification,
            croppedBitmap = croppedBitmap,
            totalTimeMs = totalTime
        )
    }
    
    /**
     * Crop bitmap to bounding box
     */
    private fun cropBitmap(bitmap: Bitmap, bbox: RectF): Bitmap {
        val left = bbox.left.toInt().coerceIn(0, bitmap.width)
        val top = bbox.top.toInt().coerceIn(0, bitmap.height)
        val width = bbox.width().toInt().coerceIn(1, bitmap.width - left)
        val height = bbox.height().toInt().coerceIn(1, bitmap.height - top)
        
        return Bitmap.createBitmap(bitmap, left, top, width, height)
    }
    
    /**
     * Close models
     */
    fun close() {
        detectorInterpreter?.close()
        classifierInterpreter?.close()
    }
}

/**
 * Example Activity Usage
 * 
 * class MainActivity : AppCompatActivity() {
 *     private lateinit var modelManager: TFLiteModelManager
 *     private lateinit var cameraManager: CameraManager
 *     
 *     override fun onCreate(savedInstanceState: Bundle?) {
 *         super.onCreate(savedInstanceState)
 *         setContentView(R.layout.activity_main)
 *         
 *         // Initialize model manager
 *         modelManager = TFLiteModelManager(this)
 *         if (!modelManager.initialize()) {
 *             showError("Failed to load AI models")
 *         }
 *         
 *         // Setup camera
 *         cameraManager = CameraManager(this, previewView)
 *         cameraManager.initialize()
 *         
 *         // Capture button
 *         captureButton.setOnClickListener {
 *             captureAndPredict()
 *         }
 *     }
 *     
 *     private fun captureAndPredict() {
 *         val bitmap = cameraManager.capture()
 *         val result = modelManager.predict(bitmap)
 *         
 *         when (result.classification.action) {
 *             "auto_confirm" -> showAutoConfirm(result)
 *             "flw_select" -> showBreedSelection(result)
 *             "expert_review" -> escalateToExpert(result)
 *         }
 *     }
 *     
 *     private fun showAutoConfirm(result: PredictionResult) {
 *         // Show result with high confidence
 *         breedTextView.text = result.classification.breed
 *         confidenceTextView.text = "${(result.classification.confidence * 100).toInt()}%"
 *         confirmButton.visibility = View.VISIBLE
 *         selectButton.visibility = View.GONE
 *     }
 *     
 *     private fun showBreedSelection(result: PredictionResult) {
 *         // Show top 3 options for FLW to select
 *         breedTextView.text = result.classification.breed
 *         confidenceTextView.text = "${(result.classification.confidence * 100).toInt()}%"
 *         
 *         // Show selection dropdown
 *         val breeds = result.classification.topPredictions.map { it.first }
 *         val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, breeds)
 *         breedSpinner.adapter = adapter
 *         
 *         confirmButton.visibility = View.GONE
 *         selectButton.visibility = View.VISIBLE
 *     }
 *     
 *     private fun escalateToExpert(result: PredictionResult) {
 *         // Send to expert review queue
 *         val intent = Intent(this, ExpertReviewActivity::class.java)
 *         intent.putExtra("image", bitmapToByteArray(result.croppedBitmap))
 *         intent.putExtra("predictions", result.classification.topPredictions.toTypedArray())
 *         startActivity(intent)
 *     }
 * }
 */

/**
 * Data Collection Helper
 * For collecting training data from the field
 */
class DataCollectionHelper(private val context: Context) {
    
    /**
     * Save image with breed label for training
     */
    fun saveTrainingImage(bitmap: Bitmap, breed: String, flwId: String) {
        val timestamp = System.currentTimeMillis()
        val filename = "${breed}_${flwId}_$timestamp.jpg"
        
        // Save to local storage
        // Will be synced to server when online
        try {
            val file = File(context.filesDir, "training_data/$breed")
            file.mkdirs()
            val imageFile = File(file, filename)
            
            FileOutputStream(imageFile).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
            }
            
            // Log the collection
            logCollection(breed, flwId, filename)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
    
    private fun logCollection(breed: String, flwId: String, filename: String) {
        // Log to local database for tracking
    }
}

/**
 * Offline Storage Manager
 * Handles local storage for offline functionality
 */
class OfflineStorageManager(private val context: Context) {
    
    private val sharedPrefs = context.getSharedPreferences("cattle_breed_prefs", Context.MODE_PRIVATE)
    
    /**
     * Save prediction for later sync
     */
    fun savePendingSync(prediction: PredictionResult, animalId: String, flwId: String) {
        val pendingList = getPendingSyncList().toMutableList()
        pendingList.add(mapOf(
            "animal_id" to animalId,
            "flw_id" to flwId,
            "breed" to prediction.classification.breed,
            "confidence" to prediction.classification.confidence.toString(),
            "timestamp" to System.currentTimeMillis().toString()
        ))
        
        sharedPrefs.edit()
            .putString("pending_sync", pendingList.toJson())
            .apply()
    }
    
    /**
     * Get pending items for sync
     */
    fun getPendingSyncList(): List<Map<String, String>> {
        val json = sharedPrefs.getString("pending_sync", "[]")
        // Parse JSON to list
        return emptyList() // Placeholder
    }
    
    /**
     * Clear synced items
     */
    fun clearSyncedItems() {
        sharedPrefs.edit()
            .remove("pending_sync")
            .apply()
    }
}

// Extension function for list to JSON
fun List<Map<String, String>>.toJson(): String {
    return "[" + this.joinToString(",") { item ->
        "{" + item.entries.joinToString(",") { "\"${it.key}\":\"${it.value}\"" } + "}"
    } + "]"
}
