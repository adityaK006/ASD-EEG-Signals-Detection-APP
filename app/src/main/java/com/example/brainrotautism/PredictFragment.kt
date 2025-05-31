package com.example.brainrotautism

import android.content.res.AssetManager
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.flex.FlexDelegate // ✅ ADD THIS
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.Random

class PredictFragment : Fragment() {

    private lateinit var interpreter: Interpreter
    private lateinit var predictTextView: TextView
    private lateinit var loadSignalButton: Button
    private lateinit var predictButton: Button

    private var loadedInput: Array<Array<FloatArray>>? = null
    private val eegFiles = listOf(
        "autism_sample_1.csv", "autism_sample_2.csv", "autism_sample_3.csv",
        "normal_sample_1.csv", "normal_sample_2.csv", "normal_sample_3.csv"
    )

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_predict, container, false)

        predictTextView = view.findViewById(R.id.resultTextView)
        loadSignalButton = view.findViewById(R.id.loadSignalButton)
        predictButton = view.findViewById(R.id.predictButton)

        try {
            // ✅ ADD FlexDelegate to interpreter options
            val options = Interpreter.Options().addDelegate(FlexDelegate())
            interpreter = Interpreter(loadModelFile(requireContext().assets, "eeg_cnn_bilstm_model.tflite"), options)
            Log.d("EEG-Predict", "Model loaded successfully with FlexDelegate")
        } catch (e: Exception) {
            Log.e("EEG-Predict", "Error loading model", e)
            Toast.makeText(context, "Error loading model", Toast.LENGTH_SHORT).show()
        }

        loadSignalButton.setOnClickListener {
            val fileName = eegFiles[Random().nextInt(eegFiles.size)]
            try {
                loadedInput = loadCsvFromAssets(fileName)
                predictTextView.text = "Signal loaded: $fileName"

                val shapeInfo = "Loaded Input Shape: (${loadedInput!!.size}, ${loadedInput!![0].size}, ${loadedInput!![0][0].size})"
                Log.d("EEG-Predict", shapeInfo)

            } catch (e: Exception) {
                predictTextView.text = "Error loading EEG signal"
                e.printStackTrace()
            }
        }

        predictButton.setOnClickListener {
            if (!::interpreter.isInitialized) {
                predictTextView.text = "Model not loaded"
                Log.e("EEG-Predict", "Interpreter not initialized!")
                return@setOnClickListener
            }

            if (loadedInput != null) {
                try {
                    val prediction = runInference(loadedInput!!)
                    predictTextView.text = "Predicted: $prediction"
                } catch (e: Exception) {
                    predictTextView.text = "Prediction failed"
                    Log.e("EEG-Predict", "Prediction failed", e)
                }
            } else {
                predictTextView.text = "Please load an EEG signal first"
            }
        }

        return view
    }

    private fun runInference(input: Array<Array<FloatArray>>): String {
        Log.d("EEG-Predict", "Running inference on shape: [${input.size}, ${input[0].size}, ${input[0][0].size}]")

        val byteBuffer = ByteBuffer.allocateDirect(1 * 350 * 16 * 4)
        byteBuffer.order(ByteOrder.nativeOrder())

        for (i in 0 until 350) {
            for (j in 0 until 16) {
                byteBuffer.putFloat(input[0][i][j])
            }
        }

        val output = Array(1) { FloatArray(2) }
        interpreter.run(byteBuffer, output)

        Log.d("EEG-Predict", "Model Output: ${output[0].joinToString()}")

        val predictedClass = output[0].indexOfFirst { it == output[0].maxOrNull() }
        return if (predictedClass == 0) "Autism" else "Normal"
    }

    private fun loadModelFile(assetManager: AssetManager, modelFile: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(modelFile)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    private fun loadCsvFromAssets(fileName: String): Array<Array<FloatArray>> {
        val inputStream = requireContext().assets.open(fileName)
        val reader = InputStreamReader(inputStream)
        val lines = reader.readLines()
        reader.close()

        // Convert CSV data to a 2D FloatArray
        val result = Array(350) { FloatArray(16) }
        for (i in 0 until 350) {
            val line = lines[i].split(",").map { it.toFloat() }
            for (j in 0 until 16) {
                result[i][j] = line[j]
            }
        }

        return arrayOf(result)  // Shape: (1, 350, 16)
    }
}
