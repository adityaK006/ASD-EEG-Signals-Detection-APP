package com.example.brainrotautism

import android.content.res.AssetManager
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.flex.FlexDelegate
import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.random.Random

class MainActivity : AppCompatActivity() {

    private lateinit var loadButton: Button
    private lateinit var predictButton: Button
    private lateinit var resultTextView: TextView

    private lateinit var interpreter: Interpreter
    private var loadedInput: Array<Array<FloatArray>>? = null

    private val eegFiles = listOf(
        "autism_sample_1.csv", "autism_sample_2.csv", "autism_sample_3.csv",
        "normal_sample_1.csv", "normal_sample_2.csv", "normal_sample_3.csv"
    )

    private val sharedViewModel: SharedViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        loadButton = findViewById(R.id.loadEEGButton)
        predictButton = findViewById(R.id.predictButton)
        resultTextView = findViewById(R.id.resultTextView)

        try {
            val options = Interpreter.Options().addDelegate(FlexDelegate())
            interpreter = Interpreter(loadModelFile(assets, "eeg_rnn_tf_model.tflite"), options)
            Log.d("EEG", "Model loaded successfully")
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to load model", Toast.LENGTH_LONG).show()
            Log.e("EEG", "Model load error", e)
        }

        loadButton.setOnClickListener { loadCSVData() }

        predictButton.setOnClickListener {
            if (loadedInput == null) {
                Toast.makeText(this, "Please load EEG signal first", Toast.LENGTH_SHORT).show()
            } else {
                val result = runInference(loadedInput!!)
                resultTextView.text = "Prediction: $result"
            }
        }
    }

    private fun loadCSVData() {
        val randomFile = eegFiles.random()
        try {
            val inputStream = assets.open(randomFile)
            val reader = BufferedReader(InputStreamReader(inputStream))
            val flatList = mutableListOf<Float>()

            reader.useLines { lines ->
                lines.forEach { line ->
                    val clean = line.trim()
                    if (clean.isNotEmpty() && clean != "########") {
                        try {
                            flatList.add(clean.toFloat())
                        } catch (_: NumberFormatException) {
                            // skip invalid rows like headers or symbols
                        }
                    }
                }
            }

            if (flatList.size < 350 * 16) {
                Toast.makeText(this, "EEG file has insufficient data", Toast.LENGTH_SHORT).show()
                return
            }

            val result = Array(350) { row ->
                FloatArray(16) { col ->
                    flatList[row * 16 + col]
                }
            }

            loadedInput = arrayOf(result) // shape: (1, 350, 16)
            Toast.makeText(this, "Loaded: $randomFile", Toast.LENGTH_SHORT).show()
            resultTextView.text = "Signal loaded: $randomFile"

        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "Error loading EEG file", Toast.LENGTH_SHORT).show()
        }
    }


    private fun runInference(input: Array<Array<FloatArray>>): String {
        val byteBuffer = ByteBuffer.allocateDirect(1 * 350 * 16 * 4)
        byteBuffer.order(ByteOrder.nativeOrder())

        for (i in 0 until 350) {
            for (j in 0 until 16) {
                byteBuffer.putFloat(input[0][i][j])
            }
        }

        val output = Array(1) { FloatArray(2) }
        interpreter.run(byteBuffer, output)

        val predictedIndex = output[0].indexOfFirst { it == output[0].maxOrNull() }
        return if (predictedIndex == 0) "Autism" else "Normal"
    }

    private fun loadModelFile(assetManager: AssetManager, fileName: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(fileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }
}
