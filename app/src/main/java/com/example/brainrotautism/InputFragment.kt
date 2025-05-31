package com.example.brainrotautism

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader

class InputFragment : Fragment() {

    private val eegFiles = listOf(
        "autism_sample_1.csv", "autism_sample_2.csv", "autism_sample_3.csv",
        "normal_sample_1.csv", "normal_sample_2.csv", "normal_sample_3.csv"
    )

    private val sharedViewModel: SharedViewModel by activityViewModels()

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        return inflater.inflate(R.layout.fragment_input, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val loadEEGButton = view.findViewById<Button>(R.id.loadEEGButton)
        loadEEGButton.setOnClickListener {
            loadCSVData()
        }
    }

    private fun loadCSVData() {
        val randomFile = eegFiles.random()
        try {
            val inputStream = requireContext().assets.open(randomFile)
            val reader = BufferedReader(InputStreamReader(inputStream))
            val lines = reader.readLines()
            reader.close()

            sharedViewModel.setCSVData(lines)
            sharedViewModel.setSelectedEEGFile(randomFile)

            Toast.makeText(requireContext(), "Loaded: $randomFile", Toast.LENGTH_SHORT).show()
        } catch (e: IOException) {
            e.printStackTrace()
            Toast.makeText(requireContext(), "Error loading EEG CSV file", Toast.LENGTH_SHORT)
                .show()
        }
    }
}
