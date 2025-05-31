package com.example.brainrotautism

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel

class SharedViewModel : ViewModel() {
    private val _selectedEEGFile = MutableLiveData<String>()
    val selectedEEGFile: LiveData<String> = _selectedEEGFile

    private val _csvData = MutableLiveData<List<String>>()  // Stores raw CSV lines
    val csvData: LiveData<List<String>> = _csvData

    fun setSelectedEEGFile(fileName: String) {
        _selectedEEGFile.value = fileName
    }

    fun setCSVData(data: List<String>) {
        _csvData.value = data
    }

}
