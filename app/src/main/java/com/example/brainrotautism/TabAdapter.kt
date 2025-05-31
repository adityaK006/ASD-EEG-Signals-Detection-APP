package com.example.brainrotautism

import androidx.fragment.app.Fragment
import androidx.fragment.app.FragmentActivity
import androidx.viewpager2.adapter.FragmentStateAdapter

class TabAdapter(fragmentActivity: FragmentActivity) : FragmentStateAdapter(fragmentActivity) {

    // Returns the total number of tabs
    override fun getItemCount(): Int = 2

    // Returns the fragment for each tab
    override fun createFragment(position: Int): Fragment {
        return when (position) {
            0 -> InputFragment()    // Input tab
            1 -> PredictFragment()  // Predict tab
            else -> throw IllegalStateException("Unexpected position $position")
        }
    }
}
