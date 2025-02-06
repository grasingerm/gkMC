#!/bin/bash

# Create output directories if they don't exist
mkdir -p combined_images gif_output

# Combine matching crystal and temperature images
for f in graph_crystal-*.png; do
    # Extract the number from the filename
    num=$(echo $f | grep -oP 'graph_crystal-\K\d+')

    # Create zero-padded number (pad to 4 digits, adjust if you need more)
    paddedNum=$(printf "%09d" $num)

    # Combine images side by side
    #convert +append graph_crystal-$num.png graph_temp-$num.png combined_images/combined-$paddedNum.png
    
    # Uncomment to combine images vertically
    convert -append graph_crystal-$num.png graph_temp-$num.png combined_images/combined-$paddedNum.png
done

# Create animated gif from combined images
convert -delay 8 -loop 0 -layers optimize combined_images/combined-*.png gif_output/crystal_temperature_animation.gif
#convert -delay 50 -loop 0 combined_images/combined-*.png gif_output/crystal_temperature_animation.gif
