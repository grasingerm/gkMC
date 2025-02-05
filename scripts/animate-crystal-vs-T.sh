#!/bin/bash

# Create output directories if they don't exist
mkdir -p combined_images gif_output

# Combine matching crystal and temperature images
for f in graph_crystal-*.png; do
    # Extract the number from the filename
    num=$(echo $f | grep -oP 'graph_crystal-\K\d+')

    # Create zero-padded number (pad to 4 digits, adjust if you need more)
    padded_num=$(printf "%09d" $num)

    # Combine images side by side
    #convert +append graph_crystal-$num.png graph_temp-$num.png combined_images/combined-$num.png
    
    # Uncomment to combine images vertically
    convert -append graph_crystal-$num.png graph_temp-$num.png combined_images/combined-$padded_num.png
done

# Create animated gif from combined images
convert -delay 50 -loop 0 combined_images/combined-*.png gif_output/crystal_temperature_animation.gif
