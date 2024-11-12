import matplotlib.pyplot as plt
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Generate a bar chart and save it to a file.")
parser.add_argument("filename", type=str, help="The filename to save the chart as (e.g., 'chart.png')")
args = parser.parse_args()

# Sample data
categories = ['Category A', 'Category B', 'Category C', 'Category D']
values = [23, 17, 35, 29]

# Create the bar chart
plt.figure(figsize=(10, 5), dpi=2000)  # Set the figure size and dpi for higher resolution
plt.bar(categories, values)

# Add labels and title
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Sample Bar Chart')

# Save the plot to the specified file
plt.savefig(args.filename)

# Optionally display the plot
# plt.show()
